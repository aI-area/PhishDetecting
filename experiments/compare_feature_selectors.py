#!/usr/bin/env python3
"""Matched comparison of SDCS with established feature-selection strategies.

The experiment uses one fixed domain-disjoint train/validation/test partition.
The structured N-gram stage is fitted once using source training data only and
is shared by every selector.  Ten seeded 90.5% training subsamples perturb the
selector-fitting data.  For each seed, SDCS determines its natural feature
count (up to 1,000), and every comparator is constrained to that same count.
The downstream focal-LightGBM model is otherwise identical across methods and
is fitted on the complete source-training partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


METHODS = ("sdcs", "mutual_information", "anova_f", "l1_logistic", "lgbm_gain", "mrmr_pearson")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(values.max(initial=0.0))
    return values / maximum if maximum > 0 else values


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
    return np.argsort(scores, kind="stable")[::-1][:k].astype(int)


def mi_scores(x: np.ndarray, y: np.ndarray, seed: int, cap: int = 20000) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if len(y) > cap:
        idx = rng.choice(len(y), cap, replace=False)
        x_use, y_use = x[idx], y[idx]
    else:
        x_use, y_use = x, y
    return mutual_info_classif(x_use, y_use, random_state=seed, discrete_features=False)


def select_mrmr_pearson(
    x: np.ndarray,
    relevance: np.ndarray,
    k: int,
    seed: int,
    sample_cap: int = 20000,
    prefilter: int = 1500,
) -> np.ndarray:
    """Greedy relevance-minus-redundancy selection with Pearson redundancy.

    Candidates are first restricted to the strongest MI features for tractable
    correlation estimation.  At each step, the selected feature maximizes
    normalized MI relevance minus its largest absolute correlation with the
    already selected set.  This fully specified variant is reported as
    mRMR-Pearson rather than treated as an exact reproduction of a particular
    software package.
    """
    candidate = top_k(relevance, min(max(prefilter, k), x.shape[1]))
    rng = np.random.RandomState(seed)
    if x.shape[0] > sample_cap:
        rows = rng.choice(x.shape[0], sample_cap, replace=False)
        z = x[rows][:, candidate]
    else:
        z = x[:, candidate]
    z = np.asarray(z, dtype=np.float32)
    z -= z.mean(axis=0, keepdims=True)
    scale = z.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    z /= scale
    corr = np.abs(np.corrcoef(z, rowvar=False)).astype(np.float32, copy=False)
    np.nan_to_num(corr, copy=False)
    rel = normalize(relevance[candidate])
    chosen_local: list[int] = []
    available = np.ones(len(candidate), dtype=bool)
    redundancy = np.zeros(len(candidate), dtype=np.float32)
    for _ in range(min(k, len(candidate))):
        score = rel - redundancy
        score[~available] = -np.inf
        j = int(np.argmax(score))
        if not np.isfinite(score[j]):
            break
        chosen_local.append(j)
        available[j] = False
        redundancy = np.maximum(redundancy, corr[j])
    return candidate[np.asarray(chosen_local, dtype=int)]


def redundancy_summary(x: np.ndarray, selected: np.ndarray, seed: int, cap: int = 20000) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    rows = rng.choice(x.shape[0], min(cap, x.shape[0]), replace=False)
    corr = np.abs(np.corrcoef(x[rows][:, selected], rowvar=False))
    upper = corr[np.triu_indices_from(corr, k=1)]
    upper = upper[np.isfinite(upper)]
    if upper.size == 0:
        return {"mean_abs_correlation": 0.0, "median_abs_correlation": 0.0,
                "p95_abs_correlation": 0.0, "max_abs_correlation": 0.0}
    return {
        "mean_abs_correlation": float(np.mean(upper)),
        "median_abs_correlation": float(np.median(upper)),
        "p95_abs_correlation": float(np.quantile(upper, 0.95)),
        "max_abs_correlation": float(np.max(upper)),
    }


def evaluate(model, x: np.ndarray, y: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    prob = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "mcc": float(matthews_corrcoef(y, pred)),
    }
    return metrics, pred, prob


def pairwise_jaccard(sets: list[set[str]]) -> list[float]:
    values = []
    for left, right in combinations(sets, 2):
        union = left | right
        values.append(len(left & right) / len(union) if union else 1.0)
    return values


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--sample-fraction", type=float, default=0.905)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.path.insert(0, str(args.pipeline_root))
    from feature_extraction import FeatureExtractor
    from feature_selection import FeatureSelector
    from model import FocalLossLGBM
    from ngram_processing import NgramProcessor

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = pd.read_csv(args.split_manifest).sort_values("row_id")
    data = pd.read_csv(args.dataset, encoding="latin1")
    if len(data) != len(manifest) or not np.array_equal(manifest.row_id.to_numpy(), np.arange(len(data))):
        raise RuntimeError("dataset and split manifest do not align by row_id")
    label_col = "phishing" if "phishing" in data.columns else "label"
    y_all = data[label_col].astype(int).to_numpy()
    if not np.array_equal(y_all, manifest.label.astype(int).to_numpy()):
        raise RuntimeError("dataset labels do not match split manifest")
    urls = data.url.astype(str)
    train_idx = np.flatnonzero(manifest.split.to_numpy() == "train")
    val_idx = np.flatnonzero(manifest.split.to_numpy() == "validation")
    test_idx = np.flatnonzero(manifest.split.to_numpy() == "test")
    if (len(data),len(train_idx),len(val_idx),len(test_idx)) != (105752,86098,8278,11376):
        raise RuntimeError('Not the canonical LitePhish source partition')
    expected_test='b37ff11d8baf705026dbe833e65a9425c3d2822cabcebc5ed9ed300504082921'
    if hashlib.sha256(test_idx.astype(np.int64).tobytes()).hexdigest()!=expected_test:
        raise RuntimeError('Held-out row order differs from LitePhish')
    import shutil
    shutil.copy2(__file__,args.output_dir/'runner_snapshot.py')
    pd.DataFrame({'row_id':train_idx}).to_csv(args.output_dir/'training_rows.csv',index=False)
    pd.DataFrame({'row_id':test_idx,'label':y_all[test_idx]}).to_csv(args.output_dir/'test_rows.csv',index=False)

    # Shared, training-fitted structured N-gram representation.
    processor = NgramProcessor()
    processor.fit_vectorizer(urls.iloc[train_idx])
    processor.select_features(urls.iloc[train_idx], pd.Series(y_all[train_idx]))
    joblib.dump(processor,args.output_dir/'ngram_processor.joblib')
    extractor = FeatureExtractor()
    handcrafted = Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(extractor.generate_features)(url) for url in urls
    )
    handcrafted = np.asarray(handcrafted, dtype=np.float32)
    ngram = processor.transform(urls).toarray().astype(np.float32, copy=False)
    x_all = np.hstack([handcrafted, ngram]).astype(np.float32, copy=False)
    del handcrafted, ngram
    feature_names = list(extractor.FEATURE_NAMES) + list(processor.get_feature_names())
    if x_all.shape[1] != len(feature_names):
        raise RuntimeError("feature-name alignment failure")
    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_test, y_test = x_all[test_idx], y_all[test_idx]
    del x_all

    selector = FeatureSelector()
    rows: list[dict[str, object]] = []
    selected_by_method: dict[str, list[set[str]]] = {method: [] for method in METHODS}
    prediction_parts: list[pd.DataFrame] = []
    seeds = args.seeds[:2] if args.pilot else args.seeds

    for seed in seeds:
        rng = np.random.RandomState(seed)
        subset = np.sort(rng.choice(len(y_train), int(args.sample_fraction * len(y_train)), replace=False))
        x_select, y_select = x_train[subset], y_train[subset]

        start = time.perf_counter()
        sdcs_idx = selector.perform_merged_selection(
            pd.DataFrame(x_select, columns=feature_names),
            pd.Series(y_select),
            stability_runs=20,
            sample_fraction=0.905,
            frequency_threshold=0.705,
            regularization_strength=0.191,
            mi_weight=0.318,
            alpha=0.767,
            num_features=1000,
            correlation_threshold=0.956,
            sample_size=20000,
            random_seed=seed,
            n_estimators=200,
        )
        sdcs_seconds = time.perf_counter() - start
        if not 1 <= len(sdcs_idx) <= 1000:
            raise RuntimeError(f"invalid SDCS feature count for seed {seed}: {len(sdcs_idx)}")
        k = len(sdcs_idx)

        score_start = time.perf_counter()
        mi = mi_scores(x_select, y_select, seed)
        mi_seconds = time.perf_counter() - score_start

        selection: dict[str, tuple[np.ndarray, float]] = {"sdcs": (sdcs_idx, sdcs_seconds)}
        selection["mutual_information"] = (top_k(mi, k), mi_seconds)

        start = time.perf_counter()
        f_scores, _ = f_classif(x_select, y_select)
        selection["anova_f"] = (top_k(f_scores, k), time.perf_counter() - start)

        start = time.perf_counter()
        l1 = LogisticRegression(
            penalty="l1", solver="liblinear", C=0.191, random_state=seed,
            max_iter=300, n_jobs=1,
        )
        l1.fit(x_select, y_select)
        selection["l1_logistic"] = (top_k(np.abs(l1.coef_[0]), k), time.perf_counter() - start)

        start = time.perf_counter()
        gain = LGBMClassifier(
            n_estimators=200, importance_type="gain", random_state=seed,
            n_jobs=args.jobs, verbosity=-1,
        )
        gain.fit(x_select, y_select)
        selection["lgbm_gain"] = (top_k(gain.feature_importances_, k), time.perf_counter() - start)

        start = time.perf_counter()
        mrmr_idx = select_mrmr_pearson(x_select, mi, k, seed)
        selection["mrmr_pearson"] = (mrmr_idx, mi_seconds + time.perf_counter() - start)

        for method in METHODS:
            chosen, selection_seconds = selection[method]
            if len(chosen) != k or len(np.unique(chosen)) != k:
                raise RuntimeError(f"{method} did not return exactly {k} unique features")
            selected_names = [feature_names[i] for i in chosen]
            selected_by_method[method].append(set(selected_names))
            scaler = StandardScaler()
            xtr = scaler.fit_transform(x_train[:, chosen])
            xte = scaler.transform(x_test[:, chosen])
            model = FocalLossLGBM(
                gamma=0.30, alpha=0.95, boosting_type="gbdt", num_leaves=50,
                max_depth=-1, learning_rate=0.15, n_estimators=500,
                min_child_samples=40, reg_alpha=0.1, reg_lambda=0.3,
                verbosity=-1, random_state=42, device="cpu", n_jobs=args.jobs,
            )
            fit_start = time.perf_counter()
            model.fit(xtr, y_train)
            fit_seconds = time.perf_counter() - fit_start
            metric_values, pred, prob = evaluate(model, xte, y_test)
            redundancy = redundancy_summary(x_train, chosen, seed)
            row: dict[str, object] = {
                "seed": seed, "method": method, "feature_count": k,
                "selection_seconds": selection_seconds, "model_fit_seconds": fit_seconds,
                **metric_values, **redundancy,
            }
            rows.append(row)
            prediction_parts.append(pd.DataFrame({
                "row_id": test_idx, "label": y_test, "seed": seed, "method": method,
                "prediction": pred, "probability": prob,
            }))
            method_dir = args.output_dir / "selected_features" / f"seed_{seed}"
            method_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"feature_index": chosen, "feature_name": selected_names}).to_csv(
                method_dir / f"{method}.csv", index=False
            )
            model.booster_.save_model(str(method_dir/f'{method}_booster.txt'))
            joblib.dump(scaler,method_dir/f'{method}_scaler.joblib',compress=3)
            del xtr, xte, model, scaler

        pd.DataFrame(rows).to_csv(args.output_dir/f'seed_{seed}_metrics_checkpoint.csv',index=False)
        pd.concat(prediction_parts[-len(METHODS):],ignore_index=True).to_csv(
            args.output_dir/f'seed_{seed}_predictions.csv.gz',index=False)
        print(json.dumps(dict(seed=seed,status='SEED_COMPLETE',methods=6,feature_budget=k)),flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    pd.concat(prediction_parts, ignore_index=True).to_csv(args.output_dir / "row_level_predictions.csv.gz", index=False)

    summary_rows = []
    for method, group in raw.groupby("method", sort=False):
        record: dict[str, object] = {"method": method, "runs": len(group)}
        for metric in ("feature_count", "accuracy", "precision", "recall", "f1", "roc_auc", "mcc",
                       "mean_abs_correlation", "p95_abs_correlation", "selection_seconds", "model_fit_seconds"):
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        jaccard = pairwise_jaccard(selected_by_method[method])
        record["pairwise_jaccard_mean"] = float(np.mean(jaccard)) if jaccard else 1.0
        record["pairwise_jaccard_sd"] = float(np.std(jaccard, ddof=1)) if len(jaccard) > 1 else 0.0
        summary_rows.append(record)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "summary.csv", index=False)

    tests = []
    for metric in ("f1", "roc_auc", "mcc"):
        base = raw[raw.method == "sdcs"].sort_values("seed").set_index("seed")[metric]
        pending = []
        for method in METHODS[1:]:
            other = raw[raw.method == method].sort_values("seed").set_index("seed")[metric]
            common = base.index.intersection(other.index)
            delta = base.loc[common].to_numpy() - other.loc[common].to_numpy()
            try:
                p = float(stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
            except ValueError:
                p = 1.0
            pending.append({
                "metric": metric, "comparison": f"sdcs_vs_{method}", "n_pairs": len(common),
                "mean_delta": float(np.mean(delta)), "median_delta": float(np.median(delta)), "p_raw": p,
            })
        adjusted = holm_adjust([item["p_raw"] for item in pending])
        for item, p_adj in zip(pending, adjusted):
            item["p_holm_within_metric"] = p_adj
            tests.append(item)
    pd.DataFrame(tests).to_csv(args.output_dir / "paired_wilcoxon.csv", index=False)

    audit = {
        "status": "PASS",
        "runner_sha256":sha256(Path(__file__)),
        "scope": "conditional comparison on one fixed domain-disjoint PhishFusion split",
        "dataset": {"path": str(args.dataset), "sha256": sha256(args.dataset), "rows": len(data)},
        "split_manifest": {"path": str(args.split_manifest), "sha256": sha256(args.split_manifest)},
        "partition_rows": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "shared_feature_pool": {"handcrafted": 86, "selected_ngrams": len(processor.get_feature_names()),
                                "total": len(feature_names)},
        "design": {
            "seeds": seeds, "selector_subsample_fraction": args.sample_fraction,
            "feature_budget": "matched within seed to natural SDCS output",
            "downstream_model": "identical focal-LightGBM; fixed threshold 0.5",
            "test_labels_used_for_selection": False,
            "statistical_scope": "Wilcoxon tests quantify paired selector perturbations conditional on the fixed test cohort",
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    with (args.output_dir / "audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    joblib.dump(processor, args.output_dir / "ngram_processor.joblib")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
