#!/usr/bin/env python3
"""Train the exact released LitePhish pipeline for cross-dataset evaluation.

This runner preserves the notebook's two-stage 5,000 n-gram representation,
XGBoost-based SDCS gain component, 1,000-feature ceiling, and 1,000-tree focal
LightGBM. Focal parameters are chosen only on the source validation partition;
target labels never influence model or parameter selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="latin1")
    if "url" not in frame and "URL" in frame:
        frame = frame.rename(columns={"URL": "url"})
    if "phishing" not in frame and "label" in frame:
        frame = frame.rename(columns={"label": "phishing"})
    if not {"url", "phishing"}.issubset(frame.columns):
        raise ValueError(f"{path} must contain url and phishing/label columns")
    frame = frame[["url", "phishing"]].copy()
    frame["url"] = frame["url"].astype(str)
    frame["phishing"] = frame["phishing"].astype(int)
    if not set(frame["phishing"].unique()).issubset({0, 1}):
        raise ValueError(f"{path} contains non-binary labels")
    return frame.reset_index(drop=True)


def materialize(dataset: pd.DataFrame, manifest_path: Path, allowed_splits: set[str]) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {"row_id", "label"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"{manifest_path} must contain {sorted(required)}")
    if "split" in manifest:
        manifest = manifest[manifest["split"].astype(str).isin(allowed_splits)].copy()
    ids = manifest["row_id"].astype(int).to_numpy()
    if len(ids) == 0 or ids.min() < 0 or ids.max() >= len(dataset):
        raise ValueError(f"invalid or empty row_id selection in {manifest_path}")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"duplicate row_id values in {manifest_path}")
    selected = dataset.iloc[ids].reset_index(drop=True)
    labels = manifest["label"].astype(int).to_numpy()
    if not np.array_equal(selected["phishing"].to_numpy(), labels):
        raise ValueError(f"label mismatch between {manifest_path} and dataset")
    selected.insert(0, "row_id", ids)
    if "split" in manifest:
        selected["split"] = manifest["split"].astype(str).to_numpy()
    return selected


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)), "positive_class": 1, "threshold": 0.5,
    }


def parse_pairs(values: list[str]) -> list[tuple[float, float]]:
    pairs = []
    for value in values:
        gamma, alpha = value.split(",", 1)
        pairs.append((float(gamma), float(alpha)))
    return pairs


def exact_xgboost_gain(x: np.ndarray, y: np.ndarray, seed: int, n_estimators: int = 200) -> np.ndarray:
    """Notebook SDCS gain component: full-data XGBoost feature importance."""
    model = XGBClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        tree_method="hist",
        device="cpu",
        n_jobs=-1,
        enable_categorical=False,
        verbosity=0,
    )
    model.fit(x, y)
    return np.asarray(model.feature_importances_, dtype=np.float64)


def exact_stability_frequency(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    stability_runs: int,
    sample_fraction: float,
    regularization_strength: float,
    n_jobs: int,
) -> np.ndarray:
    """Notebook stability-selection seed and L1 fitting procedure."""
    rng = np.random.RandomState(seed)
    seeds = [int(rng.randint(0, 20_000)) for _ in range(stability_runs)]

    def single_run(local_seed: int) -> np.ndarray:
        local_rng = np.random.RandomState(local_seed)
        size = int(sample_fraction * x.shape[0])
        idx = local_rng.choice(x.shape[0], size=size, replace=False)
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=regularization_strength,
            random_state=int(local_rng.randint(0, 5_000)),
            n_jobs=1,
        )
        model.fit(x[idx], y[idx])
        return (np.abs(model.coef_).ravel() > 1e-5).astype(np.int8)

    selected = Parallel(n_jobs=n_jobs)(delayed(single_run)(s) for s in seeds)
    return np.mean(np.vstack(selected), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dataset", type=Path)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--target", action="append", nargs=3, metavar=("NAME", "DATASET", "MANIFEST"), required=True)
    parser.add_argument("--candidate", action="append", required=True, help="gamma,alpha; repeat as needed")
    parser.add_argument("--pipeline-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument(
        "--allow-source-target-overlap",
        action="store_true",
        help="Reproduce the original full-target protocol while recording exact URL overlap.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.pipeline_root))
    sys.path.insert(0, str(args.pipeline_root / "experiments"))
    from experiments import run_litephish_experiments as suite

    # Use the specified XGBoost-gain and stability-selection implementations.
    suite.gain_scores = exact_xgboost_gain
    suite.stability_frequency = exact_stability_frequency

    source_full = read_dataset(args.source_dataset)
    source = materialize(source_full, args.source_manifest, {"train", "validation"})
    if "split" not in source:
        raise ValueError("source manifest must contain train and validation split labels")
    train = source[source["split"] == "train"].reset_index(drop=True)
    validation = source[source["split"] == "validation"].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("source manifest did not materialize both train and validation rows")

    target_frames = []
    target_audit = []
    source_train_urls = set(train["url"].astype(str))
    for name, dataset_value, manifest_value in args.target:
        dataset_path, manifest_path = Path(dataset_value), Path(manifest_value)
        target_full = read_dataset(dataset_path)
        target = materialize(target_full, manifest_path, {"test", "target_test"})
        overlap = source_train_urls.intersection(target["url"].astype(str))
        if overlap and not args.allow_source_target_overlap:
            raise ValueError(f"{name}: {len(overlap)} exact source-train/target URL overlaps")
        target["target_name"] = name
        target_frames.append(target)
        target_audit.append({
            "name": name, "dataset": str(dataset_path), "manifest": str(manifest_path),
            "dataset_sha256": sha256(dataset_path), "manifest_sha256": sha256(manifest_path),
            "rows": int(len(target)), "positives": int(target["phishing"].sum()),
            "exact_source_train_target_url_overlap": int(len(overlap)),
        })
    all_targets = pd.concat(target_frames, ignore_index=True)

    prepared = suite.PreparedData(
        train_urls=train["url"], val_urls=validation["url"], test_urls=all_targets["url"],
        train_y=train["phishing"], val_y=validation["phishing"], test_y=all_targets["phishing"],
        train_indices=train["row_id"].to_numpy(), val_indices=validation["row_id"].to_numpy(),
        test_indices=all_targets["row_id"].to_numpy(), mode="cross_manifest",
        train_path=str(args.source_dataset), test_path=";".join(x[1] for x in args.target),
    )
    pipeline_args = SimpleNamespace(
        outdir=str(args.output_dir), no_cache=False, n_jobs=args.jobs, max_ngram_features=None,
        score_sample_size=20000, selector_estimators=200, stability_runs=20,
        sample_fraction=0.905, regularization_strength=0.191, frequency_threshold=0.705,
        mi_weight=0.318, alpha_sdcs=0.767, correlation_threshold=0.956,
    )
    features = suite.build_features(prepared, pipeline_args, seed=42)
    selected, selector_meta = suite.select_features(
        features.x_train, train["phishing"].to_numpy(), features.feature_names,
        "sdcs", 42, 1000, pipeline_args,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features.x_train[:, selected].astype(np.float32))
    x_val = scaler.transform(features.x_val[:, selected].astype(np.float32))
    x_target = scaler.transform(features.x_test[:, selected].astype(np.float32))
    y_train = train["phishing"].to_numpy()
    y_val = validation["phishing"].to_numpy()

    candidates = []
    trained = {}
    for gamma, alpha in parse_pairs(args.candidate):
        model = suite.FocalLossLGBM(
            gamma=gamma,
            alpha=alpha,
            boosting_type="gbdt",
            num_leaves=50,
            max_depth=-1,
            learning_rate=0.15,
            n_estimators=1000,
            min_child_samples=40,
            reg_alpha=0.1,
            reg_lambda=0.3,
            random_state=42,
            device="cpu",
        )
        model.fit(x_train, y_train)
        val_probability = suite.predict_probabilities(model, x_val)
        metrics = binary_metrics(y_val, val_probability)
        candidates.append({"gamma": gamma, "alpha": alpha, **metrics})
        trained[(gamma, alpha)] = model
    candidates.sort(key=lambda item: (-item["f1"], -item["average_precision"], item["gamma"], item["alpha"]))
    chosen = candidates[0]
    chosen_pair = (chosen["gamma"], chosen["alpha"])
    model = trained[chosen_pair]
    target_probability = suite.predict_probabilities(model, x_target)

    cursor = 0
    scenario_metrics = []
    for target in target_frames:
        stop = cursor + len(target)
        probability = target_probability[cursor:stop]
        prediction = (probability >= 0.5).astype(int)
        name = str(target["target_name"].iloc[0])
        metrics = binary_metrics(target["phishing"].to_numpy(), probability)
        metrics.update({
            "baseline": "LitePhish", "source": args.source_name, "target": name,
            "scenario": f"{args.source_name}_to_{name}",
            "gamma": chosen_pair[0], "alpha": chosen_pair[1],
            "metric_definition": "binary, phishing class (1)",
        })
        scenario_dir = args.output_dir / f"{args.source_name}_to_{name}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "row_id": target["row_id"].to_numpy(), "url": target["url"].to_numpy(),
            "phishing": target["phishing"].to_numpy(), "prediction": prediction,
            "probability": probability,
        }).to_csv(scenario_dir / "LitePhish_predictions.csv", index=False)
        with (scenario_dir / "LitePhish_metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(metrics, stream, indent=2)
        scenario_metrics.append(metrics)
        cursor = stop

    joblib.dump(model, args.output_dir / "LitePhish_model.joblib")
    joblib.dump(scaler, args.output_dir / "LitePhish_scaler.joblib")
    with (args.output_dir / "LitePhish_preprocessing.pkl").open("wb") as stream:
        pickle.dump({
            "ngram_processor": features.ngram_processor,
            "selected_indices": selected,
            "feature_names": features.feature_names,
        }, stream, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(candidates).to_csv(args.output_dir / "source_validation_parameter_selection.csv", index=False)
    pd.DataFrame(scenario_metrics).to_csv(args.output_dir / "LitePhish_cross_transfer_summary.csv", index=False)
    audit = {
        "source_name": args.source_name, "source_dataset": str(args.source_dataset),
        "source_manifest": str(args.source_manifest), "source_dataset_sha256": sha256(args.source_dataset),
        "source_manifest_sha256": sha256(args.source_manifest), "train_rows": len(train),
        "validation_rows": len(validation), "targets": target_audit,
        "candidate_pairs": [{"gamma": g, "alpha": a} for g, a in parse_pairs(args.candidate)],
        "chosen_pair": {"gamma": chosen_pair[0], "alpha": chosen_pair[1]},
        "selection_rule": "maximum source-validation binary phishing-class F1; AP then lower gamma/alpha tie-break",
        "feature_count": int(len(selected)), "feature_budget": 1000,
        "selector": "SDCS (stability + MI + XGBoost gain + correlation pruning)",
        "selector_gain_model": "XGBoost", "n_estimators": 1000, "seed": 42,
        "selector_meta_recorded": bool(selector_meta),
    }
    with (args.output_dir / "LitePhish_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    print(json.dumps({"chosen": audit["chosen_pair"], "metrics": scenario_metrics}, indent=2))


if __name__ == "__main__":
    main()
