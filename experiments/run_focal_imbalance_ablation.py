#!/usr/bin/env python3
"""Matched focal-loss, class-weighting, and threshold ablation for LitePhish."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from scipy import sparse
from scipy.stats import t as student_t
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def tuned_f1_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, probability)
    if len(thresholds) == 0:
        return 0.5
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    best = np.flatnonzero(np.isclose(scores, np.nanmax(scores), rtol=0, atol=1e-12))
    return float(thresholds[best[np.argmin(np.abs(thresholds[best] - 0.5))]])


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold), "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def common_params(seed: int, estimators: int, threads: int) -> dict[str, object]:
    return {
        "boosting_type": "gbdt", "num_leaves": 50, "max_depth": -1,
        "learning_rate": 0.15, "n_estimators": estimators, "min_child_samples": 40,
        "reg_alpha": 0.1, "reg_lambda": 0.3, "verbosity": -1,
        "random_state": seed, "device": "cpu", "n_jobs": threads,
        "deterministic": True, "force_col_wise": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("released_artifacts", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--feature-jobs", type=int, default=24)
    parser.add_argument("--outer-workers", type=int, default=4)
    parser.add_argument("--model-threads", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.model_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.path.insert(0, str(args.pipeline_root))
    from feature_extraction import FeatureExtractor
    from main import PhishingDetector
    from model import FocalLossLGBM
    from ngram_processing import NgramProcessor
    setattr(sys.modules["__main__"], "NgramProcessor", NgramProcessor)
    setattr(sys.modules["__main__"], "FocalLossLGBM", FocalLossLGBM)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor_path = args.released_artifacts / "ngram_processor.pkl"
    selected_path = args.released_artifacts / "selected_features.pkl"
    processor = joblib.load(processor_path)
    selected = np.asarray(joblib.load(selected_path), dtype=int)
    if len(selected) != 568:
        raise RuntimeError(f"expected 568 selected features, got {len(selected)}")

    detector = PhishingDetector()
    urls, labels = detector.load_data(str(args.dataset))
    train_idx, validation_idx, test_idx = detector.split_data_indices(urls, labels)
    y_all = labels.to_numpy(dtype=int)
    y_train, y_validation, y_test = y_all[train_idx], y_all[validation_idx], y_all[test_idx]

    cache_path = args.output_dir / "compact_568_split_cache.joblib"
    extraction_start = time.perf_counter()
    if cache_path.is_file():
        cache = joblib.load(cache_path)
        if cache["dataset_sha256"] != sha256(args.dataset):
            raise RuntimeError("feature cache dataset hash mismatch")
        x_train, x_validation, x_test = cache["x_train"], cache["x_validation"], cache["x_test"]
        extraction_seconds = 0.0
        cache_reused = True
    else:
        extractor = FeatureExtractor()
        handcrafted = np.asarray(Parallel(n_jobs=args.feature_jobs, prefer="threads")(
            delayed(extractor.generate_features)(url) for url in urls), dtype=np.float32)
        if handcrafted.shape[1] != 86:
            raise RuntimeError(f"expected 86 handcrafted features, got {handcrafted.shape[1]}")

        def compact(indices: np.ndarray) -> np.ndarray:
            combined = sparse.hstack([
                sparse.csr_matrix(handcrafted[indices]), processor.transform(urls.iloc[indices])
            ], format="csr", dtype=np.float32)
            return combined[:, selected].toarray().astype(np.float32, copy=False)

        x_train, x_validation, x_test = compact(train_idx), compact(validation_idx), compact(test_idx)
        extraction_seconds = time.perf_counter() - extraction_start
        cache_reused = False
        joblib.dump({
            "dataset_sha256": sha256(args.dataset), "processor_sha256": sha256(processor_path),
            "selected_sha256": sha256(selected_path), "train_idx_sha256": array_sha256(train_idx),
            "validation_idx_sha256": array_sha256(validation_idx), "test_idx_sha256": array_sha256(test_idx),
            "x_train": x_train, "x_validation": x_validation, "x_test": x_test,
        }, cache_path, compress=3)
    if (x_train.shape, x_validation.shape, x_test.shape) != ((len(train_idx), 568), (len(validation_idx), 568), (len(test_idx), 568)):
        raise RuntimeError("compact cache has unexpected shapes")

    # Select only among the author-confirmed PhishFusion candidate pairs, using
    # threshold-free validation AP. The selected pair is then frozen for every ratio.
    full_scaler = StandardScaler().fit(x_train)
    full_train_scaled = full_scaler.transform(x_train)
    validation_full_scaled = full_scaler.transform(x_validation)
    parameter_grid = []
    for gamma in (0.55, 0.60):
        for alpha in (0.90, 0.95):
            model = FocalLossLGBM(gamma=gamma, alpha=alpha,
                                  **common_params(42, args.n_estimators, args.model_threads))
            start = time.perf_counter()
            model.fit(full_train_scaled, y_train)
            probability = model.predict_proba(validation_full_scaled)[:, 1]
            parameter_grid.append({
                "gamma": gamma, "alpha": alpha,
                "validation_average_precision": float(average_precision_score(y_validation, probability)),
                "validation_roc_auc": float(roc_auc_score(y_validation, probability)),
                "validation_f1_at_0_5": float(f1_score(y_validation, probability >= 0.5)),
                "training_seconds": time.perf_counter() - start,
            })
    chosen = max(parameter_grid, key=lambda row: (row["validation_average_precision"], row["validation_roc_auc"], -row["gamma"], -row["alpha"]))
    gamma, alpha = chosen["gamma"], chosen["alpha"]
    del full_train_scaled, validation_full_scaled, full_scaler

    seeds = [int(value) for value in args.seeds.split(",")]
    benign_positions = np.flatnonzero(y_train == 0)
    phishing_positions = np.flatnonzero(y_train == 1)
    tasks: list[tuple[str, int, np.ndarray]] = [("natural", 42, np.arange(len(y_train)))]
    for ratio_name, denominator in (("1_to_3", 3), ("1_to_5", 5), ("1_to_10", 10)):
        target_positive = len(benign_positions) // denominator
        for seed in seeds:
            rng = np.random.default_rng(seed)
            sampled_positive = rng.choice(phishing_positions, size=target_positive, replace=False)
            subset = np.concatenate([benign_positions, sampled_positive])
            rng.shuffle(subset)
            tasks.append((ratio_name, seed, subset))

    rows: list[dict[str, object]] = []
    sensitivity: list[dict[str, object]] = []
    task_dir = args.output_dir / "task_checkpoints"
    task_dir.mkdir(exist_ok=True)

    def run_task(task: tuple[str, int, np.ndarray]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        ratio_name, seed, subset = task
        scaler = StandardScaler().fit(x_train[subset])
        train_scaled = scaler.transform(x_train[subset])
        validation_scaled = scaler.transform(x_validation)
        test_scaled = scaler.transform(x_test)
        base = common_params(seed, args.n_estimators, args.model_threads)
        models = {
            "focal": FocalLossLGBM(gamma=gamma, alpha=alpha, **base),
            "standard": LGBMClassifier(objective="binary", **base),
            "class_weight_balanced": LGBMClassifier(objective="binary", class_weight="balanced", **base),
        }
        local_rows, local_sensitivity = [], []
        for method, model in models.items():
            start = time.perf_counter()
            model.fit(train_scaled, y_train[subset])
            training_seconds = time.perf_counter() - start
            validation_probability = model.predict_proba(validation_scaled)[:, 1]
            test_probability = model.predict_proba(test_scaled)[:, 1]
            tuned = tuned_f1_threshold(y_validation, validation_probability)
            shared = {
                "imbalance_ratio": ratio_name, "seed": seed, "method": method,
                "train_rows": int(len(subset)), "train_benign": int((y_train[subset] == 0).sum()),
                "train_phishing": int((y_train[subset] == 1).sum()),
                "train_positive_prevalence": float(y_train[subset].mean()),
                "training_seconds": training_seconds,
            }
            for strategy, threshold in (("fixed_0_5", 0.5), ("validation_f1_tuned", tuned)):
                local_rows.append({**shared, "threshold_strategy": strategy,
                                   **metrics(y_test, test_probability, threshold)})
            for threshold in np.arange(0.1, 1.0, 0.1):
                local_sensitivity.append({**shared, **metrics(y_test, test_probability, float(threshold))})
        return local_rows, local_sensitivity

    def checkpoint_path(task: tuple[str, int, np.ndarray]) -> Path:
        return task_dir / f"{task[0]}_seed{task[1]}.joblib"

    pending = []
    for task in tasks:
        path = checkpoint_path(task)
        if path.is_file():
            saved = joblib.load(path)
            rows.extend(saved["rows"]); sensitivity.extend(saved["sensitivity"])
            print(f"reused checkpoint: {task[:2]}", flush=True)
        else:
            pending.append(task)

    def run_and_checkpoint(task: tuple[str, int, np.ndarray]):
        task_rows, task_sensitivity = run_task(task)
        path = checkpoint_path(task)
        temporary = path.with_suffix(".tmp")
        joblib.dump({"rows": task_rows, "sensitivity": task_sensitivity}, temporary)
        temporary.replace(path)
        return task_rows, task_sensitivity

    with ThreadPoolExecutor(max_workers=args.outer_workers) as executor:
        futures = {executor.submit(run_and_checkpoint, task): task[:2] for task in pending}
        for completed, future in enumerate(as_completed(futures), 1):
            task_rows, task_sensitivity = future.result()
            rows.extend(task_rows); sensitivity.extend(task_sensitivity)
            print(f"completed {completed}/{len(pending)} pending: {futures[future]}", flush=True)

    detailed = pd.DataFrame(rows).sort_values(["imbalance_ratio", "seed", "method", "threshold_strategy"])
    detailed_path = args.output_dir / "focal_imbalance_detailed.csv"
    detailed.to_csv(detailed_path, index=False)
    sensitivity_path = args.output_dir / "threshold_sensitivity.csv"
    pd.DataFrame(sensitivity).sort_values(["imbalance_ratio", "seed", "method", "threshold"]).to_csv(sensitivity_path, index=False)

    summary_rows = []
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "mcc"]
    for keys, group in detailed.groupby(["imbalance_ratio", "method", "threshold_strategy"], sort=True):
        record = dict(zip(("imbalance_ratio", "method", "threshold_strategy"), keys))
        record["runs"] = len(group)
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            mean, std = float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half = float(student_t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
            record[f"{metric}_mean"] = mean; record[f"{metric}_sd"] = std
            record[f"{metric}_ci95_low"] = mean - half; record[f"{metric}_ci95_high"] = mean + half
        summary_rows.append(record)
    summary_path = args.output_dir / "focal_imbalance_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    audit = {
        "status": "PASS", "experiment": "focal loss versus standard/class-weighted LightGBM",
        "dataset": str(args.dataset.resolve()), "dataset_sha256": sha256(args.dataset), "dataset_rows": len(urls),
        "split": {"method": "released domain-stratified split", "train": len(train_idx),
                  "validation": len(validation_idx), "test": len(test_idx),
                  "train_idx_sha256": array_sha256(train_idx), "validation_idx_sha256": array_sha256(validation_idx),
                  "test_idx_sha256": array_sha256(test_idx)},
        "features": {"count": 568, "processor_sha256": sha256(processor_path),
                     "selected_indices_sha256": sha256(selected_path), "cache": str(cache_path.resolve()),
                     "cache_sha256": sha256(cache_path), "cache_reused": cache_reused,
                     "feature_extraction_seconds": extraction_seconds},
        "focal_parameter_selection": {"candidate_source": "author-confirmed PhishFusion ranges",
                                      "criterion": "maximum validation average precision; test set untouched",
                                      "candidates": parameter_grid, "selected": {"gamma": gamma, "alpha": alpha}},
        "design": {"ratios": ["natural", "1_to_3", "1_to_5", "1_to_10"],
                   "subsampling": "retain all benign training rows; seeded sampling without replacement of phishing rows",
                   "seeds": seeds, "methods": ["focal", "standard", "class_weight_balanced"],
                   "thresholds": ["fixed 0.5", "validation F1 tuned"],
                   "validation_role": "parameter/threshold selection only", "test_role": "final evaluation only",
                   "positive_class": {"value": 1, "name": "phishing"}, "n_estimators": args.n_estimators},
        "software": {"python": platform.python_version()},
        "outputs": {"detailed": str(detailed_path.resolve()), "detailed_sha256": sha256(detailed_path),
                    "summary": str(summary_path.resolve()), "summary_sha256": sha256(summary_path),
                    "threshold_sensitivity": str(sensitivity_path.resolve()),
                    "threshold_sensitivity_sha256": sha256(sensitivity_path)},
    }
    audit_path = args.output_dir / "experiment_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
