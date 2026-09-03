#!/usr/bin/env python3
"""Evaluate a frozen released LitePhish artifact on a positive temporal cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return centre - radius, centre + radius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort_csv", type=Path)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("pipeline_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model-freeze-utc", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.pipeline_root))
    sys.path.insert(0, str(args.pipeline_root / "experiments"))
    from model import FocalLossLGBM
    setattr(sys.modules["__main__"], "FocalLossLGBM", FocalLossLGBM)
    from feature_extraction import FeatureExtractor
    from experiments.run_litephish_experiments import predict_probabilities

    paths = {name: args.artifact_dir / name for name in
             ("model.pkl", "scaler.pkl", "ngram_processor.pkl", "selected_features.pkl",
              "all_feature_names.pkl", "predictions.pkl", "test_labels.pkl")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    model = joblib.load(paths["model.pkl"])
    scaler = joblib.load(paths["scaler.pkl"])
    ngrams = joblib.load(paths["ngram_processor.pkl"])
    selected = np.asarray(joblib.load(paths["selected_features.pkl"]), dtype=int)

    cohort = pd.read_csv(args.cohort_csv)
    if len(cohort) == 0 or not (cohort["phishing"] == 1).all():
        raise ValueError("temporal cohort must be nonempty and all-positive")
    extractor = FeatureExtractor()
    probabilities: list[np.ndarray] = []
    for start in range(0, len(cohort), args.batch_size):
        urls = cohort["url"].iloc[start:start + args.batch_size].astype(str)
        handcrafted = np.asarray([extractor.generate_features(url) for url in urls], dtype=np.float32)
        ngram = ngrams.transform(urls).toarray().astype(np.float32, copy=False)
        combined = np.hstack([handcrafted, ngram]).astype(np.float32, copy=False)
        prepared = scaler.transform(combined[:, selected])
        probabilities.append(np.asarray(predict_probabilities(model, prepared), dtype=float))
    probability = np.concatenate(probabilities)
    prediction = (probability >= args.threshold).astype(int)
    detected = int(prediction.sum())
    lower, upper = wilson(detected, len(prediction))

    old_probability = np.asarray(joblib.load(paths["predictions.pkl"]), dtype=float).reshape(-1)
    old_labels = np.asarray(joblib.load(paths["test_labels.pkl"]), dtype=int).reshape(-1)
    if len(old_probability) != len(old_labels):
        raise ValueError("released internal predictions and labels differ in length")
    old_positive = old_labels == 1
    old_detected = int((old_probability[old_positive] >= args.threshold).sum())
    old_total = int(old_positive.sum())
    old_lower, old_upper = wilson(old_detected, old_total)

    predictions = cohort.copy()
    predictions["probability"] = probability
    predictions["prediction"] = prediction
    pred_path = args.output_dir / "LitePhish_temporal_predictions.csv"
    predictions.to_csv(pred_path, index=False)
    misses_path = args.output_dir / "LitePhish_temporal_false_negatives.csv"
    predictions.loc[prediction == 0].to_csv(misses_path, index=False)

    artifact_hashes = {name: sha256(path) for name, path in paths.items()}
    result = {
        "experiment": "frozen LitePhish prospective temporal generalization",
        "model_freeze_utc": args.model_freeze_utc,
        "retraining_performed": False,
        "threshold": args.threshold,
        "positive_class": {"value": 1, "name": "phishing"},
        "cohort_csv": str(args.cohort_csv.resolve()),
        "cohort_sha256": sha256(args.cohort_csv),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "artifact_sha256": artifact_hashes,
        "selected_feature_count": int(len(selected)),
        "temporal_cohort": {
            "n_positive": int(len(prediction)), "true_positives": detected,
            "false_negatives": int(len(prediction) - detected),
            "recall": detected / len(prediction),
            "recall_wilson_95_ci": [lower, upper],
            "mean_probability": float(probability.mean()),
            "median_probability": float(np.median(probability)),
        },
        "released_internal_test_positive_reference": {
            "n_positive": old_total, "true_positives": old_detected,
            "false_negatives": old_total - old_detected,
            "recall": old_detected / old_total,
            "recall_wilson_95_ci": [old_lower, old_upper],
            "scope": "in-distribution reference only; not an independent temporal cohort",
        },
        "recall_difference_temporal_minus_internal": detected / len(prediction) - old_detected / old_total,
        "metric_scope": "recall/detection rate only because the prospective cohort contains only confirmed phishing URLs",
        "predictions_csv": str(pred_path.resolve()),
        "predictions_sha256": sha256(pred_path),
        "false_negatives_csv": str(misses_path.resolve()),
        "false_negatives_sha256": sha256(misses_path),
    }
    (args.output_dir / "LitePhish_temporal_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
