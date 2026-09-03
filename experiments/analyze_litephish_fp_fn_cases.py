#!/usr/bin/env python3
"""Categorize LitePhish false positives and false negatives on the fixed test set."""

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
    p = successes / total; denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - radius, centre + radius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("split_cache", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("scaler", type=Path)
    parser.add_argument("pipeline_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.pipeline_root)); sys.path.insert(0, str(args.pipeline_root / "experiments"))
    from model import FocalLossLGBM
    setattr(sys.modules["__main__"], "FocalLossLGBM", FocalLossLGBM)
    from main import PhishingDetector
    from experiments.run_litephish_experiments import predict_probabilities
    from experiments.evaluate_litephish_failure_cases import natural_flags

    detector = PhishingDetector(); urls, labels = detector.load_data(str(args.dataset))
    _, _, test_idx = detector.split_data_indices(urls, labels)
    cache = joblib.load(args.split_cache); model = joblib.load(args.model); scaler = joblib.load(args.scaler)
    x_test = cache["x_test"]; y = labels.iloc[test_idx].to_numpy(dtype=int)
    test_urls = urls.iloc[test_idx].reset_index(drop=True)
    probability = np.asarray(predict_probabilities(model, scaler.transform(x_test)), dtype=float)
    prediction = (probability >= args.threshold).astype(int)
    flags = pd.DataFrame([natural_flags(url) for url in test_urls])

    def primary(row) -> str:
        if row.natural_shortener: return "shortener"
        if row.natural_shared_or_reputable_host: return "shared_or_reputable_host"
        if row.natural_obfuscated: return "obfuscated"
        if row.natural_clean_looking: return "clean_looking"
        return "other"

    frame = pd.DataFrame({"test_position": range(len(test_idx)), "dataset_row_id": test_idx,
                          "url": test_urls, "label": y, "probability": probability,
                          "prediction": prediction}).join(flags)
    frame["primary_category"] = flags.apply(primary, axis=1)
    frame["error_type"] = np.where((y == 1) & (prediction == 0), "false_negative",
                            np.where((y == 0) & (prediction == 1), "false_positive", "correct"))
    category_rows = []
    for category in ("all", "shortener", "shared_or_reputable_host", "obfuscated", "clean_looking", "other"):
        mask = np.ones(len(frame), dtype=bool) if category == "all" else frame.primary_category.eq(category).to_numpy()
        for label, measure in ((1, "false_negative_rate"), (0, "false_positive_rate")):
            cohort = mask & (y == label); n = int(cohort.sum())
            errors = int(((prediction[cohort] != y[cohort])).sum()) if n else 0
            lower, upper = wilson(errors, n)
            category_rows.append({"category": category, "class": "phishing" if label else "benign",
                                  "n": n, "errors": errors, "measure": measure,
                                  "error_rate": errors / n if n else float("nan"),
                                  "error_rate_ci95_low": lower, "error_rate_ci95_high": upper})
    category_path = args.output_dir / "fp_fn_category_rates.csv"
    pd.DataFrame(category_rows).to_csv(category_path, index=False)
    errors_path = args.output_dir / "misclassified_cases.csv.gz"
    frame.loc[frame.error_type != "correct"].to_csv(errors_path, index=False, compression="gzip")
    full_path = args.output_dir / "full_test_predictions_with_categories.csv.gz"
    frame.to_csv(full_path, index=False, compression="gzip")
    audit = {
        "status": "PASS", "analysis": "natural false-positive and false-negative categories",
        "dataset_sha256": sha256(args.dataset), "test_rows": len(frame),
        "phishing_rows": int((y == 1).sum()), "benign_rows": int((y == 0).sum()),
        "false_negatives": int(((y == 1) & (prediction == 0)).sum()),
        "false_positives": int(((y == 0) & (prediction == 1)).sum()),
        "threshold": args.threshold,
        "category_rule": "mutually exclusive priority: shortener, shared/reputable host, obfuscated, clean-looking, other",
        "category_definitions_predefined_in": "evaluate_litephish_failure_cases.py",
        "outputs": {path.name: sha256(path) for path in (category_path, errors_path, full_path)},
    }
    audit_path = args.output_dir / "fp_fn_analysis_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(pd.DataFrame(category_rows).to_string(index=False)); print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
