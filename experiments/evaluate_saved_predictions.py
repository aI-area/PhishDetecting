"""Recompute binary phishing metrics from saved baseline predictions.

The positive class is always phishing (label 1). The script also records
weighted scores for audit purposes, but those scores are never used for the
common comparison table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
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


def cohort_hash(frame: pd.DataFrame) -> str:
    """Return an order-independent hash of URL/label pairs."""
    rows = (frame["url"].astype(str) + "\t" + frame["phishing"].astype(str)).sort_values()
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def evaluate(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    required = {"url", "phishing", "prediction", "probability"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    y_true = frame["phishing"].astype(int)
    y_pred = frame["prediction"].astype(int)
    y_score = frame["probability"].astype(float)
    if not set(y_true.unique()).issubset({0, 1}):
        raise ValueError(f"{path.name}: labels are not binary 0/1")
    if not set(y_pred.unique()).issubset({0, 1}):
        raise ValueError(f"{path.name}: predictions are not binary 0/1")
    if not y_score.between(0, 1).all():
        raise ValueError(f"{path.name}: probabilities fall outside [0, 1]")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    threshold_mismatches = int((y_pred != (y_score >= 0.5).astype(int)).sum())
    return {
        "file": path.name,
        "n": len(frame),
        "n_benign": int((y_true == 0).sum()),
        "n_phishing": int((y_true == 1).sum()),
        "cohort_sha256": cohort_hash(frame),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_binary": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_binary": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_binary": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision_weighted_audit": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted_audit": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted_audit": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "threshold_0_5_mismatches": threshold_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pattern", default="*_results.csv")
    args = parser.parse_args()

    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files match {args.pattern!r} in {args.input_dir}")

    rows = [evaluate(path) for path in files]
    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "common_binary_metrics.csv", index=False)
    with (args.output_dir / "evaluation_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "positive_class": 1,
                "positive_class_name": "phishing",
                "decision_threshold": 0.5,
                "input_directory": str(args.input_dir.resolve()),
                "files": [path.name for path in files],
            },
            handle,
            indent=2,
        )

    display = result[
        [
            "file",
            "n",
            "n_benign",
            "n_phishing",
            "accuracy",
            "precision_binary",
            "recall_binary",
            "f1_binary",
            "roc_auc",
            "average_precision",
            "mcc",
            "threshold_0_5_mismatches",
        ]
    ].copy()
    numeric = display.select_dtypes(include="number").columns.difference(
        ["n", "n_benign", "n_phishing", "threshold_0_5_mismatches"]
    )
    display[numeric] = display[numeric].round(6)
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
