"""Aggregate common domain-split baseline metrics into one validated table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASETS = ("PhishStorm", "Ebbu2017", "PhishFusion")
EXPERIMENTS = (
    ("MUDS", "internal/MUDS"),
    ("E2Phish", "internal/E2Phish"),
    ("Ebbu", "internal/Ebbu"),
    ("TabNet", "internal/TabNet"),
    ("StealthPhisher", "internal/StealthPhisher"),
    ("CNN-Fusion", "internal/CNN-Fusion"),
    ("DEPHIDES", "internal/DEPHIDES"),
    ("GramBeddings", "internal/GramBeddings"),
)


def from_json(path: Path, baseline: str, dataset: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]["test"]
    return {
        "baseline": baseline,
        "dataset": dataset,
        "n": metrics["n"],
        "benign": metrics["benign"],
        "phishing": metrics["phishing"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "average_precision": metrics["average_precision"],
        "mcc": metrics["mcc"],
        "source": str(path.resolve()),
    }


def from_urlnet(path: Path, dataset: str) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one URLNet result in {path}, found {len(rows)}")
    item = rows[0]
    integer = lambda key: int(item[key])
    number = lambda key: float(item[key])
    return {
        "baseline": "URLNet",
        "dataset": dataset,
        "n": integer("n"),
        "benign": integer("n_benign"),
        "phishing": integer("n_phishing"),
        "tn": integer("tn"),
        "fp": integer("fp"),
        "fn": integer("fn"),
        "tp": integer("tp"),
        "accuracy": number("accuracy"),
        "precision": number("precision_binary"),
        "recall": number("recall_binary"),
        "f1": number("f1_binary"),
        "roc_auc": number("roc_auc"),
        "average_precision": number("average_precision"),
        "mcc": number("mcc"),
        "source": str(path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for baseline, experiment in EXPERIMENTS:
        for dataset in DATASETS:
            path = args.root / experiment / dataset / f"{dataset}_metrics.json"
            if path.exists():
                rows.append(from_json(path, baseline, dataset))
            else:
                missing.append(str(path))

    for dataset in DATASETS:
        path = (
            args.root
            / "experiment_urlnet_shared_split"
            / dataset
            / "test"
            / "common_binary_metrics.csv"
        )
        if path.exists():
            rows.append(from_urlnet(path, dataset))
        else:
            missing.append(str(path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "baseline", "dataset", "n", "benign", "phishing", "tn", "fp",
        "fn", "tp", "accuracy", "precision", "recall", "f1", "roc_auc",
        "average_precision", "mcc", "source",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    status = {
        "expected": len(EXPERIMENTS) * len(DATASETS) + len(DATASETS),
        "complete": len(rows),
        "missing": missing,
        "positive_class": {"value": 1, "name": "phishing"},
        "metric_definition": "binary, phishing class (1), threshold 0.5",
    }
    args.output.with_suffix(".status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
