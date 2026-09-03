"""Aggregate and validate the nine-model, four-scenario transfer matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCENARIOS = (
    "PhishStorm_to_PhishFusion",
    "PhishStorm_to_Ebbu2017",
    "Ebbu2017_to_PhishFusion",
    "Ebbu2017_to_PhishStorm",
)
EXPERIMENTS = (
    ("MUDS", "cross_dataset/MUDS"),
    ("E2Phish", "cross_dataset/E2Phish"),
    ("Ebbu", "cross_dataset/Ebbu"),
    ("TabNet", "cross_dataset/TabNet"),
    ("StealthPhisher", "cross_dataset/StealthPhisher"),
    ("CNN-Fusion", "cross_dataset/CNN-Fusion"),
    ("DEPHIDES", "cross_dataset/DEPHIDES"),
    ("GramBeddings", "cross_dataset/GramBeddings"),
    ("URLNet", "cross_dataset/URLNet"),
)
FIELDS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "mcc",
)


def one_match(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.rglob(pattern)) if directory.exists() else []
    if len(matches) > 1:
        raise ValueError(f"Expected at most one {pattern} in {directory}: {matches}")
    return matches[0] if matches else None


def json_result(path: Path, baseline: str, scenario: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    split_metrics = payload["metrics"]
    metrics = split_metrics.get("target_test", split_metrics.get("test"))
    if metrics is None:
        raise KeyError(f"{path}: neither target_test nor test metrics exist")
    row: dict[str, object] = {
        "baseline": baseline,
        "scenario": scenario,
        "n": metrics["n"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
        "source": str(path.resolve()),
    }
    row.update({field: metrics[field] for field in FIELDS})
    return row


def urlnet_result(path: Path, scenario: str) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != 1:
        raise ValueError(f"Expected one URLNet record in {path}, found {len(records)}")
    item = records[0]
    return {
        "baseline": "URLNet",
        "scenario": scenario,
        "n": int(item["n"]),
        "tn": int(item["tn"]),
        "fp": int(item["fp"]),
        "fn": int(item["fn"]),
        "tp": int(item["tp"]),
        "accuracy": float(item["accuracy"]),
        "precision": float(item["precision_binary"]),
        "recall": float(item["recall_binary"]),
        "f1": float(item["f1_binary"]),
        "roc_auc": float(item["roc_auc"]),
        "average_precision": float(item["average_precision"]),
        "mcc": float(item["mcc"]),
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
        for scenario in SCENARIOS:
            directory = args.root / experiment / scenario
            if baseline == "URLNet":
                path = one_match(directory, "common_binary_metrics.csv")
                if path:
                    rows.append(urlnet_result(path, scenario))
                else:
                    missing.append(f"{directory}/**/common_binary_metrics.csv")
            else:
                path = one_match(directory, "*_metrics.json")
                if path:
                    rows.append(json_result(path, baseline, scenario))
                else:
                    missing.append(f"{directory}/**/*_metrics.json")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "baseline", "scenario", "n", "tn", "fp", "fn", "tp", *FIELDS, "source"
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    by_key = {(row["baseline"], row["scenario"]): row for row in rows}
    latex_lines: list[str] = []
    for baseline, _ in EXPERIMENTS:
        if not all((baseline, scenario) in by_key for scenario in SCENARIOS):
            continue
        cells: list[str] = []
        for scenario in SCENARIOS:
            row = by_key[(baseline, scenario)]
            cells.extend(f"{100.0 * float(row[field]):.2f}" for field in FIELDS[:5])
        latex_lines.append(f"{baseline} & " + " & ".join(cells) + r" \\")
    args.output.with_suffix(".tex").write_text(
        "\n".join(latex_lines) + ("\n" if latex_lines else ""), encoding="utf-8"
    )

    status = {
        "expected": len(EXPERIMENTS) * len(SCENARIOS),
        "complete": len(rows),
        "missing": missing,
        "positive_class": {"value": 1, "name": "phishing"},
        "decision_threshold": 0.5,
        "metric_definition": "binary, phishing class (1)",
        "scenario_order": list(SCENARIOS),
    }
    args.output.with_suffix(".status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
