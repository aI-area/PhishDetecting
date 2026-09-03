#!/usr/bin/env python3
"""Aggregate and validate the cross-dataset experiment results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


BASELINES = [
    "E2Phish", "MUDS", "StealthPhisher", "TabNet", "Ebbu", "DEPHIDES",
    "CNN-Fusion", "GramBeddings", "URLNet", "LitePhish",
]
SCENARIOS = [
    "PhishStorm_to_PhishFusion", "PhishStorm_to_Ebbu2017",
    "Ebbu2017_to_PhishFusion", "Ebbu2017_to_PhishStorm",
]
EXPECTED_N = {
    "PhishStorm_to_PhishFusion": 105752,
    "PhishStorm_to_Ebbu2017": 73575,
    "Ebbu2017_to_PhishFusion": 105752,
    "Ebbu2017_to_PhishStorm": 95911,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(block: dict) -> dict:
    def get(*names):
        for name in names:
            if name in block:
                return block[name]
        raise KeyError(names)

    return {
        "n": int(get("n")),
        "accuracy": float(get("accuracy")),
        "precision": float(get("precision", "precision_binary")),
        "recall": float(get("recall", "recall_binary")),
        "f1": float(get("f1", "f1_binary")),
        "roc_auc": float(get("roc_auc")),
        "average_precision": float(get("average_precision")),
        "mcc": float(get("mcc")),
        "tn": int(get("tn")), "fp": int(get("fp")),
        "fn": int(get("fn")), "tp": int(get("tp")),
    }


def add(rows: list[dict], baseline: str, scenario: str, block: dict, path: Path) -> None:
    row = {"baseline": baseline, "scenario": scenario, **normalize(block)}
    row["source_file"] = str(path)
    row["source_file_sha256"] = sha256(path)
    rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root, out = args.root, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    old_dirs = {
        "MUDS": "cross_dataset/MUDS",
        "E2Phish": "cross_dataset/E2Phish",
        "Ebbu": "cross_dataset/Ebbu",
        "TabNet": "cross_dataset/TabNet",
        "StealthPhisher": "cross_dataset/StealthPhisher",
        "CNN-Fusion": "cross_dataset/CNN-Fusion",
        "DEPHIDES": "cross_dataset/DEPHIDES",
        "GramBeddings": "cross_dataset/GramBeddings",
    }
    for baseline, dirname in old_dirs.items():
        for scenario in ("PhishStorm_to_Ebbu2017", "Ebbu2017_to_PhishStorm"):
            path = root / dirname / scenario / f"{scenario}_metrics.json"
            data = json.loads(path.read_text())
            metrics = data["metrics"]
            block = metrics.get("test", metrics.get("target_test"))
            if block is None:
                raise KeyError(f"No target test metrics in {path}")
            add(rows, baseline, scenario, block, path)

    url_path = root / "cross_dataset/URLNet" / "URLNet_cross_transfer_summary.csv"
    for row in pd.read_csv(url_path).to_dict("records"):
        scenario = str(row["scenario"])
        if scenario in ("PhishStorm_to_Ebbu2017", "Ebbu2017_to_PhishStorm"):
            add(rows, "URLNet", scenario, row, url_path)

    full_root = root / "cross_dataset/full_phishfusion_baselines"
    for baseline in ("E2Phish", "MUDS", "TabNet", "Ebbu"):
        path = full_root / baseline / "summary.csv"
        for row in pd.read_csv(path).to_dict("records"):
            scenario = f"{row['source']}_to_PhishFusion"
            add(rows, baseline, scenario, row, path)

    deep_path = root / "cross_dataset/full_phishfusion_deep_models" / "summary.csv"
    deep_names = {"cnn_fusion": "CNN-Fusion", "deephides": "DEPHIDES", "stealthphisher": "StealthPhisher"}
    for row in pd.read_csv(deep_path).to_dict("records"):
        add(rows, deep_names[str(row["baseline"])], f"{row['source']}_to_PhishFusion", row, deep_path)

    for source in ("PhishStorm", "Ebbu2017"):
        scenario = f"{source}_to_PhishFusion"
        path = full_root / "GramBeddings" / scenario / f"{scenario}_repository_full_metrics.json"
        data = json.loads(path.read_text())
        add(rows, "GramBeddings", scenario, data["metrics"], path)

        path = full_root / "URLNet" / scenario / "metrics" / "common_binary_metrics.csv"
        add(rows, "URLNet", scenario, pd.read_csv(path).iloc[0].to_dict(), path)

    lite_root = root / "cross_dataset/LitePhish"
    for source in ("PhishStorm", "Ebbu2017"):
        path = lite_root / f"{source}_source" / "LitePhish_cross_transfer_summary.csv"
        for row in pd.read_csv(path).to_dict("records"):
            add(rows, "LitePhish", str(row["scenario"]), row, path)

    frame = pd.DataFrame(rows)
    duplicates = frame.duplicated(["baseline", "scenario"], keep=False)
    if duplicates.any():
        raise ValueError(f"Duplicate result rows:\n{frame.loc[duplicates, ['baseline', 'scenario']]}")
    expected = {(b, s) for b in BASELINES for s in SCENARIOS}
    actual = set(map(tuple, frame[["baseline", "scenario"]].itertuples(index=False, name=None)))
    if actual != expected:
        raise ValueError(f"Missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
    for scenario, expected_n in EXPECTED_N.items():
        observed = set(frame.loc[frame.scenario == scenario, "n"].astype(int))
        if observed != {expected_n}:
            raise ValueError(f"{scenario}: expected n={expected_n}, got {observed}")

    frame["baseline"] = pd.Categorical(frame["baseline"], BASELINES, ordered=True)
    frame["scenario"] = pd.Categorical(frame["scenario"], SCENARIOS, ordered=True)
    frame = frame.sort_values(["scenario", "baseline"]).reset_index(drop=True)
    long_path = out / "cross_dataset_metrics_long.csv"
    frame.to_csv(long_path, index=False)

    percent = frame.copy()
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "mcc"):
        percent[metric] = (100 * percent[metric]).round(2)
    percent.to_csv(out / "cross_dataset_metrics_percent.csv", index=False)
    compact = percent.pivot(index="baseline", columns="scenario", values=["precision", "recall", "f1", "roc_auc"])
    compact.to_csv(out / "cross_dataset_metrics_wide.csv")

    audit = {
        "rows": len(frame), "baselines": BASELINES, "scenarios": SCENARIOS,
        "expected_rows_per_scenario": EXPECTED_N,
        "metric_definition": "binary phishing class (1), threshold 0.5",
        "phishfusion_dataset": "dataset/PhishFusion.csv",
        "phishfusion_rows": 105752,
        "phishfusion_sha256": "c8af8da76bb6a521fcb7515eab802fec450dd10eafdfb49593096dcc07f1de8c",
        "long_csv": str(long_path), "long_csv_sha256": sha256(long_path),
    }
    (out / "cross_dataset_provenance.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
