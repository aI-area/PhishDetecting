#!/usr/bin/env python3
"""Inference-only evaluation of trained classical baselines on one frozen target."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_baseline_features", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_named(values):
    parsed = {}
    for value in values:
        name, path = value.split("=", 1)
        parsed[name] = Path(path).resolve()
    return parsed


def metric_record(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "n_benign": int((y == 0).sum()),
        "n_phishing": int((y == 1).sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_binary": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_binary": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "f1_binary": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "threshold_0_5_mismatches": int((pred != (prob >= 0.5).astype(int)).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=["MUDS", "E2Phish", "Ebbu", "TabNet"])
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--expected-target-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--feature-code", type=Path, required=True)
    parser.add_argument("--source-artifact", action="append", required=True)
    parser.add_argument("--selected", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.target = args.target.resolve()
    target_hash = sha256(args.target)
    if target_hash.lower() != args.expected_target_sha256.lower():
        raise ValueError("Target SHA-256 mismatch")
    data = pd.read_csv(args.target, usecols=["url", "phishing"]).reset_index(drop=True)
    if len(data) != args.expected_rows:
        raise ValueError("Target row-count mismatch")
    y = data["phishing"].astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Target labels are not binary")

    feature_path = args.feature_code.resolve()
    feature_module = load_module(feature_path)
    artifacts = parse_named(args.source_artifact)
    selected_paths = parse_named(args.selected)

    if args.baseline == "MUDS":
        common_features = feature_module.extract_features(data.copy())
    elif args.baseline in ("E2Phish", "Ebbu"):
        processed = feature_module.extract_features_from_dataframe(data.copy())
        if len(processed) != len(data):
            raise ValueError("Feature extraction changed target row count")
        common_features = processed.drop(columns=["labels"], errors="ignore")
        if args.baseline == "E2Phish":
            common_features = common_features.select_dtypes(include=[np.number])
    else:
        common_features = data["url"].astype(str).apply(
            feature_module.extract_features
        ).apply(pd.Series).fillna(0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for source, artifact_path in artifacts.items():
        loaded = joblib.load(artifact_path)
        if args.baseline == "TabNet":
            model = loaded["model"]
            transformed = loaded["scaler"].transform(
                loaded["imputer"].transform(common_features)
            )
            x_target = transformed[:, loaded["selected_indices"]]
        elif args.baseline == "E2Phish":
            model = loaded
            selected = joblib.load(selected_paths[source])
            missing = sorted(set(selected).difference(common_features.columns))
            if missing:
                raise ValueError("Missing selected features for {}: {}".format(source, missing))
            x_target = common_features.loc[:, selected]
        else:
            model = loaded
            x_target = common_features

        prob = np.asarray(model.predict_proba(x_target)[:, 1], dtype=float)
        pred = (prob >= 0.5).astype(int)
        metrics = metric_record(y, pred, prob)
        source_dir = args.output_dir / "{}_to_PhishFusion".format(source)
        source_dir.mkdir(parents=True, exist_ok=True)
        predictions = data[["url", "phishing"]].copy()
        predictions.insert(0, "row_id", np.arange(len(data), dtype=int))
        predictions["prediction"] = pred
        predictions["probability"] = prob
        prediction_path = source_dir / "{}_results.csv".format(args.baseline)
        predictions.to_csv(prediction_path, index=False)
        record = {
            "baseline": args.baseline,
            "source": source,
            "target": str(args.target),
            "target_sha256": target_hash,
            "target_exact_duplicate_pairs": int(data.duplicated(["url", "phishing"]).sum()),
            "target_order": "original pandas CSV row order; row_id is zero-based",
            "feature_code": str(feature_path),
            "feature_code_sha256": sha256(feature_path),
            "artifact": str(artifact_path),
            "artifact_sha256": sha256(artifact_path),
            "selected_features_artifact": str(selected_paths.get(source)) if source in selected_paths else None,
            "selected_features_sha256": sha256(selected_paths[source]) if source in selected_paths else None,
            "decision_threshold": 0.5,
            "positive_class": 1,
            "prediction_file": str(prediction_path),
            "prediction_sha256": sha256(prediction_path),
            "metrics": metrics,
        }
        (source_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        summaries.append({"source": source, **metrics})

    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
