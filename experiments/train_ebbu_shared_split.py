"""Train the Ebbu random-forest baseline on a shared split manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
    spec = importlib.util.spec_from_file_location("ebbu_shared_features", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(y_true, prediction, probability):
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "n": len(y_true), "benign": int((y_true == 0).sum()),
        "phishing": int((y_true == 1).sum()), "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, prediction, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, prediction, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "average_precision": average_precision_score(y_true, probability),
        "mcc": matthews_corrcoef(y_true, prediction),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("feature_code", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--jobs", type=int, default=18)
    args = parser.parse_args()

    data = pd.read_csv(args.dataset, encoding="ISO-8859-1", usecols=["url", "phishing"]).reset_index(drop=True)
    manifest = pd.read_csv(args.manifest)
    if manifest.row_id.tolist() != list(range(len(data))):
        raise ValueError("Manifest does not cover source rows exactly")
    if not data.phishing.astype(int).equals(manifest.label.astype(int)):
        raise ValueError("Manifest labels differ from source labels")

    module = load_module(args.feature_code)
    processed = module.extract_features_from_dataframe(data)
    if len(processed) != len(data):
        raise ValueError(f"Feature extractor returned {len(processed)} of {len(data)} rows")
    x_all = processed.drop(columns=["labels"])
    y_all = data.phishing.astype(int).to_numpy()
    masks = {split: manifest.split.eq(split).to_numpy() for split in ("train", "validation", "test")}

    model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=args.jobs)
    model.fit(x_all.loc[masks["train"]], y_all[masks["train"]])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split in ("validation", "test"):
        mask = masks[split]
        probability = model.predict_proba(x_all.loc[mask])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        summaries[split] = evaluate(y_all[mask], prediction, probability)
        rows = data.loc[mask, ["url", "phishing"]].copy()
        rows.insert(0, "row_id", manifest.loc[mask, "row_id"].to_numpy())
        rows["prediction"] = prediction
        rows["probability"] = probability
        rows.to_csv(args.output_dir / f"{args.name}_{split}_predictions.csv", index=False)

    model_path = args.output_dir / f"{args.name}_model.joblib"
    joblib.dump(model, model_path)
    record = {
        "baseline": "Ebbu", "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset), "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest), "feature_code_sha256": sha256(args.feature_code),
        "model_sha256": sha256(model_path), "positive_class": {"value": 1, "name": "phishing"},
        "decision_threshold": 0.5, "validation_used_for_fitting": False,
        "metrics": summaries,
    }
    (args.output_dir / f"{args.name}_metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
