"""Train the TabNet-selection stacking baseline on a shared manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(y), "benign": int((y == 0).sum()), "phishing": int((y == 1).sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, pos_label=1, zero_division=0),
        "recall": recall_score(y, pred, pos_label=1, zero_division=0),
        "f1": f1_score(y, pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
        "average_precision": average_precision_score(y, prob),
        "mcc": matthews_corrcoef(y, pred),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    p.add_argument("manifest", type=Path)
    p.add_argument("preprocessing_code", type=Path)
    p.add_argument("selection_code", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    data = pd.read_csv(args.dataset, encoding="ISO-8859-1", usecols=["url", "phishing"]).reset_index(drop=True)
    manifest = pd.read_csv(args.manifest)
    if manifest.row_id.tolist() != list(range(len(data))): raise ValueError("Manifest row mismatch")
    if not data.phishing.astype(int).equals(manifest.label.astype(int)): raise ValueError("Manifest label mismatch")
    masks = {s: manifest.split.eq(s).to_numpy() for s in ("train", "validation", "test")}
    y = data.phishing.astype(int).to_numpy()

    prep = load_module("tabnet_shared_preprocessing", args.preprocessing_code)
    selector = load_module("tabnet_shared_selection", args.selection_code)
    raw = data.url.astype(str).apply(prep.extract_features).apply(pd.Series).fillna(0)
    imputer = SimpleImputer(strategy="mean").fit(raw.loc[masks["train"]])
    train_imp = imputer.transform(raw.loc[masks["train"]])
    scaler = StandardScaler().fit(train_imp)
    scaled = {s: scaler.transform(imputer.transform(raw.loc[masks[s]])) for s in masks}
    feature_names = raw.columns.tolist()
    indices, selected = selector.run_tabnet_feature_selection(scaled["train"], y[masks["train"]], feature_names)
    # The remaining stacking stage is CPU-only. Release the TabNet allocator so
    # another dataset can use the GPU while this process fits its SVM.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=args.jobs)),
            ("lr", LogisticRegression(max_iter=1000, n_jobs=args.jobs)),
            ("svm", SVC(probability=True, random_state=42, cache_size=8000)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5, n_jobs=args.jobs,
    )
    model.fit(scaled["train"][:, indices], y[masks["train"]])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split in ("validation", "test"):
        prob = model.predict_proba(scaled[split][:, indices])[:, 1]
        pred = (prob >= 0.5).astype(int)
        summaries[split] = evaluate(y[masks[split]], pred, prob)
        rows = data.loc[masks[split], ["url", "phishing"]].copy()
        rows.insert(0, "row_id", manifest.loc[masks[split], "row_id"].to_numpy())
        rows["prediction"] = pred; rows["probability"] = prob
        rows.to_csv(args.output_dir / f"{args.name}_{split}_predictions.csv", index=False)

    artifact_path = args.output_dir / f"{args.name}_artifacts.joblib"
    joblib.dump({"model": model, "imputer": imputer, "scaler": scaler,
                 "selected_features": selected, "selected_indices": indices}, artifact_path)
    record = {
        "baseline": "TabNet", "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset), "manifest_sha256": sha256(args.manifest),
        "preprocessing_code_sha256": sha256(args.preprocessing_code),
        "selection_code_sha256": sha256(args.selection_code),
        "artifact_sha256": sha256(artifact_path), "selected_features": selected,
        "preprocessing_fit": "train only", "feature_selection_fit": "train only",
        "positive_class": {"value": 1, "name": "phishing"}, "decision_threshold": 0.5,
        "validation_used_for_fitting": False, "metrics": summaries,
    }
    (args.output_dir / f"{args.name}_metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__": main()
