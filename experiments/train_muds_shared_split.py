"""Train and evaluate MUDS using a precomputed shared split manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
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


def load_features(path: Path):
    spec = importlib.util.spec_from_file_location("muds_shared_split_features", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MUDS feature module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(y_true, prediction, probability) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "n": len(y_true),
        "benign": int((y_true == 0).sum()),
        "phishing": int((y_true == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
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
    parser.add_argument("features", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--jobs", type=int, default=18)
    args = parser.parse_args()

    data = pd.read_csv(args.dataset, encoding="ISO-8859-1", usecols=["url", "phishing"])
    data = data.reset_index(drop=True)
    manifest = pd.read_csv(args.manifest)
    if manifest["row_id"].tolist() != list(range(len(data))):
        raise ValueError("Manifest row IDs do not exactly cover the source dataset")
    if not data["phishing"].astype(int).equals(manifest["label"].astype(int)):
        raise ValueError("Manifest labels do not match source labels")

    feature_module = load_features(args.features)
    x_all = feature_module.extract_features(data.copy())
    train_mask = manifest["split"].eq("train").to_numpy()
    validation_mask = manifest["split"].eq("validation").to_numpy()
    test_mask = manifest["split"].eq("test").to_numpy()
    y_all = data["phishing"].astype(int).to_numpy()

    smote = SMOTE(random_state=42)
    x_train, y_train = smote.fit_resample(x_all.loc[train_mask], y_all[train_mask])
    model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_jobs=args.jobs,
        random_state=5,
        verbosity=-1,
    )
    model.fit(x_train, y_train)

    summaries = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, mask in (("validation", validation_mask), ("test", test_mask)):
        probability = model.predict_proba(x_all.loc[mask])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        y_true = y_all[mask]
        summaries[split] = metrics(y_true, prediction, probability)
        rows = data.loc[mask, ["url", "phishing"]].copy()
        rows.insert(0, "row_id", manifest.loc[mask, "row_id"].to_numpy())
        rows["prediction"] = prediction
        rows["probability"] = probability
        rows.to_csv(args.output_dir / f"{args.name}_{split}_predictions.csv", index=False)

    model_path = args.output_dir / f"{args.name}_model.joblib"
    joblib.dump(model, model_path)
    record = {
        "baseline": "MUDS",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "feature_code_sha256": sha256(args.features),
        "model_sha256": sha256(model_path),
        "positive_class": {"value": 1, "name": "phishing"},
        "decision_threshold": 0.5,
        "training_rows_before_smote": int(train_mask.sum()),
        "training_rows_after_smote": len(y_train),
        "validation_used_for_fitting": False,
        "metrics": summaries,
    }
    (args.output_dir / f"{args.name}_metrics.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
