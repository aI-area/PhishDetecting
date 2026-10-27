"""Best-effort reproduction of Roy et al. (IEEE TCE 2024).

The paper reports 17 URL-derived features and identifies Random Forest using all
features as its strongest handcrafted-feature classifier. It does not report the
Random Forest hyperparameters, random seed, exact URL-normalization code, or the
URL-shortening service list. This implementation therefore uses the published
feature definitions, a standard 100-tree scikit-learn Random Forest, and an
explicit shortener-domain list. It supports the shared internal domain-disjoint
manifests and the fixed cross-dataset cohorts used by LitePhish.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import platform
import sys
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
import sklearn
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


FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "top_level_domain_length",
    "underscore_count",
    "at_count",
    "question_mark_count",
    "percent_count",
    "dot_count",
    "equals_count",
    "http_count",
    "https_count",
    "letter_count",
    "www_count",
    "digit_count",
    "uses_ip_address",
    "uses_shortening_service",
]


# The paper defines a binary shortened-URL feature but does not publish the
# associated service list. This fixed list is therefore an explicit best-effort
# implementation choice and is recorded in each experiment manifest.
SHORTENER_DOMAINS = frozenset(
    {
        "1url.com",
        "adf.ly",
        "bc.vc",
        "bit.do",
        "bit.ly",
        "buff.ly",
        "cutt.ly",
        "db.tt",
        "goo.gl",
        "is.gd",
        "lc.chat",
        "lnkd.in",
        "ow.ly",
        "rb.gy",
        "rebrand.ly",
        "s.id",
        "soo.gd",
        "t.co",
        "tiny.cc",
        "tinyurl.com",
        "trib.al",
        "v.gd",
        "x.co",
        "youtu.be",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parsed_url(raw_url: str):
    value = str(raw_url).strip()
    candidate = value if "://" in value else f"//{value}"
    return value, urlsplit(candidate, allow_fragments=True)


def is_ip_address(hostname: str) -> int:
    host = hostname.strip("[]")
    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        return 0


def is_shortener(hostname: str) -> int:
    host = hostname.lower().rstrip(".")
    return int(any(host == domain or host.endswith(f".{domain}") for domain in SHORTENER_DOMAINS))


def extract_one(raw_url: str) -> list[int]:
    value, parsed = parsed_url(raw_url)
    lowered = value.lower()
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    tld = "" if is_ip_address(hostname) else hostname.rsplit(".", 1)[-1] if "." in hostname else ""
    return [
        len(value),
        len(hostname),
        len(path),
        len(tld),
        value.count("_"),
        value.count("@"),
        value.count("?"),
        value.count("%"),
        value.count("."),
        value.count("="),
        lowered.count("http"),
        lowered.count("https"),
        sum(character.isalpha() for character in value),
        lowered.count("www"),
        sum(character.isdigit() for character in value),
        is_ip_address(hostname),
        is_shortener(hostname),
    ]


def extract_features(urls: pd.Series) -> pd.DataFrame:
    matrix = np.asarray([extract_one(url) for url in urls.astype(str)], dtype=np.int32)
    return pd.DataFrame(matrix, columns=FEATURE_NAMES)


def load_rows(dataset_path: Path, manifest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(dataset_path, encoding="ISO-8859-1", usecols=["url", "phishing"]).reset_index(drop=True)
    manifest = pd.read_csv(manifest_path).reset_index(drop=True)
    required = {"row_id", "split", "label"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest lacks columns: {sorted(required - set(manifest.columns))}")
    row_ids = manifest["row_id"].astype(int).to_numpy()
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("Manifest row_id values are not unique")
    if row_ids.min(initial=0) < 0 or row_ids.max(initial=-1) >= len(data):
        raise ValueError("Manifest row_id is outside the dataset")
    selected = data.iloc[row_ids].reset_index(drop=True)
    if not selected["phishing"].astype(int).equals(manifest["label"].astype(int)):
        raise ValueError("Manifest labels do not match dataset labels")
    return selected, manifest


def calculate_metrics(y_true: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "benign": int((y_true == 0).sum()),
        "phishing": int((y_true == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
    }


def predict_and_save(
    model: RandomForestClassifier,
    rows: pd.DataFrame,
    manifest: pd.DataFrame,
    mask: np.ndarray,
    output_path: Path,
) -> dict:
    selected_rows = rows.loc[mask].reset_index(drop=True)
    selected_manifest = manifest.loc[mask].reset_index(drop=True)
    features = extract_features(selected_rows["url"])
    probability = model.predict_proba(features)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    y_true = selected_rows["phishing"].astype(int).to_numpy()
    output = pd.DataFrame(
        {
            "row_id": selected_manifest["row_id"].astype(int),
            "url": selected_rows["url"],
            "phishing": y_true,
            "prediction": prediction,
            "probability": probability,
        }
    )
    output.to_csv(output_path, index=False, float_format="%.17g")
    serialized = pd.read_csv(output_path)
    return calculate_metrics(
        serialized["phishing"].astype(int).to_numpy(),
        serialized["prediction"].astype(int).to_numpy(),
        serialized["probability"].astype(float).to_numpy(),
    )


def fit_model(train_rows: pd.DataFrame, jobs: int) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        class_weight=None,
        random_state=42,
        n_jobs=jobs,
    )
    model.fit(extract_features(train_rows["url"]), train_rows["phishing"].astype(int))
    return model


def common_record(args: argparse.Namespace, model_path: Path) -> dict:
    return {
        "baseline": "Roy et al. (2024) best-effort Random Forest reproduction",
        "doi": "10.1109/TCE.2024.3404459",
        "published_method": "Random Forest using all 17 handcrafted URL features in Table I",
        "reproduction_status": "best effort; official code and complete implementation settings were not reported",
        "approximations": [
            "The paper does not report Random Forest hyperparameters; standard scikit-learn settings with 100 trees are used.",
            "The paper does not report a random seed; random_state=42 is used.",
            "The paper does not define 'high level domain'; it is implemented as the final hostname label (top-level domain).",
            "The paper does not publish the URL-shortening service list; the fixed list in this script is used.",
            "Feature counts are computed from the original URL string because query delimiters and equals signs are published features.",
        ],
        "feature_names": FEATURE_NAMES,
        "shortener_domains": sorted(SHORTENER_DOMAINS),
        "positive_class": {"value": 1, "name": "phishing"},
        "decision_threshold": 0.5,
        "model_parameters": {
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "class_weight": None,
            "random_state": 42,
            "n_jobs": args.jobs,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
        "model_sha256": sha256(model_path),
    }


def run_internal(args: argparse.Namespace) -> None:
    rows, manifest = load_rows(args.source_dataset, args.source_manifest)
    train_mask = manifest["split"].eq("train").to_numpy()
    if not train_mask.any():
        raise ValueError("No training rows in source manifest")
    model = fit_model(rows.loc[train_mask].reset_index(drop=True), args.jobs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "roy_random_forest.joblib"
    joblib.dump(model, model_path)
    metrics = {}
    for split in ("validation", "test"):
        mask = manifest["split"].eq(split).to_numpy()
        if mask.any():
            metrics[split] = predict_and_save(
                model, rows, manifest, mask, args.output_dir / f"{split}_predictions.csv"
            )
    record = common_record(args, model_path)
    record.update(
        {
            "mode": "internal_domain_disjoint",
            "experiment_name": args.name,
            "source_dataset": str(args.source_dataset.resolve()),
            "source_dataset_sha256": sha256(args.source_dataset),
            "source_manifest": str(args.source_manifest.resolve()),
            "source_manifest_sha256": sha256(args.source_manifest),
            "training_rows": int(train_mask.sum()),
            "validation_used_for_fitting": False,
            "metrics": metrics,
        }
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


def run_cross(args: argparse.Namespace) -> None:
    source_rows, source_manifest = load_rows(args.source_dataset, args.source_manifest)
    target_rows, target_manifest = load_rows(args.target_dataset, args.target_manifest)
    train_mask = source_manifest["split"].eq("train").to_numpy()
    if not train_mask.any():
        raise ValueError("No training rows in source manifest")
    target_mask = np.ones(len(target_manifest), dtype=bool)
    model = fit_model(source_rows.loc[train_mask].reset_index(drop=True), args.jobs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "roy_random_forest.joblib"
    joblib.dump(model, model_path)
    target_metrics = predict_and_save(
        model,
        target_rows,
        target_manifest,
        target_mask,
        args.output_dir / "target_predictions.csv",
    )
    record = common_record(args, model_path)
    record.update(
        {
            "mode": "cross_dataset",
            "experiment_name": args.name,
            "source_dataset": str(args.source_dataset.resolve()),
            "source_dataset_sha256": sha256(args.source_dataset),
            "source_manifest": str(args.source_manifest.resolve()),
            "source_manifest_sha256": sha256(args.source_manifest),
            "target_dataset": str(args.target_dataset.resolve()),
            "target_dataset_sha256": sha256(args.target_dataset),
            "target_manifest": str(args.target_manifest.resolve()),
            "target_manifest_sha256": sha256(args.target_manifest),
            "training_rows": int(train_mask.sum()),
            "target_rows": int(target_mask.sum()),
            "target_labels_used_for_fitting": False,
            "metrics": {"target": target_metrics},
        }
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("internal", "cross"), required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--jobs", type=int, default=18)
    args = parser.parse_args()
    if args.mode == "cross" and (args.target_dataset is None or args.target_manifest is None):
        parser.error("--target-dataset and --target-manifest are required in cross mode")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "internal":
        run_internal(arguments)
    else:
        run_cross(arguments)
