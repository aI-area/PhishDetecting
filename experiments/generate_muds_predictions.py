"""Run the saved MUDS model on an explicit URL/label cohort."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import joblib
import pandas as pd


def load_feature_module(path: Path):
    spec = importlib.util.spec_from_file_location("muds_feature_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MUDS features from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort, usecols=["url", "phishing"])
    cohort["phishing"] = cohort["phishing"].astype(int)
    if cohort.empty or not set(cohort["phishing"].unique()).issubset({0, 1}):
        raise ValueError("Cohort must contain nonempty binary phishing labels")

    feature_module = load_feature_module(args.features)
    features = feature_module.extract_features(cohort.copy())
    model = joblib.load(args.model)
    if getattr(model, "n_features_in_", features.shape[1]) != features.shape[1]:
        raise ValueError(
            f"Feature mismatch: model expects {model.n_features_in_}, got {features.shape[1]}"
        )

    probability = model.predict_proba(features)[:, 1]
    prediction = model.predict(features).astype(int)
    output = cohort.copy()
    output["prediction"] = prediction
    output["probability"] = probability
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print(f"rows={len(output)}")
    print(f"labels={output['phishing'].value_counts().sort_index().to_dict()}")
    print(f"model_sha256={sha256(args.model)}")
    print(f"features_sha256={sha256(args.features)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
