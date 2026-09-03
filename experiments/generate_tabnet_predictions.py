"""Run the saved TabNet-baseline ensemble on an explicit URL/label cohort."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import joblib
import pandas as pd
from joblib import Parallel, delayed


def load_feature_module(path: Path):
    spec = importlib.util.spec_from_file_location("tabnet_feature_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load TabNet features from {path}")
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
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=18)
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort, usecols=["url", "phishing"])
    cohort["phishing"] = cohort["phishing"].astype(int)
    feature_module = load_feature_module(args.features)
    extract = feature_module.extract_features
    rows = Parallel(n_jobs=args.jobs)(delayed(extract)(url) for url in cohort["url"].astype(str))
    features = pd.DataFrame(rows).fillna(0)

    artifact = joblib.load(args.artifact)
    expected = list(artifact["imputer"].feature_names_in_)
    features = features.reindex(columns=expected, fill_value=0)
    imputed = artifact["imputer"].transform(features)
    scaled = artifact["scaler"].transform(imputed)
    selected = artifact.get("selected_features")
    if selected is not None:
        indices = [expected.index(name) for name in selected]
        scaled = scaled[:, indices]

    probability = artifact["model"].predict_proba(scaled)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    output = cohort.copy()
    output["probability"] = probability
    output["prediction"] = prediction
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print(f"rows={len(output)}")
    print(f"labels={output['phishing'].value_counts().sort_index().to_dict()}")
    print(f"artifact_sha256={sha256(args.artifact)}")
    print(f"features_sha256={sha256(args.features)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
