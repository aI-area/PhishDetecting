#!/usr/bin/env python3
"""Train matched raw-56,601 and compact-568 LitePhish configurations.

The script reuses the released PhishFusion fitted N-gram vocabulary and the
released 568 selected-feature indices, but refits both scalers and both models
on the same domain-stratified training partition. Large feature matrices are
never serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(model, x_test, y_test) -> dict[str, float]:
    pred = np.asarray(model.predict(x_test), dtype=int)
    prob = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("released_compact_artifacts", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--n-estimators", type=int, default=500)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.path.insert(0, str(args.pipeline_root))

    from feature_extraction import FeatureExtractor
    from main import PhishingDetector
    from model import FocalLossLGBM
    from ngram_processing import NgramProcessor

    setattr(sys.modules["__main__"], "NgramProcessor", NgramProcessor)
    setattr(sys.modules["__main__"], "FocalLossLGBM", FocalLossLGBM)

    processor_path = args.released_compact_artifacts / "ngram_processor.pkl"
    selected_path = args.released_compact_artifacts / "selected_features.pkl"
    processor = joblib.load(processor_path)
    selected = np.asarray(joblib.load(selected_path), dtype=int)
    vocabulary_size = len(processor.vectorizer.vocabulary_)
    if vocabulary_size != 56515 or len(selected) != 568:
        raise RuntimeError(f"expected vocabulary=56515 and compact=568; got {vocabulary_size}, {len(selected)}")

    detector = PhishingDetector()
    urls, labels = detector.load_data(str(args.dataset))
    train_idx, validation_idx, test_idx = detector.split_data_indices(urls, labels)
    train_urls = urls.iloc[train_idx]
    test_urls = urls.iloc[test_idx]
    y_train = labels.iloc[train_idx].to_numpy(dtype=int)
    y_test = labels.iloc[test_idx].to_numpy(dtype=int)

    extractor = FeatureExtractor()
    start = time.perf_counter()
    handcrafted = Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(extractor.generate_features)(url) for url in urls
    )
    handcrafted = np.asarray(handcrafted, dtype=np.float32)
    handcrafted_seconds = time.perf_counter() - start
    if handcrafted.shape[1] != 86:
        raise RuntimeError(f"expected 86 handcrafted features, got {handcrafted.shape[1]}")

    hc_train = sparse.csr_matrix(handcrafted[train_idx])
    hc_test = sparse.csr_matrix(handcrafted[test_idx])
    raw_train = sparse.hstack(
        [hc_train, processor.vectorizer.transform(train_urls)], format="csr", dtype=np.float32
    )
    raw_test = sparse.hstack(
        [hc_test, processor.vectorizer.transform(test_urls)], format="csr", dtype=np.float32
    )
    if raw_train.shape[1] != 56601:
        raise RuntimeError(f"expected 56,601 raw features, got {raw_train.shape[1]}")

    filtered_train = sparse.hstack(
        [hc_train, processor.transform(train_urls)], format="csr", dtype=np.float32
    )
    filtered_test = sparse.hstack(
        [hc_test, processor.transform(test_urls)], format="csr", dtype=np.float32
    )
    compact_train = filtered_train[:, selected].toarray().astype(np.float32, copy=False)
    compact_test = filtered_test[:, selected].toarray().astype(np.float32, copy=False)

    model_params = dict(
        gamma=0.30, alpha=0.95, boosting_type="gbdt", num_leaves=50,
        max_depth=-1, learning_rate=0.15, n_estimators=args.n_estimators,
        min_child_samples=40, reg_alpha=0.1, reg_lambda=0.3,
        verbosity=-1, random_state=42, device="cpu", n_jobs=1,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}

    for name, x_train, x_test, with_mean in (
        ("raw_56601", raw_train, raw_test, False),
        ("compact_568", compact_train, compact_test, True),
    ):
        scaler = StandardScaler(with_mean=with_mean)
        scale_start = time.perf_counter()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        scale_seconds = time.perf_counter() - scale_start
        model = FocalLossLGBM(**model_params)
        train_start = time.perf_counter()
        model.fit(x_train_scaled, y_train)
        train_seconds = time.perf_counter() - train_start
        config_dir = args.output_dir / name
        config_dir.mkdir(exist_ok=True)
        joblib.dump(model, config_dir / "model.joblib")
        joblib.dump(scaler, config_dir / "scaler.joblib")
        results[name] = {
            "feature_count": int(x_train.shape[1]),
            "train_rows": int(x_train.shape[0]),
            "test_rows": int(x_test.shape[0]),
            "matrix_kind": "sparse_csr" if sparse.issparse(x_train) else "dense_float32",
            "scaler_with_mean": with_mean,
            "scaling_seconds": scale_seconds,
            "training_seconds": train_seconds,
            "metrics": metrics(model, x_test_scaled, y_test),
        }
        del x_train_scaled, x_test_scaled, model, scaler

    # Shared inference support files; raw and compact use different views.
    joblib.dump(processor, args.output_dir / "ngram_processor.joblib")
    joblib.dump(selected, args.output_dir / "selected_568.joblib")
    audit = {
        "status": "PASS",
        "dataset": {"name": args.dataset.name, "rows": int(len(urls)), "sha256": sha256(args.dataset)},
        "split": {
            "train_rows": int(len(train_idx)), "validation_rows": int(len(validation_idx)),
            "test_rows": int(len(test_idx)), "method": "released PhishingDetector domain-stratified split",
        },
        "shared": {
            "handcrafted_features": 86, "raw_ngram_vocabulary": vocabulary_size,
            "raw_features": 86 + vocabulary_size, "filtered_ngram_features": 5000,
            "compact_features": int(len(selected)), "gamma": 0.30, "alpha": 0.95,
            "n_estimators": args.n_estimators, "random_seed": 42,
            "handcrafted_extraction_seconds_all_rows": handcrafted_seconds,
            "processor_sha256": sha256(processor_path), "selected_indices_sha256": sha256(selected_path),
            "python": platform.python_version(),
        },
        "configurations": results,
    }
    with (args.output_dir / "training_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
