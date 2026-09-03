"""Cross-dataset GramBeddings evaluation on canonical transfer manifests.

The source manifest supplies train/validation rows. The target manifest supplies
the untouched test cohort. N-gram TF-IDF/chi2 selection and all vocabularies are
fit exclusively on source-training URLs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from train_grambeddings_shared_split import (
    build_model,
    evaluate,
    fit_transformers,
    load_baseline,
    sha256,
)


def read_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="ISO-8859-1", usecols=["url", "phishing"])
    frame = frame.reset_index(drop=True)
    frame["url"] = frame["url"].fillna("").astype(str)
    frame["phishing"] = frame["phishing"].astype(int)
    return frame


def validate_manifest(dataset: pd.DataFrame, manifest: pd.DataFrame, path: Path) -> pd.DataFrame:
    required = {"row_id", "label"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"{path} is missing columns: {sorted(required - set(manifest.columns))}")
    manifest = manifest.copy()
    manifest["row_id"] = pd.to_numeric(manifest["row_id"], errors="raise").astype(np.int64)
    manifest["label"] = pd.to_numeric(manifest["label"], errors="raise").astype(np.int32)
    if manifest["row_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate row_id values")
    if (manifest["row_id"] < 0).any() or (manifest["row_id"] >= len(dataset)).any():
        raise ValueError(f"{path} contains row_id values outside the source CSV")
    actual = dataset.iloc[manifest["row_id"].to_numpy()]["phishing"].to_numpy()
    if not np.array_equal(actual, manifest["label"].to_numpy()):
        raise ValueError(f"{path} labels do not match its source CSV")
    return manifest


def select_source_rows(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "split" not in manifest.columns:
        raise ValueError("Source transfer manifest must contain a split column")
    train = manifest.loc[manifest["split"].eq("train")].copy()
    validation = manifest.loc[manifest["split"].eq("validation")].copy()
    if train.empty or validation.empty:
        raise ValueError("Source transfer manifest must contain nonempty train and validation rows")
    return train, validation


def select_target_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    if "split" not in manifest.columns:
        target = manifest.copy()
    elif manifest["split"].eq("test").any():
        target = manifest.loc[manifest["split"].eq("test")].copy()
    elif manifest["split"].eq("target_test").any():
        target = manifest.loc[manifest["split"].eq("target_test")].copy()
    else:
        raise ValueError("Target manifest has split but contains neither test nor target_test rows")
    if target.empty:
        raise ValueError("Target transfer cohort is empty")
    return target


def rows(dataset: pd.DataFrame, manifest_rows: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    selected = dataset.iloc[manifest_rows["row_id"].to_numpy()]
    return selected["url"].tolist(), selected["phishing"].to_numpy(dtype=np.int32)


def save_predictions(
    path: Path,
    dataset: pd.DataFrame,
    manifest_rows: pd.DataFrame,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> None:
    selected = dataset.iloc[manifest_rows["row_id"].to_numpy()][["url", "phishing"]].copy()
    selected.insert(0, "row_id", manifest_rows["row_id"].to_numpy())
    if "esld" in manifest_rows.columns:
        selected["esld"] = manifest_rows["esld"].fillna("").astype(str).to_numpy()
    selected["prediction"] = prediction
    selected["probability"] = probability
    selected.to_csv(path, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dataset", type=Path)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("target_dataset", type=Path)
    parser.add_argument("target_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument(
        "--baseline-dir", type=Path, default=Path("Baselines/GramBeddings")
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--ngrams", type=int, nargs=3, default=(4, 5, 6))
    parser.add_argument("--max-features", type=int, default=160000)
    parser.add_argument("--min-df", type=float, default=1e-6)
    parser.add_argument("--max-df", type=float, default=0.7)
    parser.add_argument("--embedding-dim", type=int, default=15)
    parser.add_argument("--char-embedding-dim", type=int, default=95)
    parser.add_argument("--attention-width", type=int, default=10)
    parser.add_argument("--rnn-cell-size", type=int, default=128)
    parser.add_argument("--case-insensitive", action="store_true")
    parser.add_argument("--skip-url-overlap-check", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    source_data = read_dataset(args.source_dataset)
    target_data = read_dataset(args.target_dataset)
    source_manifest = validate_manifest(source_data, pd.read_csv(args.source_manifest), args.source_manifest)
    target_manifest = validate_manifest(target_data, pd.read_csv(args.target_manifest), args.target_manifest)
    source_train, source_validation = select_source_rows(source_manifest)
    target_test = select_target_rows(target_manifest)
    train_urls, train_labels = rows(source_data, source_train)
    validation_urls, validation_labels = rows(source_data, source_validation)
    target_urls, target_labels = rows(target_data, target_test)

    # Canonical transfer cohorts preserve the full target population, including
    # target-internal repetitions. Leakage control concerns exact overlap between
    # source training and target testing, which is checked independently below.
    duplicate_count = int(pd.Series(target_urls).duplicated().sum())
    overlap_count = len(set(train_urls).intersection(target_urls))
    if overlap_count and not args.skip_url_overlap_check:
        raise ValueError(
            f"Target test has {overlap_count} exact URL overlaps with source train; "
            "construct the canonical transfer manifest"
        )

    classes = load_baseline(args.baseline_dir)
    arrays, transformers, channel_specs = fit_transformers(
        args, train_urls, train_labels, validation_urls, target_urls, classes
    )
    NBeddingModel, _, _, _ = classes
    model = build_model(args, channel_specs, NBeddingModel)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.source_name}_to_{args.target_name}"
    weights_path = args.output_dir / f"{run_name}_best.weights.h5"
    history = model.fit(
        arrays["train"],
        train_labels,
        validation_data=(arrays["validation"], validation_labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=[
            ModelCheckpoint(str(weights_path), monitor="val_loss", mode="min", save_best_only=True, save_weights_only=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.85, patience=5, min_delta=1e-4, mode="min"),
        ],
    )
    model.load_weights(str(weights_path))

    probabilities = {
        "source_validation": model.predict(arrays["validation"], batch_size=1024, verbose=0).reshape(-1),
        "target_test": model.predict(arrays["test"], batch_size=1024, verbose=0).reshape(-1),
    }
    predictions = {key: (value >= 0.5).astype(np.int32) for key, value in probabilities.items()}
    metrics = {
        "source_validation": evaluate(validation_labels, predictions["source_validation"], probabilities["source_validation"]),
        "target_test": evaluate(target_labels, predictions["target_test"], probabilities["target_test"]),
    }
    save_predictions(
        args.output_dir / f"{run_name}_source_validation_predictions.csv",
        source_data,
        source_validation,
        predictions["source_validation"],
        probabilities["source_validation"],
    )
    save_predictions(
        args.output_dir / f"{run_name}_target_test_predictions.csv",
        target_data,
        target_test,
        predictions["target_test"],
        probabilities["target_test"],
    )
    preprocessing_path = args.output_dir / f"{run_name}_preprocessing.pkl.gz"
    with gzip.open(preprocessing_path, "wb", compresslevel=3) as handle:
        pickle.dump(transformers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    record = {
        "baseline": "GramBeddings",
        "scenario": f"{args.source_name}->{args.target_name}",
        "source_dataset": str(args.source_dataset.resolve()),
        "source_dataset_sha256": sha256(args.source_dataset),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256(args.source_manifest),
        "target_dataset": str(args.target_dataset.resolve()),
        "target_dataset_sha256": sha256(args.target_dataset),
        "target_manifest": str(args.target_manifest.resolve()),
        "target_manifest_sha256": sha256(args.target_manifest),
        "target_internal_exact_duplicate_count": duplicate_count,
        "source_train_target_exact_overlap_count": overlap_count,
        "weights_sha256": sha256(weights_path),
        "preprocessing_sha256": sha256(preprocessing_path),
        "seed": args.seed,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "ngrams": list(args.ngrams),
            "max_features_per_ngram": args.max_features,
            "embedding_dim": args.embedding_dim,
            "char_embedding_dim": args.char_embedding_dim,
            "attention_width": args.attention_width,
            "rnn_cell_size": args.rnn_cell_size,
            "case_insensitive": args.case_insensitive,
        },
        "preprocessing_fit": "source train only",
        "feature_selection_fit": "source train only",
        "positive_class": {"value": 1, "name": "phishing"},
        "decision_threshold": 0.5,
        "validation_role": "source validation; best validation-loss checkpoint selection only",
        "best_epoch_by_source_val_loss": int(np.argmin(history.history["val_loss"]) + 1),
        "metrics": metrics,
    }
    metrics_path = args.output_dir / f"{run_name}_metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
