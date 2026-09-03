"""Evaluate a fitted GramBeddings checkpoint on a target CSV without retraining."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tensorflow as tf

from train_grambeddings_shared_split import build_model, evaluate, load_baseline, sha256


def row_sequence_sha256(frame: pd.DataFrame) -> str:
    """Hash parsed row order, URL text, and labels independently of CSV layout."""
    digest = hashlib.sha256()
    for row_id, url, label in frame[["url", "phishing"]].itertuples(index=True, name=None):
        digest.update(str(row_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(url).encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
        digest.update(str(int(label)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def find_single(artifact_dir: Path, pattern: str) -> Path:
    matches = sorted(artifact_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {pattern} in {artifact_dir}; found {len(matches)}")
    return matches[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("target_dataset", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--target-name", default="PhishFusion_repository_full")
    parser.add_argument(
        "--baseline-dir", type=Path, default=Path("Baselines/GramBeddings")
    )
    parser.add_argument("--prediction-batch-size", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    weights_path = find_single(args.artifact_dir, "*_best.weights.h5")
    preprocessing_path = find_single(args.artifact_dir, "*_preprocessing.pkl.gz")
    training_metrics_path = find_single(args.artifact_dir, "*_metrics.json")
    training_record = json.loads(training_metrics_path.read_text(encoding="utf-8"))
    if sha256(weights_path) != training_record["weights_sha256"]:
        raise ValueError("Checkpoint hash does not match its common-split training audit")
    if sha256(preprocessing_path) != training_record["preprocessing_sha256"]:
        raise ValueError("Preprocessor hash does not match its common-split training audit")

    classes = load_baseline(args.baseline_dir)
    with gzip.open(preprocessing_path, "rb") as handle:
        transformers = pickle.load(handle)
    if len(transformers) != 4:
        raise ValueError(f"Expected char plus three n-gram transformers; found {len(transformers)}")

    target = pd.read_csv(args.target_dataset, encoding="ISO-8859-1", usecols=["url", "phishing"])
    target = target.reset_index(drop=True)
    target["url"] = target["url"].fillna("").astype(str)
    target["phishing"] = target["phishing"].astype(np.int32)
    if not target["phishing"].isin([0, 1]).all():
        raise ValueError("Target labels must be binary 0/1")
    urls = target["url"].tolist()
    labels = target["phishing"].to_numpy(dtype=np.int32)

    # Transform only: no Fit call is permitted during checkpoint re-evaluation.
    arrays = [np.asarray(transformer.Transform(urls), dtype=np.float32) for transformer in transformers]
    hyperparameters = training_record["hyperparameters"]
    model_args = SimpleNamespace(
        max_seq_len=int(hyperparameters["max_seq_len"]),
        rnn_cell_size=int(hyperparameters["rnn_cell_size"]),
        attention_width=int(hyperparameters["attention_width"]),
    )
    channel_specs = []
    for channel, transformer in enumerate(transformers):
        vocab_size = max(transformer.tk.word_index.values()) + 1
        embedding_dim = int(transformer.embedding_dim)
        layer_name = "embed_char" if channel == 0 else f"embed_ngram_{channel}"
        channel_specs.append((vocab_size, embedding_dim, None, layer_name))
    NBeddingModel, _, _, _ = classes
    model = build_model(model_args, channel_specs, NBeddingModel)
    model.load_weights(str(weights_path))
    probability = model.predict(arrays, batch_size=args.prediction_batch_size, verbose=1).reshape(-1)
    prediction = (probability >= 0.5).astype(np.int32)
    metrics = evaluate(labels, prediction, probability)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.source_name}_to_{args.target_name}"
    predictions_path = args.output_dir / f"{run_name}_predictions.csv"
    predictions = target.copy()
    predictions.insert(0, "row_id", np.arange(len(target), dtype=np.int64))
    predictions["prediction"] = prediction
    predictions["probability"] = probability
    predictions.to_csv(predictions_path, index=False)

    record = {
        "baseline": "GramBeddings",
        "evaluation": "existing common-split checkpoint on full repository PhishFusion target",
        "source_name": args.source_name,
        "target_name": args.target_name,
        "retrained": False,
        "checkpoint_reused": True,
        "preprocessor_reused": True,
        "target_preprocessing_fit": False,
        "artifact_dir": str(args.artifact_dir.resolve()),
        "weights": str(weights_path.resolve()),
        "weights_sha256": sha256(weights_path),
        "preprocessing": str(preprocessing_path.resolve()),
        "preprocessing_sha256": sha256(preprocessing_path),
        "common_split_training_audit": str(training_metrics_path.resolve()),
        "common_split_training_audit_sha256": sha256(training_metrics_path),
        "source_training_dataset": training_record["dataset"],
        "source_training_dataset_sha256": training_record["dataset_sha256"],
        "source_training_manifest": training_record["manifest"],
        "source_training_manifest_sha256": training_record["manifest_sha256"],
        "target_dataset": str(args.target_dataset.resolve()),
        "target_dataset_sha256": sha256(args.target_dataset),
        "target_parsed_row_sequence_sha256": row_sequence_sha256(target),
        "target_rows": int(len(target)),
        "target_benign": int((labels == 0).sum()),
        "target_phishing": int((labels == 1).sum()),
        "target_internal_exact_duplicate_count": int(target["url"].duplicated().sum()),
        "positive_class": {"value": 1, "name": "phishing"},
        "decision_threshold": 0.5,
        "prediction_file": str(predictions_path.resolve()),
        "prediction_file_sha256": sha256(predictions_path),
        "metrics": metrics,
    }
    metrics_path = args.output_dir / f"{run_name}_metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
