"""Train CNN-Fusion on a validated shared row manifest.

The character vocabulary is learned from the training partition only.  The
network architecture and default optimization settings match the released
CNN-Fusion baseline; CUDA device assignment is left to CUDA_VISIBLE_DEVICES so
several baselines can be scheduled safely in parallel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
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
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import (
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
    SpatialDropout1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def evaluate(y_true: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "benign": int((y_true == 0).sum()),
        "phishing": int((y_true == 1).sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
    }


def encode(urls: pd.Series, vocabulary: dict[str, int], max_length: int) -> np.ndarray:
    sequences = [[vocabulary[c] for c in str(url) if c in vocabulary] for url in urls]
    return pad_sequences(sequences, padding="post", truncating="post", maxlen=max_length)


def build_model(vocab_size: int, max_length: int) -> Model:
    inputs = Input(shape=(max_length,), dtype="int32")
    embedding = Embedding(vocab_size + 1, 16)(inputs)
    branches = []
    for filters, kernel in ((128, 8), (128, 10), (256, 12)):
        branch = Conv1D(filters=filters, kernel_size=kernel, activation="relu")(embedding)
        branch = SpatialDropout1D(0.4)(branch)
        branches.append(GlobalMaxPooling1D()(branch))
    merged = Concatenate()(branches)
    output = Dense(128, activation="relu")(merged)
    output = Dropout(0.4)(output)
    output = Dense(1, activation="sigmoid")(output)
    model = Model(inputs, output)
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.001, epsilon=1e-8),
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    data = pd.read_csv(args.dataset, encoding="ISO-8859-1", usecols=["url", "phishing"]).reset_index(drop=True)
    data["url"] = data["url"].fillna("").astype(str)
    data["phishing"] = data["phishing"].astype(int)
    manifest = pd.read_csv(args.manifest)
    if manifest["row_id"].tolist() != list(range(len(data))):
        raise ValueError("Manifest does not cover source rows exactly and in order")
    if not data["phishing"].equals(manifest["label"].astype(int)):
        raise ValueError("Manifest labels differ from source labels")
    masks = {split: manifest["split"].eq(split).to_numpy() for split in ("train", "validation", "test")}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("Manifest must contain non-empty train, validation, and test partitions")

    # Sorted characters make the train-only vocabulary deterministic.
    vocabulary = {char: i + 1 for i, char in enumerate(sorted(set("".join(data.loc[masks["train"], "url"]))))}
    arrays = {split: encode(data.loc[mask, "url"], vocabulary, args.max_length) for split, mask in masks.items()}
    labels = {split: data.loc[mask, "phishing"].to_numpy() for split, mask in masks.items()}
    model = build_model(len(vocabulary), args.max_length)
    history = model.fit(
        arrays["train"], labels["train"],
        validation_data=(arrays["validation"], labels["validation"]),
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
        callbacks=[ReduceLROnPlateau(monitor="val_accuracy", patience=3, factor=0.8, min_lr=1e-5)],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{args.name}_model.h5"
    tokenizer_path = args.output_dir / f"{args.name}_vocabulary.pickle"
    model.save(model_path)
    with tokenizer_path.open("wb") as handle:
        pickle.dump(vocabulary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(history.history).to_csv(args.output_dir / f"{args.name}_history.csv", index=False)

    summaries = {}
    for split in ("validation", "test"):
        probability = model.predict(arrays[split], batch_size=args.batch_size, verbose=0).reshape(-1)
        prediction = (probability >= 0.5).astype(int)
        summaries[split] = evaluate(labels[split], prediction, probability)
        rows = data.loc[masks[split], ["url", "phishing"]].copy()
        rows.insert(0, "row_id", manifest.loc[masks[split], "row_id"].to_numpy())
        rows["prediction"] = prediction
        rows["probability"] = probability
        rows.to_csv(args.output_dir / f"{args.name}_{split}_predictions.csv", index=False)

    record = {
        "baseline": "CNN-Fusion", "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset), "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest), "model_sha256": sha256(model_path),
        "vocabulary_sha256": sha256(tokenizer_path), "preprocessing_fit": "train only",
        "positive_class": {"value": 1, "name": "phishing"}, "decision_threshold": 0.5,
        "validation_in_training_rows": False,
        "validation_role": "learning-rate monitoring (released protocol)", "seed": args.seed,
        "epochs": args.epochs, "batch_size": args.batch_size, "max_length": args.max_length,
        "tensorflow_version": tf.__version__, "metrics": summaries,
    }
    (args.output_dir / f"{args.name}_metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
