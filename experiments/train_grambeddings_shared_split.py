"""Train the GramBeddings baseline on a validated shared split.

The baseline feature selectors/tokenizers are fit on the training partition only.
Validation is used only for model selection; all reported scores use untouched
validation/test rows and binary phishing-class metrics.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import random
import sys
import types
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
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_nltk_ngrams_fallback() -> bool:
    """Provide the only NLTK primitive used by the released baseline if absent."""
    if importlib.util.find_spec("nltk") is not None:
        return False
    module = types.ModuleType("nltk")

    def ngrams(sequence, n):
        sequence = tuple(sequence)
        return (sequence[i : i + n] for i in range(max(0, len(sequence) - n + 1)))

    module.ngrams = ngrams
    sys.modules["nltk"] = module
    return True


def load_baseline(baseline_dir: Path):
    baseline_dir = baseline_dir.resolve()
    if not (baseline_dir / "Model.py").is_file() or not (
        baseline_dir / "NGramSequenceTransformer.py"
    ).is_file():
        raise FileNotFoundError(f"Not a GramBeddings checkout: {baseline_dir}")
    sys.path.insert(0, str(baseline_dir))
    install_nltk_ngrams_fallback()
    from Model import NBeddingModel  # pylint: disable=import-outside-toplevel
    from NGramSequenceTransformer import (  # pylint: disable=import-outside-toplevel
        CharacterLevelTransformer,
        NBeddingTransformer,
        WeightInitializer,
    )

    return NBeddingModel, CharacterLevelTransformer, NBeddingTransformer, WeightInitializer


def evaluate(y_true: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict:
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
    }


def fit_transformers(args, train_urls, train_labels, validation_urls, test_urls, classes):
    _, CharacterLevelTransformer, NBeddingTransformer, WeightInitializer = classes
    char = CharacterLevelTransformer(
        args.max_seq_len, embedding_dim=args.char_embedding_dim, case_insensitive=args.case_insensitive
    )
    char_vocab_size, char_embedding_matrix = char.Fit()
    arrays = {
        "train": [char.Transform(train_urls)],
        "validation": [char.Transform(validation_urls)],
        "test": [char.Transform(test_urls)],
    }
    transformers = [char]
    channel_specs = [(char_vocab_size, args.char_embedding_dim, char_embedding_matrix, "embed_char")]

    weight_mode = WeightInitializer.randomly_initialize.value
    for channel, ngram_width in enumerate(args.ngrams, start=1):
        transformer = NBeddingTransformer(
            ngram_value=ngram_width,
            max_num_features=args.max_features,
            max_document_length=args.max_seq_len,
            min_df=args.min_df,
            max_df=args.max_df,
            embedding_dim=args.embedding_dim,
            case_insensitive=args.case_insensitive,
            weight_mode=weight_mode,
        )
        _, _, weight_matrix, _, _ = transformer.Fit(train_urls, train_labels)
        # The release returns len(word_index), although its OOV id is that value.
        # Embedding input_dim must therefore be max_id + 1 to avoid out-of-range OOV ids.
        vocab_size = max(transformer.tk.word_index.values()) + 1
        for split, urls in (
            ("train", train_urls),
            ("validation", validation_urls),
            ("test", test_urls),
        ):
            arrays[split].append(np.asarray(transformer.Transform(urls), dtype=np.float32))
        transformers.append(transformer)
        channel_specs.append((vocab_size, args.embedding_dim, weight_matrix, f"embed_ngram_{channel}"))
    return arrays, transformers, channel_specs


def build_model(args, channel_specs, NBeddingModel) -> Model:
    outputs, inputs = [], []
    for vocab_size, embedding_dim, embedding_matrix, layer_name in channel_specs:
        branch = NBeddingModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_seq_length=args.max_seq_len,
            embedding_matrix=embedding_matrix,
            rnn_cell_size=args.rnn_cell_size,
            attention_width=args.attention_width,
            warm_start=False,
        )
        output, input_layer = branch.CreateModel(embedding_layer_name=layer_name)
        outputs.append(output)
        inputs.append(input_layer)
    merged = Concatenate()(outputs)
    deep = Dense(2 * args.rnn_cell_size, activation="relu", name="deep_features")(merged)
    prediction = Dense(1, activation="sigmoid")(Dropout(0.2)(deep))
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--name", required=True)
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
    parser.add_argument("--smoke-test", action="store_true")
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

    classes = load_baseline(args.baseline_dir)
    if args.smoke_test:
        NBeddingModel, _, _, _ = classes
        specs = [(98, args.char_embedding_dim, None, "embed_char")]
        specs.extend((102, args.embedding_dim, None, f"embed_ngram_{i}") for i in range(1, 4))
        model = build_model(args, specs, NBeddingModel)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "tensorflow": tf.__version__,
                    "gpu_count": len(tf.config.list_physical_devices("GPU")),
                    "parameter_count": model.count_params(),
                    "nltk_fallback_active": getattr(sys.modules.get("nltk"), "__spec__", None) is None,
                },
                indent=2,
            )
        )
        return

    dataset = pd.read_csv(args.dataset, encoding="ISO-8859-1", usecols=["url", "phishing"])
    dataset = dataset.reset_index(drop=True)
    dataset["url"] = dataset["url"].fillna("").astype(str)
    dataset["phishing"] = dataset["phishing"].astype(int)
    manifest = pd.read_csv(args.manifest)
    required = {"row_id", "split", "label", "esld"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest is missing columns: {sorted(required - set(manifest.columns))}")
    if manifest["row_id"].tolist() != list(range(len(dataset))):
        raise ValueError("Manifest row_id is not an exact row-aligned index of the dataset")
    if not manifest["label"].astype(int).equals(dataset["phishing"]):
        raise ValueError("Manifest labels do not match the source dataset")
    if set(manifest["split"]) != {"train", "validation", "test"}:
        raise ValueError("Manifest must contain train, validation, and test splits")

    metadata_path = args.manifest.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    dataset_digest = sha256(args.dataset)
    if metadata and metadata.get("dataset_sha256") != dataset_digest:
        raise ValueError("Dataset SHA-256 differs from the manifest metadata")

    masks = {split: manifest["split"].eq(split).to_numpy() for split in ("train", "validation", "test")}
    urls = {split: dataset.loc[mask, "url"].tolist() for split, mask in masks.items()}
    labels = {split: dataset.loc[mask, "phishing"].to_numpy(dtype=np.int32) for split, mask in masks.items()}
    arrays, transformers, channel_specs = fit_transformers(
        args, urls["train"], labels["train"], urls["validation"], urls["test"], classes
    )
    NBeddingModel, _, _, _ = classes
    model = build_model(args, channel_specs, NBeddingModel)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / f"{args.name}_best.weights.h5"
    history = model.fit(
        arrays["train"],
        labels["train"],
        validation_data=(arrays["validation"], labels["validation"]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=[
            ModelCheckpoint(str(weights_path), monitor="val_loss", mode="min", save_best_only=True, save_weights_only=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.85, patience=5, min_delta=1e-4, mode="min"),
        ],
    )
    model.load_weights(str(weights_path))

    summaries = {}
    for split in ("validation", "test"):
        probability = model.predict(arrays[split], batch_size=1024, verbose=0).reshape(-1)
        prediction = (probability >= 0.5).astype(np.int32)
        summaries[split] = evaluate(labels[split], prediction, probability)
        rows = dataset.loc[masks[split], ["url", "phishing"]].copy()
        rows.insert(0, "row_id", manifest.loc[masks[split], "row_id"].to_numpy())
        rows["esld"] = manifest.loc[masks[split], "esld"].to_numpy()
        rows["prediction"] = prediction
        rows["probability"] = probability
        rows.to_csv(args.output_dir / f"{args.name}_{split}_predictions.csv", index=False)

    preprocessing_path = args.output_dir / f"{args.name}_preprocessing.pkl.gz"
    with gzip.open(preprocessing_path, "wb", compresslevel=3) as handle:
        pickle.dump(transformers, handle, protocol=pickle.HIGHEST_PROTOCOL)
    record = {
        "baseline": "GramBeddings",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": dataset_digest,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "baseline_dir": str(args.baseline_dir.resolve()),
        "preprocessing_sha256": sha256(preprocessing_path),
        "weights_sha256": sha256(weights_path),
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
        "preprocessing_fit": "train only",
        "feature_selection_fit": "train only",
        "positive_class": {"value": 1, "name": "phishing"},
        "decision_threshold": 0.5,
        "validation_role": "best validation-loss checkpoint selection only",
        "best_epoch_by_val_loss": int(np.argmin(history.history["val_loss"]) + 1),
        "metrics": summaries,
    }
    metrics_path = args.output_dir / f"{args.name}_metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    # Avoid TensorFlow preallocating every visible GPU when runs are parallelized.
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    main()
