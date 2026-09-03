#!/usr/bin/env python3
"""Materialize URLNet inputs from an audited shared-split manifest.

The source CSV row order is preserved through ``row_id``.  No resplitting is
performed, and the emitted files contain only ``url,phishing`` so the legacy
URLNet loader cannot accidentally use group or split metadata as predictors.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    metadata_path = Path(args.metadata).resolve()
    output_dir = Path(args.output_dir).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if sha256(manifest_path) != metadata["manifest_sha256"]:
        raise RuntimeError("Manifest SHA-256 does not match its metadata")

    dataset_path = Path(metadata["dataset"]).resolve()
    if sha256(dataset_path) != metadata["dataset_sha256"]:
        raise RuntimeError("Dataset SHA-256 does not match the split metadata")

    dataset = pd.read_csv(dataset_path)
    manifest = pd.read_csv(manifest_path)
    expected = {"row_id", "split", "label", "esld"}
    if not expected.issubset(manifest.columns):
        raise ValueError("Manifest is missing: {}".format(expected - set(manifest.columns)))
    if manifest["row_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate row_id values")
    if len(manifest) != len(dataset):
        raise ValueError("Manifest and dataset row counts differ")
    if manifest["row_id"].min() != 0 or manifest["row_id"].max() != len(dataset) - 1:
        raise ValueError("Manifest row_id does not cover the complete dataset")

    ordered = manifest.sort_values("row_id")
    source_labels = dataset.iloc[ordered["row_id"].to_numpy()]["phishing"].astype(int).to_numpy()
    if not (source_labels == ordered["label"].astype(int).to_numpy()).all():
        raise ValueError("Manifest labels do not match source dataset labels")

    split_groups = {
        name: set(manifest.loc[manifest["split"] == name, "esld"].astype(str))
        for name in ("train", "validation", "test")
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_groups[left] & split_groups[right]
        if overlap:
            raise ValueError("eSLD overlap between {} and {}: {}".format(left, right, len(overlap)))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "positive_class": 1,
        "partitions": {},
    }
    for split in ("train", "validation", "test"):
        selected = manifest.loc[manifest["split"] == split].sort_values("row_id")
        frame = dataset.iloc[selected["row_id"].to_numpy()][["url", "phishing"]].copy()
        frame["phishing"] = frame["phishing"].astype(int)
        output_path = output_dir / "{}.csv".format(split)
        frame.to_csv(output_path, index=False)
        summary["partitions"][split] = {
            "path": str(output_path),
            "sha256": sha256(output_path),
            "rows": int(len(frame)),
            "phishing": int(frame["phishing"].sum()),
            "benign": int((frame["phishing"] == 0).sum()),
            "esld_groups": int(len(split_groups[split])),
        }

    summary_path = output_dir / "materialization_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
