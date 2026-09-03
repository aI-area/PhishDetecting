#!/usr/bin/env python3
"""Materialize and audit a canonical cross-dataset target cohort for URLNet."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-value", default=None)
    parser.add_argument("--expected-manifest-sha256", default=None)
    parser.add_argument("--expected-dataset-sha256", default=None)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--allow-internal-duplicates", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest_hash = sha256(manifest_path)
    if args.expected_manifest_sha256 and manifest_hash.lower() != args.expected_manifest_sha256.lower():
        raise ValueError("Canonical transfer-manifest SHA-256 mismatch")

    manifest = pd.read_csv(manifest_path)
    if args.split_value is not None:
        if "split" not in manifest.columns:
            raise ValueError("--split-value requires a split column")
        manifest = manifest.loc[manifest["split"].astype(str) == args.split_value].copy()

    dataset_hash = None
    dataset_path = None
    if {"url", "phishing"}.issubset(manifest.columns):
        cohort = manifest[["url", "phishing"]].copy()
        mode = "manifest_rows"
    else:
        if "row_id" not in manifest.columns or not args.dataset_csv:
            raise ValueError("Manifest must contain url/phishing or row_id with --dataset-csv")
        dataset_path = Path(args.dataset_csv).resolve()
        dataset_hash = sha256(dataset_path)
        if args.expected_dataset_sha256 and dataset_hash.lower() != args.expected_dataset_sha256.lower():
            raise ValueError("Canonical target-dataset SHA-256 mismatch")
        dataset = pd.read_csv(dataset_path)
        row_ids = manifest["row_id"].astype(int)
        if row_ids.duplicated().any():
            raise ValueError("Transfer manifest contains duplicate row_id values")
        if (row_ids < 0).any() or (row_ids >= len(dataset)).any():
            raise ValueError("Transfer manifest row_id is outside the dataset")
        cohort = dataset.iloc[row_ids.to_numpy()][["url", "phishing"]].copy()
        if "label" in manifest.columns:
            expected_labels = manifest["label"].astype(int).reset_index(drop=True)
            actual_labels = cohort["phishing"].astype(int).reset_index(drop=True)
            if not expected_labels.equals(actual_labels):
                raise ValueError("Transfer-manifest labels do not match source rows")
        mode = "row_id_selection"

    cohort["url"] = cohort["url"].astype(str).str.strip()
    cohort["phishing"] = cohort["phishing"].astype(int)
    if not set(cohort["phishing"].unique()).issubset({0, 1}):
        raise ValueError("Target labels are not binary 0/1")
    if cohort["url"].eq("").any():
        raise ValueError("Target cohort contains empty URLs")
    duplicate_pairs = int(cohort.duplicated(["url", "phishing"]).sum())
    if duplicate_pairs and not args.allow_internal_duplicates:
        raise ValueError("Target cohort contains {} exact duplicate pairs".format(duplicate_pairs))
    if args.expected_rows is not None and len(cohort) != args.expected_rows:
        raise ValueError("Canonical target row-count mismatch")

    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(destination, index=False)
    audit = {
        "mode": mode,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "dataset": str(dataset_path) if dataset_path else None,
        "dataset_sha256": dataset_hash,
        "output": str(destination),
        "output_sha256": sha256(destination),
        "rows": int(len(cohort)),
        "benign": int((cohort["phishing"] == 0).sum()),
        "phishing": int((cohort["phishing"] == 1).sum()),
        "exact_duplicate_pairs": duplicate_pairs,
        "internal_duplicates_explicitly_allowed": bool(args.allow_internal_duplicates),
        "distinct_urls": int(cohort["url"].nunique()),
    }
    audit_path = destination.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
