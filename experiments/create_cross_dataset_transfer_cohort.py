"""Materialize an audited, exact-cross-dataset-duplicate-free transfer cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"url", "phishing"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    frame = frame[["url", "phishing"]].copy()
    frame["url"] = frame["url"].astype(str).str.strip()
    frame["phishing"] = frame["phishing"].astype(int)
    if not set(frame["phishing"].unique()).issubset({0, 1}):
        raise ValueError(f"{path}: labels are not binary 0/1")
    return frame


def load_manifest(path: Path, dataset: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"row_id", "split", "label", "esld"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    manifest = manifest.sort_values("row_id").reset_index(drop=True)
    expected = pd.Series(range(len(dataset)), name="row_id")
    if not manifest["row_id"].reset_index(drop=True).equals(expected):
        raise ValueError(f"{path}: row_id is not an exact 0..N-1 sequence")
    if not manifest["label"].astype(int).reset_index(drop=True).equals(
        dataset["phishing"].reset_index(drop=True)
    ):
        raise ValueError(f"{path}: manifest labels do not match dataset rows")
    return manifest


def counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "benign": int((frame["phishing"] == 0).sum()),
        "phishing": int((frame["phishing"] == 1).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dataset", type=Path)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("target_dataset", type=Path)
    parser.add_argument("target_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--target-name", required=True)
    args = parser.parse_args()

    source = load_dataset(args.source_dataset)
    target = load_dataset(args.target_dataset)
    source_manifest = load_manifest(args.source_manifest, source)
    target_manifest = load_manifest(args.target_manifest, target)

    # Enforce exact source-training--target URL separation.
    # transfer: a target URL is excluded if it occurs anywhere in the complete
    # source dataset, including the source partition not used for fitting.
    source_urls = set(source["url"])
    target_is_cross_duplicate = target["url"].isin(source_urls)
    removed = target.loc[target_is_cross_duplicate].copy()
    retained_target = target.loc[~target_is_cross_duplicate].copy()
    retained_target_manifest = target_manifest.loc[~target_is_cross_duplicate].copy()

    source_labels_by_url = source.groupby("url")["phishing"].agg(set)
    conflict_count = 0
    for row in removed.itertuples(index=False):
        if row.phishing not in source_labels_by_url[row.url]:
            conflict_count += 1

    source_keep = source_manifest["split"].isin(["train", "validation"])
    source_part = source.loc[source_keep].copy()
    source_part_manifest = source_manifest.loc[source_keep].copy()

    parts: list[pd.DataFrame] = []
    manifests: list[pd.DataFrame] = []
    for split in ("train", "validation"):
        mask = source_part_manifest["split"].eq(split)
        part = source_part.loc[mask, ["url", "phishing"]].copy()
        part["origin_dataset"] = args.source_name
        part["origin_row_id"] = source_part_manifest.loc[mask, "row_id"].to_numpy()
        parts.append(part)
        manifests.append(
            pd.DataFrame(
                {
                    "split": split,
                    "label": part["phishing"].to_numpy(),
                    "esld": source_part_manifest.loc[mask, "esld"].to_numpy(),
                }
            )
        )

    test_part = retained_target[["url", "phishing"]].copy()
    test_part["origin_dataset"] = args.target_name
    test_part["origin_row_id"] = retained_target.index.to_numpy()
    parts.append(test_part)
    manifests.append(
        pd.DataFrame(
            {
                "split": "test",
                "label": test_part["phishing"].to_numpy(),
                "esld": retained_target_manifest["esld"].to_numpy(),
            }
        )
    )

    cohort = pd.concat(parts, ignore_index=True)
    manifest = pd.concat(manifests, ignore_index=True)
    manifest.insert(0, "row_id", range(len(manifest)))
    if not cohort["phishing"].reset_index(drop=True).equals(
        manifest["label"].reset_index(drop=True)
    ):
        raise AssertionError("Materialized cohort and manifest labels diverged")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = args.output_dir / "cohort.csv"
    manifest_path = args.output_dir / "manifest.csv"
    source_transfer_manifest_path = args.output_dir / "source_manifest.csv"
    target_transfer_manifest_path = args.output_dir / "target_manifest.csv"
    cohort.to_csv(cohort_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    source_transfer_manifest = source_part_manifest[
        ["row_id", "split", "label", "esld"]
    ].copy()
    target_transfer_manifest = retained_target_manifest[
        ["row_id", "label", "esld"]
    ].copy()
    target_transfer_manifest.insert(1, "split", "target_test")
    source_transfer_manifest.to_csv(source_transfer_manifest_path, index=False)
    target_transfer_manifest.to_csv(target_transfer_manifest_path, index=False)

    split_counts = {
        split: counts(cohort.loc[manifest["split"].eq(split)])
        for split in ("train", "validation", "test")
    }
    audit = {
        "source_name": args.source_name,
        "target_name": args.target_name,
        "source_dataset": str(args.source_dataset.resolve()),
        "source_dataset_sha256": sha256(args.source_dataset),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256(args.source_manifest),
        "target_dataset": str(args.target_dataset.resolve()),
        "target_dataset_sha256": sha256(args.target_dataset),
        "target_manifest": str(args.target_manifest.resolve()),
        "target_manifest_sha256": sha256(args.target_manifest),
        "source_rows_excluded_by_internal_test_split": int(
            source_manifest["split"].eq("test").sum()
        ),
        "target_rows_before_cross_duplicate_removal": int(len(target)),
        "target_rows_removed_as_exact_cross_dataset_duplicates": int(
            target_is_cross_duplicate.sum()
        ),
        "removed_cross_dataset_label_conflicts": int(conflict_count),
        "target_internal_duplicate_url_rows": int(target["url"].duplicated(False).sum()),
        "split_counts": split_counts,
        "cohort_sha256": sha256(cohort_path),
        "manifest_sha256": sha256(manifest_path),
        "source_transfer_manifest_sha256": sha256(source_transfer_manifest_path),
        "target_transfer_manifest_sha256": sha256(target_transfer_manifest_path),
        "cross_partition_exact_url_overlap": {
            "train_test": len(
                set(cohort.loc[manifest["split"].eq("train"), "url"])
                & set(cohort.loc[manifest["split"].eq("test"), "url"])
            ),
            "validation_test": len(
                set(cohort.loc[manifest["split"].eq("validation"), "url"])
                & set(cohort.loc[manifest["split"].eq("test"), "url"])
            ),
        },
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
