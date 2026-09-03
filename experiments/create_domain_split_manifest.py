"""Create one deterministic, label-balanced, eSLD-disjoint split manifest.

All baseline pipelines must consume the resulting row assignments instead of
calling their own random split functions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tldextract
from sklearn.model_selection import StratifiedGroupKFold


EXTRACT = tldextract.TLDExtract(suffix_list_urls=None)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def esld(url: str) -> str:
    value = str(url).strip()
    parsed = urlsplit(value if "://" in value else f"//{value}")
    host = (parsed.hostname or "").strip(".").lower()
    if host:
        extracted = EXTRACT(host)
        registered = getattr(extracted, "top_domain_under_public_suffix", None)
        if registered is None:
            registered = extracted.registered_domain
        return registered or host
    return "invalid:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def split_score(indices: np.ndarray, labels: np.ndarray, target: float) -> float:
    total_term = abs(len(indices) / len(labels) - target) / target
    class_terms = []
    for label in (0, 1):
        denom = max(int((labels == label).sum()), 1)
        class_terms.append(abs(int((labels[indices] == label).sum()) / denom - target) / target)
    return total_term + sum(class_terms)


def select_fold(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    target: float,
    seed: int,
    candidates: int,
) -> tuple[np.ndarray, dict[str, object]]:
    placeholder = np.zeros(len(labels), dtype=np.uint8)
    best: tuple[float, int, int, np.ndarray] | None = None
    for offset in range(candidates):
        candidate_seed = seed + offset
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=candidate_seed
        )
        for fold, (_, held_out) in enumerate(splitter.split(placeholder, labels, groups)):
            score = split_score(held_out, labels, target)
            item = (score, candidate_seed, fold, held_out)
            if best is None or item[:3] < best[:3]:
                best = item
    assert best is not None
    return best[3], {"balance_score": best[0], "candidate_seed": best[1], "fold": best[2]}


def counts(frame: pd.DataFrame, split: str) -> dict[str, int]:
    part = frame[frame["split"] == split]
    return {
        "rows": len(part),
        "benign": int((part["label"] == 0).sum()),
        "phishing": int((part["label"] == 1).sum()),
        "groups": int(part["esld"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=100)
    args = parser.parse_args()

    data = pd.read_csv(args.dataset, encoding="ISO-8859-1", usecols=["url", "phishing"])
    data = data.reset_index(drop=True)
    data["phishing"] = data["phishing"].astype(int)
    if not set(data["phishing"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be encoded as benign=0 and phishing=1")

    labels = data["phishing"].to_numpy()
    groups = data["url"].astype(str).map(esld).to_numpy()
    all_indices = np.arange(len(data))
    test_idx, test_choice = select_fold(
        labels, groups, n_splits=10, target=0.1, seed=args.seed, candidates=args.candidates
    )
    remaining = np.setdiff1d(all_indices, test_idx, assume_unique=True)
    val_relative, val_choice = select_fold(
        labels[remaining],
        groups[remaining],
        n_splits=9,
        target=1 / 9,
        seed=args.seed + args.candidates,
        candidates=args.candidates,
    )
    val_idx = remaining[val_relative]
    train_idx = np.setdiff1d(remaining, val_idx, assume_unique=True)

    assignment = np.full(len(data), "", dtype=object)
    assignment[train_idx] = "train"
    assignment[val_idx] = "validation"
    assignment[test_idx] = "test"
    manifest = pd.DataFrame(
        {
            "row_id": all_indices,
            "split": assignment,
            "label": labels,
            "esld": groups,
        }
    )

    group_sets = {
        split: set(manifest.loc[manifest["split"] == split, "esld"])
        for split in ("train", "validation", "test")
    }
    overlaps = {
        "train_validation": len(group_sets["train"] & group_sets["validation"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "validation_test": len(group_sets["validation"] & group_sets["test"]),
    }
    if any(overlaps.values()) or (manifest["split"] == "").any():
        raise RuntimeError(f"Invalid split assignment: {overlaps}")

    name = args.name or args.dataset.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"{name}_domain_split_seed{args.seed}.csv"
    summary_path = args.output_dir / f"{name}_domain_split_seed{args.seed}.json"
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "manifest_sha256": file_sha256(manifest_path),
        "seed": args.seed,
        "candidate_count_per_stage": args.candidates,
        "method": "StratifiedGroupKFold candidate selection; eSLD groups; offline PSL snapshot",
        "positive_class": {"value": 1, "name": "phishing"},
        "test_choice": test_choice,
        "validation_choice": val_choice,
        "counts": {split: counts(manifest, split) for split in ("train", "validation", "test")},
        "group_overlap": overlaps,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
