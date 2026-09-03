#!/usr/bin/env python3
"""Create an audited, leakage-filtered prospective phishing cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import tldextract


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)


def registrable_domain(url: str) -> str:
    hostname = (urlsplit(url).hostname or "").lower().rstrip(".")
    if not hostname:
        return ""
    parts = EXTRACT(hostname)
    registered = getattr(parts, "top_domain_under_public_suffix", None)
    if registered is None:
        registered = getattr(parts, "registered_domain", "")
    return registered or hostname


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("openphish_feed", type=Path)
    parser.add_argument("training_dataset", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--snapshot-utc", required=True)
    parser.add_argument("--model-freeze-utc", required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = pd.read_csv(args.training_dataset, encoding="latin1")
    if "url" not in training:
        raise ValueError("training dataset must contain a url column")
    training_urls = set(training["url"].dropna().astype(str).str.strip())
    training_domains = {registrable_domain(url) for url in training_urls}
    training_domains.discard("")

    raw_lines = args.openphish_feed.read_text(encoding="utf-8", errors="replace").splitlines()
    records = []
    invalid = 0
    for position, value in enumerate(raw_lines):
        url = value.strip()
        parsed = urlsplit(url)
        if not url or parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            invalid += 1
            continue
        records.append({"feed_row": position, "url": url, "phishing": 1,
                        "registrable_domain": registrable_domain(url)})
    frame = pd.DataFrame(records)
    valid_before_dedup = len(frame)
    frame = frame.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    exact_overlap = frame["url"].isin(training_urls)
    domain_overlap = frame["registrable_domain"].isin(training_domains)
    retained = frame.loc[~exact_overlap & ~domain_overlap].copy().reset_index(drop=True)
    retained.insert(0, "row_id", range(len(retained)))
    retained["source"] = "OpenPhish community feed"
    retained["snapshot_utc"] = args.snapshot_utc

    cohort_path = args.output_dir / "openphish_post_freeze_cohort.csv"
    retained.to_csv(cohort_path, index=False)
    audit = {
        "experiment": "prospective temporal generalization",
        "model_freeze_utc": args.model_freeze_utc,
        "snapshot_utc": args.snapshot_utc,
        "source": "official OpenPhish community feed",
        "label_semantics": "all feed entries are phishing positives",
        "openphish_feed": str(args.openphish_feed.resolve()),
        "openphish_feed_sha256": sha256(args.openphish_feed),
        "training_dataset": str(args.training_dataset.resolve()),
        "training_dataset_sha256": sha256(args.training_dataset),
        "training_dataset_rows": int(len(training)),
        "raw_feed_lines": int(len(raw_lines)),
        "invalid_or_empty_lines": int(invalid),
        "valid_rows_before_deduplication": int(valid_before_dedup),
        "exact_duplicate_urls_removed": int(valid_before_dedup - len(frame)),
        "exact_training_url_overlaps_removed": int(exact_overlap.sum()),
        "training_registrable_domain_overlaps_removed": int((~exact_overlap & domain_overlap).sum()),
        "retained_rows": int(len(retained)),
        "retained_unique_domains": int(retained["registrable_domain"].nunique()),
        "cohort_csv": str(cohort_path.resolve()),
        "cohort_sha256": sha256(cohort_path),
        "leakage_policy": "remove exact URL and registrable-domain overlap against the complete source dataset",
        "temporal_claim_boundary": "prospective post-freeze phishing recall; precision/F1 are undefined for an all-positive cohort",
    }
    (args.output_dir / "cohort_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
