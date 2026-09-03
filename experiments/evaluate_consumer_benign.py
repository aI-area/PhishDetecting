#!/usr/bin/env python3
"""Evaluate a frozen LitePhish model on a filtered popular-DNS benign proxy."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
import tldextract


EXTRACT = tldextract.TLDExtract(suffix_list_urls=())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def esld(value: str) -> str:
    host = (urlsplit(value).hostname or value).lower().rstrip(".")
    result = EXTRACT(host)
    registered = getattr(result, "top_domain_under_public_suffix", None)
    if registered is None:
        registered = getattr(result, "registered_domain", "")
    return registered or host


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    if not total:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - radius, centre + radius


def read_known_urls(paths: list[Path]) -> tuple[set[str], set[str], list[dict]]:
    exact: set[str] = set()
    domains: set[str] = set()
    audit = []
    for path in paths:
        frame = pd.read_csv(path)
        url_col = "url" if "url" in frame else frame.columns[0]
        labels = frame["phishing"] if "phishing" in frame else pd.Series(1, index=frame.index)
        phishing = frame.loc[pd.to_numeric(labels, errors="coerce").fillna(1).astype(int) == 1, url_col].astype(str)
        exact.update(phishing)
        domains.update(esld(value) for value in phishing)
        audit.append({"path": str(path.resolve()), "sha256": sha256(path), "rows": len(frame),
                      "phishing_rows_used_for_exclusion": len(phishing)})
    return exact, domains, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("umbrella_zip", type=Path)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("scaler_path", type=Path)
    parser.add_argument("pipeline_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--known-dataset", action="append", type=Path, default=[])
    parser.add_argument("--top-ranks", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.pipeline_root))
    sys.path.insert(0, str(args.pipeline_root / "experiments"))
    from feature_extraction import FeatureExtractor
    from experiments.run_litephish_experiments import predict_probabilities

    processor_path = args.artifact_dir / "ngram_processor.pkl"
    selected_path = args.artifact_dir / "selected_features.pkl"
    processor = joblib.load(processor_path)
    selected = np.asarray(joblib.load(selected_path), dtype=int)
    model = joblib.load(args.model_path)
    scaler = joblib.load(args.scaler_path)

    with zipfile.ZipFile(args.umbrella_zip) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as stream:
            ranked = pd.read_csv(stream, header=None, names=["rank", "domain"], nrows=args.top_ranks)
        source_time = archive.getinfo(member).date_time
    ranked["domain"] = ranked["domain"].astype(str).str.lower().str.rstrip(".")
    ranked = ranked[ranked["domain"].str.match(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9-]{2,63}$")]
    ranked = ranked.drop_duplicates("domain", keep="first").copy()
    ranked["url"] = "https://" + ranked["domain"] + "/"
    ranked["esld"] = ranked["domain"].map(esld)

    known_exact, known_domains, known_audit = read_known_urls(args.known_dataset)
    ranked["excluded_exact_known_phishing"] = ranked["url"].isin(known_exact)
    ranked["excluded_esld_known_phishing"] = ranked["esld"].isin(known_domains)
    cohort = ranked.loc[~(ranked["excluded_exact_known_phishing"] | ranked["excluded_esld_known_phishing"])].copy()
    cohort["phishing"] = 0
    cohort_path = args.output_dir / "umbrella_popular_dns_benign_proxy.csv.gz"
    cohort.to_csv(cohort_path, index=False, compression="gzip")

    extractor = FeatureExtractor()
    probabilities: list[np.ndarray] = []
    start_time = time.perf_counter()
    for start in range(0, len(cohort), args.batch_size):
        urls = cohort["url"].iloc[start:start + args.batch_size].astype(str)
        handcrafted = np.asarray([extractor.generate_features(url) for url in urls], dtype=np.float32)
        ngram = processor.transform(urls).toarray().astype(np.float32, copy=False)
        combined = np.hstack((handcrafted, ngram)).astype(np.float32, copy=False)
        prepared = scaler.transform(combined[:, selected])
        probabilities.append(np.asarray(predict_probabilities(model, prepared), dtype=float))
    elapsed = time.perf_counter() - start_time
    probability = np.concatenate(probabilities)
    prediction = (probability >= args.threshold).astype(int)
    false_positives = int(prediction.sum())
    low, high = wilson(false_positives, len(prediction))
    predictions = cohort[["rank", "domain", "esld", "url", "phishing"]].copy()
    predictions["probability"] = probability
    predictions["prediction"] = prediction
    prediction_path = args.output_dir / "LitePhish_consumer_benign_predictions.csv.gz"
    predictions.to_csv(prediction_path, index=False, compression="gzip")

    rank_strata = []
    for ceiling in (1000, 10000, 100000):
        mask = predictions["rank"].le(ceiling).to_numpy()
        count = int(mask.sum()); flagged = int(prediction[mask].sum())
        stratum_low, stratum_high = wilson(flagged, count)
        rank_strata.append({"rank_ceiling": ceiling, "n_retained": count, "false_positives": flagged,
                            "false_positive_rate": flagged / count if count else None,
                            "false_positive_rate_wilson_95_ci": [stratum_low, stratum_high]})
    result = {
        "status": "PASS",
        "experiment": "frozen LitePhish on consumer-realistic popular-DNS benign proxy",
        "claim_boundary": "Umbrella ranks reflect DNS popularity; labels are likely-benign proxies, not individually verified benign browser visits.",
        "source": {"url": "https://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip",
                   "archive": str(args.umbrella_zip.resolve()), "sha256": sha256(args.umbrella_zip),
                   "member_timestamp": source_time, "requested_top_ranks": args.top_ranks},
        "filtering": {"candidate_rows": len(ranked), "retained_rows": len(cohort),
                      "excluded_exact_known_phishing": int(ranked["excluded_exact_known_phishing"].sum()),
                      "excluded_esld_known_phishing": int(ranked["excluded_esld_known_phishing"].sum()),
                      "known_datasets": known_audit},
        "model": {"retraining_performed": False, "threshold": args.threshold,
                  "selected_features": len(selected), "model_sha256": sha256(args.model_path),
                  "scaler_sha256": sha256(args.scaler_path), "ngram_processor_sha256": sha256(processor_path),
                  "selected_features_sha256": sha256(selected_path)},
        "result": {"n_likely_benign": len(prediction), "false_positives": false_positives,
                   "false_positive_rate": false_positives / len(prediction),
                   "false_positive_rate_wilson_95_ci": [low, high],
                   "mean_probability": float(probability.mean()), "median_probability": float(np.median(probability)),
                   "evaluation_seconds": elapsed, "cumulative_rank_strata": rank_strata},
        "outputs": {"cohort": str(cohort_path.resolve()), "cohort_sha256": sha256(cohort_path),
                    "predictions": str(prediction_path.resolve()), "predictions_sha256": sha256(prediction_path)},
    }
    (args.output_dir / "experiment_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
