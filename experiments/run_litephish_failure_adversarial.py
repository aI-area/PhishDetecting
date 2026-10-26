#!/usr/bin/env python3
"""Natural failure strata and paired counterfactual stress tests for LitePhish."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import joblib
import numpy as np
import pandas as pd
import tldextract
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


SHORTENERS = {"bit.ly", "bitly.com", "goo.gl", "tinyurl.com", "t.co", "is.gd", "buff.ly", "adf.ly", "bc.vc"}
SHARED_HOSTING = {
    "weebly.com", "weeblysite.com", "webflow.io", "godaddysites.com", "wixsite.com",
    "netlify.app", "herokuapp.com", "000webhost.com", "000webhostapp.com", "wordpress.com",
    "blogspot.com", "sites.google.com", "firebaseapp.com", "github.io", "squarespace.com",
    "myshopify.com", "glitch.me", "repl.co", "surge.sh", "now.sh", "vercel.app", "pages.dev",
    "sharepoint.com", "s3.amazonaws.com", "drive.google.com", "docs.google.com", "dropbox.com",
    "mediafire.com", "onedrive.live.com", "box.com",
}
REPUTABLE_DOMAINS = {
    "google.com", "microsoft.com", "apple.com", "amazon.com", "youtube.com", "facebook.com",
    "twitter.com", "instagram.com", "linkedin.com", "github.com", "yahoo.com", "netflix.com",
    "wikipedia.org", "baidu.com", "live.com", "bing.com", "ebay.com", "pinterest.com",
    "reddit.com", "dropbox.com", "whatsapp.com",
}
BRAND_TOKENS = (
    "paypal", "microsoft", "office365", "apple", "google", "facebook", "instagram", "amazon",
    "netflix", "twitter", "linkedin", "adobe", "outlook", "icloud", "dropbox", "whatsapp",
    "bank", "credit", "debit",
)
SUSPICIOUS_TOKENS = (
    "secure-login", "verify-account", "confirm-identity", "update-billing", "password-reset",
    "account-verify", "banking-login", "payment-update", "identity-confirm", "signin-verify",
    "password-update", "secure-signin", "wallet-restore", "verify-now", "account-suspended",
    "unusual-activity", "account-limitation", "security-alert", "validation-required",
    "authorize-login", "validate-account", "authentication-required", "myaccount-login",
    "verification", "authenticate", "paypal", "microsoft", "apple", "google", "facebook",
    "instagram", "amazon", "netflix", "twitter", "linkedin", "adobe", "outlook", "icloud",
    "signin", "validate", "authorize", "authentication", "session", "myaccount", "confirm",
    "invoice", "payment", "password", "account", "login",
)
EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - radius, centre + radius


def host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().rstrip(".")


def domain_matches(hostname: str, domains: set[str]) -> bool:
    return any(hostname == item or hostname.endswith("." + item) for item in domains)


def is_ip(hostname: str) -> bool:
    parts = hostname.split(".")
    return len(parts) == 4 and all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)


def natural_flags(url: str) -> dict[str, bool]:
    parsed = urlsplit(url)
    hostname = host(url)
    lower = url.lower()
    extracted = EXTRACT(hostname)
    registered = getattr(extracted, "top_domain_under_public_suffix", None)
    if registered is None:
        registered = getattr(extracted, "registered_domain", "")
    subdepth = len([part for part in extracted.subdomain.split(".") if part])
    pathdepth = len([part for part in parsed.path.split("/") if part])
    query_count = 0 if not parsed.query else len(parsed.query.split("&"))
    suspicious = any(token in lower for token in SUSPICIOUS_TOKENS)
    obfuscated = ("%" in url or "@" in url or "xn--" in lower or is_ip(hostname) or
                  bool(re.search(r"[a-z]\d[a-z]|\d[a-z]\d", lower)) or subdepth >= 4)
    clean = (len(url) <= 80 and not suspicious and not obfuscated and
             hostname.count("-") <= 1 and subdepth <= 1 and pathdepth <= 2 and query_count <= 1)
    return {
        "natural_shortener": domain_matches(hostname, SHORTENERS),
        "natural_shared_or_reputable_host": domain_matches(hostname, SHARED_HOSTING | REPUTABLE_DOMAINS),
        "natural_clean_looking": clean,
        "natural_obfuscated": obfuscated,
        "registered_domain_present": bool(registered),
    }


def replace_tokens(url: str, tokens: tuple[str, ...], replacement: str) -> str | None:
    changed = url
    for token in sorted(tokens, key=len, reverse=True):
        changed = re.sub(re.escape(token), replacement, changed, flags=re.IGNORECASE)
    return changed if changed != url else None


def separator_obfuscation(url: str) -> str | None:
    lower = url.lower()
    candidates = [token for token in SUSPICIOUS_TOKENS if len(token) >= 5 and token in lower]
    if not candidates:
        return None
    token = max(candidates, key=len)
    midpoint = len(token) // 2
    replacement = token[:midpoint] + "-" + token[midpoint:]
    return re.sub(re.escape(token), replacement, url, count=1, flags=re.IGNORECASE)


def clean_surface(url: str) -> str | None:
    parsed = urlsplit(url)
    if parsed.path in {"", "/"} and not parsed.query and not parsed.fragment:
        return None
    return urlunsplit((parsed.scheme or "https", parsed.netloc, "/", "", ""))


def shortener_wrap(url: str) -> str:
    return "https://bit.ly/" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]


def reputable_host_proxy(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme or "https", "www.google.com", parsed.path or "/", parsed.query, parsed.fragment))


def percent_encode_path(url: str) -> str | None:
    parsed = urlsplit(url)
    match = re.search(r"[A-Za-z]", parsed.path)
    if not match:
        return None
    character = match.group(0)
    encoded = "%" + format(ord(character), "02X")
    path = parsed.path[:match.start()] + encoded + parsed.path[match.end():]
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def holm(values: list[float]) -> list[float]:
    count = len(values); order = np.argsort(values); adjusted = np.empty(count, dtype=float); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_delta(original: np.ndarray, transformed: np.ndarray, seed: int = 42,
                    repetitions: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed); count = len(original); values = np.empty(repetitions)
    for index in range(repetitions):
        selection = rng.integers(0, count, count)
        values[index] = transformed[selection].mean() - original[selection].mean()
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("split_cache", type=Path)
    parser.add_argument("pipeline_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--gamma", type=float, default=0.55)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--n-estimators", type=int, default=500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.pipeline_root))
    sys.path.insert(0, str(args.pipeline_root / "experiments"))
    from feature_extraction import FeatureExtractor
    from main import PhishingDetector
    from model import FocalLossLGBM
    from ngram_processing import NgramProcessor
    from experiments.run_litephish_experiments import predict_probabilities
    setattr(sys.modules["__main__"], "NgramProcessor", NgramProcessor)
    setattr(sys.modules["__main__"], "FocalLossLGBM", FocalLossLGBM)

    processor_path = args.artifact_dir / "ngram_processor.pkl"
    selected_path = args.artifact_dir / "selected_features.pkl"
    processor = joblib.load(processor_path); selected = np.asarray(joblib.load(selected_path), dtype=int)
    cache = joblib.load(args.split_cache)
    if cache["dataset_sha256"] != sha256(args.dataset) or cache["processor_sha256"] != sha256(processor_path):
        raise RuntimeError("feature cache provenance mismatch")

    detector = PhishingDetector(); urls, labels = detector.load_data(str(args.dataset))
    train_idx, validation_idx, test_idx = detector.split_data_indices(urls, labels)
    if cache["train_idx_sha256"] != array_sha256(train_idx) or cache["test_idx_sha256"] != array_sha256(test_idx):
        raise RuntimeError("feature cache split mismatch")
    y_train = labels.iloc[train_idx].to_numpy(dtype=int); y_test = labels.iloc[test_idx].to_numpy(dtype=int)
    x_train = cache["x_train"]; x_test = cache["x_test"]
    scaler = StandardScaler().fit(x_train); x_train_scaled = scaler.transform(x_train)
    model = FocalLossLGBM(
        gamma=args.gamma, alpha=args.alpha, boosting_type="gbdt", num_leaves=50, max_depth=-1,
        learning_rate=0.15, n_estimators=args.n_estimators, min_child_samples=40,
        reg_alpha=0.1, reg_lambda=0.3, verbosity=-1, random_state=42, device="cpu",
        n_jobs=args.jobs, deterministic=True, force_col_wise=True,
    )
    train_start = time.perf_counter(); model.fit(x_train_scaled, y_train); training_seconds = time.perf_counter() - train_start
    original_all_probability = np.asarray(predict_probabilities(model, scaler.transform(x_test)), dtype=float)
    positive_mask = y_test == 1
    positive_urls = urls.iloc[test_idx].reset_index(drop=True)[positive_mask].reset_index(drop=True)
    positive_source_rows = np.asarray(test_idx)[positive_mask]
    original_probability = original_all_probability[positive_mask]
    original_prediction = (original_probability >= args.threshold).astype(int)

    flags = pd.DataFrame([natural_flags(url) for url in positive_urls])
    natural_rows = []
    for category in ("all_positive", "natural_shortener", "natural_shared_or_reputable_host",
                     "natural_clean_looking", "natural_obfuscated"):
        mask = np.ones(len(flags), dtype=bool) if category == "all_positive" else flags[category].to_numpy(dtype=bool)
        count = int(mask.sum()); detected = int(original_prediction[mask].sum()) if count else 0
        lower, upper = wilson(detected, count)
        natural_rows.append({"category": category, "n": count, "detected": detected,
                             "missed": count - detected, "recall": detected / count if count else float("nan"),
                             "recall_ci95_low": lower, "recall_ci95_high": upper,
                             "mean_probability": float(original_probability[mask].mean()) if count else float("nan")})
    natural_path = args.output_dir / "natural_failure_strata.csv"
    pd.DataFrame(natural_rows).to_csv(natural_path, index=False)

    transformations = {
        "brand_token_mask": lambda value: replace_tokens(value, BRAND_TOKENS, "portal"),
        "clean_surface": clean_surface,
        "shortener_wrap_proxy": shortener_wrap,
        "reputable_host_proxy": reputable_host_proxy,
        "separator_obfuscation": separator_obfuscation,
        "percent_encoded_path": percent_encode_path,
    }
    extractor = FeatureExtractor()

    def probabilities_for(values: pd.Series) -> np.ndarray:
        result = []
        for start in range(0, len(values), args.batch_size):
            batch = values.iloc[start:start + args.batch_size].astype(str)
            handcrafted = np.asarray(Parallel(n_jobs=args.jobs, prefer="threads")(
                delayed(extractor.generate_features)(url) for url in batch), dtype=np.float32)
            ngrams = processor.transform(batch).toarray().astype(np.float32, copy=False)
            compact = np.hstack([handcrafted, ngrams])[:, selected].astype(np.float32, copy=False)
            result.append(np.asarray(predict_probabilities(model, scaler.transform(compact)), dtype=float))
        return np.concatenate(result)

    prediction_records = []; comparison_rows = []
    for transform_name, transform in transformations.items():
        transformed = positive_urls.map(transform)
        applicable = transformed.notna() & (transformed != positive_urls)
        indices = np.flatnonzero(applicable.to_numpy())
        if len(indices) == 0:
            continue
        transformed_values = transformed.iloc[indices].astype(str).reset_index(drop=True)
        transformed_probability = probabilities_for(transformed_values)
        transformed_prediction = (transformed_probability >= args.threshold).astype(int)
        original_subset_probability = original_probability[indices]
        original_subset_prediction = original_prediction[indices]
        lost = int(((original_subset_prediction == 1) & (transformed_prediction == 0)).sum())
        gained = int(((original_subset_prediction == 0) & (transformed_prediction == 1)).sum())
        discordant = lost + gained
        p_raw = float(binomtest(min(lost, gained), discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
        delta = float(transformed_prediction.mean() - original_subset_prediction.mean())
        low, high = bootstrap_delta(original_subset_prediction, transformed_prediction)
        comparison_rows.append({
            "transformation": transform_name, "n_applicable": len(indices),
            "original_recall": float(original_subset_prediction.mean()),
            "transformed_recall": float(transformed_prediction.mean()),
            "recall_delta": delta, "recall_delta_bootstrap_ci95_low": low,
            "recall_delta_bootstrap_ci95_high": high,
            "original_mean_probability": float(original_subset_probability.mean()),
            "transformed_mean_probability": float(transformed_probability.mean()),
            "mean_probability_delta": float((transformed_probability - original_subset_probability).mean()),
            "original_detected_transformed_missed": lost,
            "original_missed_transformed_detected": gained,
            "exact_mcnemar_p_raw": p_raw,
        })
        for local, source_index in enumerate(indices):
            prediction_records.append({
                "transformation": transform_name, "positive_index": int(source_index),
                "dataset_row_id": int(positive_source_rows[source_index]),
                "original_url": positive_urls.iloc[source_index], "transformed_url": transformed_values.iloc[local],
                "original_probability": original_subset_probability[local],
                "transformed_probability": transformed_probability[local],
                "original_prediction": original_subset_prediction[local],
                "transformed_prediction": transformed_prediction[local],
            })
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["exact_mcnemar_p_holm"] = holm(comparisons.exact_mcnemar_p_raw.tolist())
    comparison_path = args.output_dir / "counterfactual_comparisons.csv"; comparisons.to_csv(comparison_path, index=False)
    prediction_path = args.output_dir / "counterfactual_predictions.csv.gz"
    pd.DataFrame(prediction_records).to_csv(prediction_path, index=False, compression="gzip")

    cases = pd.DataFrame({"positive_index": range(len(positive_urls)), "dataset_row_id": positive_source_rows,
                          "url": positive_urls, "probability": original_probability,
                          "prediction": original_prediction}).join(flags)
    cases_path = args.output_dir / "natural_positive_cases.csv.gz"; cases.to_csv(cases_path, index=False, compression="gzip")
    model_path = args.output_dir / "LitePhish_failure_analysis_model.joblib"; scaler_path = args.output_dir / "LitePhish_failure_analysis_scaler.joblib"
    joblib.dump(model, model_path); joblib.dump(scaler, scaler_path)
    artifact_hashes = {path.name: sha256(path) for path in (processor_path, selected_path, args.split_cache, model_path, scaler_path)}
    audit = {
        "status": "PASS", "experiment": "LitePhish natural failure and adversarial counterfactual evaluation",
        "dataset": str(args.dataset.resolve()), "dataset_sha256": sha256(args.dataset),
        "split": {"method": "released domain-stratified split", "test_rows": len(test_idx),
                  "test_positive_rows": int(positive_mask.sum()), "test_idx_sha256": array_sha256(test_idx)},
        "model": {"gamma": args.gamma, "alpha": args.alpha, "n_estimators": args.n_estimators,
                  "selected_features": len(selected), "threshold": args.threshold,
                  "training_seconds": training_seconds, "retrained_on": "training partition only"},
        "natural_categories_predefined_before_prediction_review": True,
        "counterfactuals": {
            "independent_unit": "original held-out phishing URL", "paired": True,
            "claim_boundary": "synthetic transformations are conditional stress-test proxies, not verified live phishing URLs",
            "primary_response": "paired change in phishing detection at threshold 0.5",
            "inference": "10,000 paired bootstrap CI and exact McNemar/binomial test; Holm correction across transformations",
        },
        "artifact_sha256": artifact_hashes,
        "outputs": {path.name: sha256(path) for path in (natural_path, comparison_path, prediction_path, cases_path)},
        "software": {"python": platform.python_version()},
    }
    audit_path = args.output_dir / "experiment_audit.json"; audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(pd.DataFrame(natural_rows).to_string(index=False)); print(comparisons.to_string(index=False)); print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
