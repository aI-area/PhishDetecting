"""Run the principal LitePhish robustness and component analyses.

The suite includes:

* SDCS isolation and SDCS x learner interaction.
* Focal-loss alternatives and alpha/gamma sensitivity.
* Repeated-run statistics and pairwise tests.
* Retained/discarded feature inventory.
* Compression under domain/cross-dataset generalization.
* Handcrafted vs n-gram stability.
* External-cohort overlap and benign-distribution analysis.
* End-to-end latency measurement.
* Perturbation-based adversarial stress tests.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import pickle
import random
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from tldextract import extract as extract_tld_parts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def default_data_file(filename: str) -> str:
    """Prefer the repo-level data directory, with dataset as local fallback."""
    for data_dir in (ROOT / "data", ROOT / "dataset"):
        candidate = data_dir / filename
        if candidate.exists():
            return str(candidate)
    return str(ROOT / "data" / filename)

from feature_extraction import FeatureExtractor  # noqa: E402
from model import FocalLossLGBM  # noqa: E402
from ngram_processing import NgramProcessor  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402


LOGGER = logging.getLogger("litephish_experiments")

DEFAULT_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]
HANDCRAFTED_COUNT = len(FeatureExtractor.FEATURE_NAMES)
NOVEL_FEATURES = {
    "Path Entropy Variance",
    "Rolling Entropy Range",
    "Subdomain Entropy Gradient",
    "Cookie Param Entropy",
    "Homoglyph Score",
    "Special Char Cluster",
    "Pattern Density",
    "Obfuscation Chars",
    "Encoding Keywords",
    "Redirect Keywords",
    "Consecutive Special Chars",
    "Encoding Bypass Ratio",
    "Redirect TLD Disparity",
}
HOSTING_CLEAN_LINK_TERMS = [
    "sites.google.",
    "sharepoint.",
    "github.io",
    "pages.dev",
    "weebly.",
    "wixsite.",
    "netlify.",
    "vercel.",
    "webflow.",
    "blogspot.",
    "wordpress.",
]


@dataclass
class PreparedData:
    train_urls: pd.Series
    val_urls: pd.Series
    test_urls: pd.Series
    train_y: pd.Series
    val_y: pd.Series
    test_y: pd.Series
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    mode: str
    train_path: str
    test_path: str


@dataclass
class FeatureBlocks:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    feature_names: List[str]
    ngram_processor: NgramProcessor


@dataclass
class InferenceBundle:
    model: Any
    scaler: StandardScaler
    ngram_processor: NgramProcessor
    selected_indices: np.ndarray
    feature_names: List[str]
    selected_names: List[str]
    threshold: float = 0.5


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("lightgbm").setLevel(logging.ERROR)


def stable_hash(parts: Iterable[Any]) -> str:
    h = hashlib.sha1()
    for part in parts:
        if isinstance(part, bytes):
            chunk = part
        else:
            chunk = str(part).encode("utf-8", errors="ignore")
        h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()[:16]


def read_csv_dataset(path: Path, sample: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    if "url" not in df.columns:
        if "URL" in df.columns:
            df = df.rename(columns={"URL": "url"})
        else:
            raise ValueError(f"{path} must contain a url column")
    if "phishing" not in df.columns:
        if "label" in df.columns:
            df = df.rename(columns={"label": "phishing"})
        else:
            raise ValueError(f"{path} must contain a phishing/label column")
    df = df[["url", "phishing"] + [c for c in df.columns if c not in {"url", "phishing"}]].copy()
    df["url"] = df["url"].astype(str)
    df["phishing"] = df["phishing"].astype(int)
    df = df.dropna(subset=["url", "phishing"]).reset_index(drop=True)
    if sample and sample > 0 and sample < len(df):
        sampled_parts = []
        for _, group in df.groupby("phishing", group_keys=False):
            n_group = max(1, int(round(sample * len(group) / len(df))))
            sampled_parts.append(group.sample(n=min(n_group, len(group)), random_state=seed))
        df = pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if len(df) > sample:
            df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
    return df


def normalize_url(url: str) -> str:
    text = str(url).strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        text = "http://" + text
    try:
        parsed = urllib.parse.urlsplit(text)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        if scheme == "http" and netloc.endswith(":80"):
            netloc = netloc[:-3]
        if scheme == "https" and netloc.endswith(":443"):
            netloc = netloc[:-4]
        path = urllib.parse.unquote(parsed.path or "/")
        path = re.sub(r"/+", "/", path).rstrip("/") or "/"
        query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        query = urllib.parse.urlencode(sorted(query_pairs))
        return urllib.parse.urlunsplit((scheme, netloc, path, query, "")).lower()
    except Exception:
        return text.lower().rstrip("/")


def root_domain(url: str) -> str:
    try:
        ext = extract_tld_parts(str(url))
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
        return (ext.domain or str(url)).lower()
    except Exception:
        return "unknown"


def domain_groups(urls: pd.Series) -> np.ndarray:
    return np.array([root_domain(u) for u in urls], dtype=object)


def split_internal(
    df: pd.DataFrame,
    seed: int,
    split: str = "domain",
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = df["phishing"].values
    indices = np.arange(len(df))
    if split == "random":
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=val_size + test_size,
            random_state=seed,
            stratify=labels,
        )
        temp_labels = labels[temp_idx]
        rel_val, rel_test = train_test_split(
            np.arange(len(temp_idx)),
            test_size=test_size / (val_size + test_size),
            random_state=seed,
            stratify=temp_labels,
        )
        return train_idx, temp_idx[rel_val], temp_idx[rel_test]

    groups = domain_groups(df["url"])
    outer = GroupShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=seed)
    train_idx, temp_idx = next(outer.split(df["url"], labels, groups))
    temp_groups = groups[temp_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=test_size / (val_size + test_size), random_state=seed)
    rel_val, rel_test = next(inner.split(df.iloc[temp_idx]["url"], labels[temp_idx], temp_groups))
    return train_idx, temp_idx[rel_val], temp_idx[rel_test]


def split_train_val(df: pd.DataFrame, seed: int, split: str = "domain", val_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    labels = df["phishing"].values
    indices = np.arange(len(df))
    if split == "random":
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,
            random_state=seed,
            stratify=labels,
        )
        return train_idx, val_idx

    groups = domain_groups(df["url"])
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    return next(gss.split(df["url"], labels, groups))


def prepare_data(args: argparse.Namespace, seed: int) -> PreparedData:
    dataset = Path(args.dataset)
    if args.train_dataset:
        dataset = Path(args.train_dataset)
    train_df = read_csv_dataset(dataset, sample=args.sample, seed=seed)

    if args.test_dataset:
        test_df = read_csv_dataset(Path(args.test_dataset), sample=args.test_sample or args.sample, seed=seed)
        if args.dedup_cross_dataset:
            train_norm = set(train_df["url"].map(normalize_url))
            before = len(test_df)
            test_df = test_df[~test_df["url"].map(normalize_url).isin(train_norm)].reset_index(drop=True)
            LOGGER.info("Removed %d exact URL overlaps from external test set", before - len(test_df))
        train_idx, val_idx = split_train_val(train_df, seed, split=args.split)
        test_idx = np.arange(len(test_df))
        return PreparedData(
            train_urls=train_df.loc[train_idx, "url"].reset_index(drop=True),
            val_urls=train_df.loc[val_idx, "url"].reset_index(drop=True),
            test_urls=test_df["url"].reset_index(drop=True),
            train_y=train_df.loc[train_idx, "phishing"].reset_index(drop=True),
            val_y=train_df.loc[val_idx, "phishing"].reset_index(drop=True),
            test_y=test_df["phishing"].reset_index(drop=True),
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            mode="cross",
            train_path=str(dataset),
            test_path=str(Path(args.test_dataset)),
        )

    train_idx, val_idx, test_idx = split_internal(train_df, seed, split=args.split)
    return PreparedData(
        train_urls=train_df.loc[train_idx, "url"].reset_index(drop=True),
        val_urls=train_df.loc[val_idx, "url"].reset_index(drop=True),
        test_urls=train_df.loc[test_idx, "url"].reset_index(drop=True),
        train_y=train_df.loc[train_idx, "phishing"].reset_index(drop=True),
        val_y=train_df.loc[val_idx, "phishing"].reset_index(drop=True),
        test_y=train_df.loc[test_idx, "phishing"].reset_index(drop=True),
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        mode=args.split,
        train_path=str(dataset),
        test_path=str(dataset),
    )


def extract_handcrafted(urls: pd.Series, cache_dir: Path, use_cache: bool, n_jobs: int) -> np.ndarray:
    url_hash = stable_hash(list(urls.values))
    cache_path = cache_dir / f"handcrafted_{len(urls)}_{url_hash}.pkl"
    if use_cache and cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)

    extractor = FeatureExtractor()
    LOGGER.info("Extracting handcrafted features for %d URLs", len(urls))
    rows = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(extractor.generate_features)(url) for url in urls
    )
    arr = np.asarray(rows, dtype=np.float32)
    if use_cache:
        ensure_dir(cache_dir)
        with cache_path.open("wb") as f:
            pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
    return arr


def build_features(prepared: PreparedData, args: argparse.Namespace, seed: int) -> FeatureBlocks:
    cache_dir = ensure_dir(Path(args.outdir) / "_cache")
    ngram_processor = NgramProcessor()
    random.seed(seed)
    np.random.seed(seed)

    LOGGER.info("Fitting and selecting stage-1 n-grams")
    ngram_processor.fit_vectorizer(prepared.train_urls)
    ngram_processor.select_features(
        prepared.train_urls,
        prepared.train_y,
        prepared.val_urls,
        prepared.val_y,
    )
    if args.max_ngram_features and len(ngram_processor.selected_ngram_indices) > args.max_ngram_features:
        LOGGER.info(
            "Capping selected n-grams from %d to %d for this run",
            len(ngram_processor.selected_ngram_indices),
            args.max_ngram_features,
        )
        ngram_processor.selected_ngram_indices = ngram_processor.selected_ngram_indices[: args.max_ngram_features]

    all_urls = pd.concat([prepared.train_urls, prepared.val_urls, prepared.test_urls], ignore_index=True)
    n_train = len(prepared.train_urls)
    n_val = len(prepared.val_urls)

    handcrafted = extract_handcrafted(all_urls, cache_dir, use_cache=not args.no_cache, n_jobs=args.n_jobs)

    LOGGER.info("Transforming selected n-grams for %d URLs", len(all_urls))
    ngram_matrix = ngram_processor.transform(all_urls)
    ngram_array = ngram_matrix.toarray().astype(np.float32) if hasattr(ngram_matrix, "toarray") else np.asarray(ngram_matrix, dtype=np.float32)

    x_all = np.hstack([handcrafted, ngram_array]).astype(np.float32, copy=False)
    feature_names = list(FeatureExtractor.FEATURE_NAMES) + list(ngram_processor.get_feature_names())

    x_train = x_all[:n_train]
    x_val = x_all[n_train : n_train + n_val]
    x_test = x_all[n_train + n_val :]
    return FeatureBlocks(x_train=x_train, x_val=x_val, x_test=x_test, feature_names=feature_names, ngram_processor=ngram_processor)


def normalise_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = float(np.max(scores)) if scores.size else 0.0
    if max_val <= 0:
        return np.zeros_like(scores, dtype=np.float64)
    return scores / max_val


def subsample_for_scoring(x: np.ndarray, y: np.ndarray, seed: int, sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if sample_size and x.shape[0] > sample_size:
        rng = np.random.RandomState(seed)
        idx = rng.choice(x.shape[0], sample_size, replace=False)
        return x[idx], y[idx]
    return x, y


def mi_scores(x: np.ndarray, y: np.ndarray, seed: int, sample_size: int) -> np.ndarray:
    x_sub, y_sub = subsample_for_scoring(x, y, seed, sample_size)
    LOGGER.info("Computing mutual information scores on shape %s", x_sub.shape)
    return mutual_info_classif(x_sub, y_sub, random_state=seed, discrete_features=False)


def gain_scores(x: np.ndarray, y: np.ndarray, seed: int, n_estimators: int = 200) -> np.ndarray:
    LOGGER.info("Computing LightGBM gain scores on shape %s", x.shape)
    model = LGBMClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        importance_type="gain",
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x, y)
    return np.asarray(model.feature_importances_, dtype=np.float64)


def stability_frequency(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    stability_runs: int,
    sample_fraction: float,
    regularization_strength: float,
    n_jobs: int,
) -> np.ndarray:
    LOGGER.info("Computing stability frequencies: runs=%d", stability_runs)
    rng = np.random.RandomState(seed)
    seeds = [int(rng.randint(0, 1_000_000)) for _ in range(stability_runs)]

    def single_run(local_seed: int) -> np.ndarray:
        local_rng = np.random.RandomState(local_seed)
        size = max(2, int(sample_fraction * x.shape[0]))
        idx = local_rng.choice(x.shape[0], size=size, replace=False)
        if len(np.unique(y[idx])) < 2:
            return np.zeros(x.shape[1], dtype=np.int8)
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=regularization_strength,
            random_state=local_seed,
            max_iter=200,
        )
        model.fit(x[idx], y[idx])
        return (np.abs(model.coef_).ravel() > 1e-5).astype(np.int8)

    selected = Parallel(n_jobs=n_jobs)(delayed(single_run)(s) for s in seeds)
    return np.mean(np.vstack(selected), axis=0)


def correlation_prune(x: np.ndarray, ranked_indices: np.ndarray, max_features: int, threshold: float) -> np.ndarray:
    if len(ranked_indices) == 0:
        return ranked_indices
    limited = ranked_indices[: min(len(ranked_indices), 5000)]
    if len(limited) <= max_features:
        return limited
    candidate_data = x[:, limited]
    LOGGER.info("Computing correlation matrix for %d candidates", len(limited))
    corr = np.abs(np.corrcoef(candidate_data, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    redundant = np.zeros(len(limited), dtype=bool)
    final: List[int] = []
    for i, original_idx in enumerate(limited):
        if redundant[i]:
            continue
        final.append(int(original_idx))
        if len(final) >= max_features:
            break
        mask = corr[i] > threshold
        mask[: i + 1] = False
        redundant |= mask
    return np.asarray(final, dtype=int)


def select_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    selector: str,
    seed: int,
    budget: int,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    y = np.asarray(y_train).astype(int)
    n_features = x_train.shape[1]
    budget = min(max(1, budget), n_features)
    meta: Dict[str, Any] = {"selector": selector}

    if selector in {"none", "no_sdcs", "ngram_preselection_only"}:
        idx = np.arange(n_features, dtype=int)
        meta["ranked_indices"] = idx.tolist()
        return idx, meta

    if selector == "handcrafted_only":
        idx = np.arange(min(HANDCRAFTED_COUNT, n_features), dtype=int)
        meta["ranked_indices"] = idx.tolist()
        return idx, meta

    if selector == "ngram_only":
        pool = np.arange(HANDCRAFTED_COUNT, n_features, dtype=int)
        if len(pool) <= budget:
            meta["ranked_indices"] = pool.tolist()
            return pool, meta
        scores = mi_scores(x_train[:, pool], y, seed, args.score_sample_size)
        ranked = pool[np.argsort(scores)[::-1]]
        meta["ranked_indices"] = ranked.tolist()
        meta["mi_scores"] = {feature_names[int(i)]: float(scores[j]) for j, i in enumerate(pool)}
        return ranked[:budget], meta

    mi = gain = freq = None
    if selector in {"mi_only", "mi_gain_no_stability", "sdcs", "sdcs_no_redundancy"}:
        mi = mi_scores(x_train, y, seed, args.score_sample_size)
        meta["mi_max"] = float(np.max(mi)) if len(mi) else 0.0
    if selector in {"gain_only", "mi_gain_no_stability", "sdcs", "sdcs_no_redundancy"}:
        gain = gain_scores(x_train, y, seed, args.selector_estimators)
        meta["gain_max"] = float(np.max(gain)) if len(gain) else 0.0
    if selector in {"stability_only", "sdcs", "sdcs_no_redundancy"}:
        freq = stability_frequency(
            x_train,
            y,
            seed,
            args.stability_runs,
            args.sample_fraction,
            args.regularization_strength,
            args.n_jobs,
        )
        meta["stability_frequency"] = {feature_names[i]: float(freq[i]) for i in range(n_features)}

    if selector == "mi_only":
        ranked = np.argsort(mi)[::-1]
        meta["ranked_indices"] = ranked.tolist()
        return ranked[:budget].astype(int), meta

    if selector == "gain_only":
        ranked = np.argsort(gain)[::-1]
        meta["ranked_indices"] = ranked.tolist()
        return ranked[:budget].astype(int), meta

    if selector == "mi_gain_no_stability":
        hybrid = args.mi_weight * normalise_scores(mi) + (1.0 - args.mi_weight) * normalise_scores(gain)
        ranked = np.argsort(hybrid)[::-1]
        meta["ranked_indices"] = ranked.tolist()
        return ranked[:budget].astype(int), meta

    if selector == "stability_only":
        candidates = np.where(freq >= args.frequency_threshold)[0]
        ranked = candidates[np.argsort(freq[candidates])[::-1]] if len(candidates) else np.argsort(freq)[::-1]
        meta["ranked_indices"] = ranked.tolist()
        return ranked[:budget].astype(int), meta

    if selector in {"sdcs", "sdcs_no_redundancy"}:
        hybrid = args.mi_weight * normalise_scores(mi) + (1.0 - args.mi_weight) * normalise_scores(gain)
        merged = args.alpha_sdcs * freq + (1.0 - args.alpha_sdcs) * hybrid
        candidates = np.where(freq >= args.frequency_threshold)[0]
        if len(candidates) == 0:
            candidates = np.arange(n_features)
        ranked = candidates[np.argsort(merged[candidates])[::-1]]
        meta["ranked_indices"] = ranked.tolist()
        meta["merged_score_top"] = [
            {"feature": feature_names[int(i)], "score": float(merged[int(i)]), "freq": float(freq[int(i)])}
            for i in ranked[: min(50, len(ranked))]
        ]
        if selector == "sdcs_no_redundancy":
            return ranked[:budget].astype(int), meta
        pruned = correlation_prune(x_train, ranked, budget, args.correlation_threshold)
        return pruned.astype(int), meta

    raise ValueError(f"Unknown selector: {selector}")


def make_lgbm(seed: int, learner: str, y_train: np.ndarray, alpha: float, gamma: float) -> Any:
    base_params = dict(
        boosting_type="gbdt",
        num_leaves=50,
        max_depth=-1,
        learning_rate=0.15,
        n_estimators=500,
        min_child_samples=40,
        reg_alpha=0.1,
        reg_lambda=0.3,
        verbosity=-1,
        random_state=seed,
        device="cpu",
    )
    if learner == "focal":
        return FocalLossLGBM(alpha=alpha, gamma=gamma, **base_params)
    if learner == "weighted":
        return LGBMClassifier(class_weight="balanced", **base_params)
    if learner == "scale_pos_weight":
        positives = max(1, int(np.sum(y_train == 1)))
        negatives = max(1, int(np.sum(y_train == 0)))
        return LGBMClassifier(scale_pos_weight=negatives / positives, **base_params)
    if learner in {"std", "threshold_tuned"}:
        return LGBMClassifier(**base_params)
    raise ValueError(f"Unknown learner: {learner}")


def predict_probabilities(model: Any, x: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x)
    if probs.ndim == 1:
        return probs
    return probs[:, 1]


def tune_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        preds = (probs >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def ece_score(y_true: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1.0:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)
        if not np.any(mask):
            continue
        ece += np.mean(mask) * abs(float(np.mean(probs[mask])) - float(np.mean(y_true[mask])))
    return float(ece)


def safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan")


def safe_average_precision(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan")


def recall_at_precision(y_true: np.ndarray, probs: np.ndarray, precision_target: float) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    precisions, recalls, _ = precision_recall_curve(y_true, probs)
    valid = recalls[precisions >= precision_target]
    return float(np.max(valid)) if len(valid) else 0.0


def hard_negative_fpr(urls: Sequence[str], y_true: np.ndarray, preds: np.ndarray) -> Tuple[float, int]:
    hard = []
    for url, label in zip(urls, y_true):
        if int(label) != 0:
            hard.append(False)
            continue
        text = str(url)
        hard.append(len(text) > 100 or text.count("/") > 5)
    hard_mask = np.asarray(hard, dtype=bool)
    count = int(np.sum(hard_mask))
    if count == 0:
        return float("nan"), 0
    return float(np.mean(preds[hard_mask] == 1)), count


def compute_metrics(y_true: Sequence[int], probs: Sequence[float], threshold: float = 0.5, urls: Optional[Sequence[str]] = None) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probs, dtype=float)
    preds = (p >= threshold).astype(int)
    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "auc": safe_auc(y, p),
        "pr_auc": safe_average_precision(y, p),
        "mcc": float(matthews_corrcoef(y, preds)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece_score(y, p),
        "recall_at_precision_90": recall_at_precision(y, p, 0.90),
        "recall_at_precision_95": recall_at_precision(y, p, 0.95),
    }
    if urls is not None:
        h_fpr, h_count = hard_negative_fpr(urls, y, preds)
        result["hard_negative_fpr"] = h_fpr
        result["hard_negative_count"] = h_count
    return result


def fit_evaluate(
    prepared: PreparedData,
    features: FeatureBlocks,
    selector: str,
    learner: str,
    seed: int,
    args: argparse.Namespace,
    budget: Optional[int] = None,
    alpha: Optional[float] = None,
    gamma: Optional[float] = None,
) -> Tuple[Dict[str, Any], InferenceBundle, Dict[str, Any]]:
    budget = budget or args.feature_budget
    alpha = args.focal_alpha if alpha is None else alpha
    gamma = args.focal_gamma if gamma is None else gamma

    selected, meta = select_features(
        features.x_train,
        prepared.train_y.values,
        features.feature_names,
        selector,
        seed,
        budget,
        args,
    )
    selected_names = [features.feature_names[int(i)] for i in selected]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(features.x_train[:, selected].astype(np.float32))
    x_val = scaler.transform(features.x_val[:, selected].astype(np.float32))
    x_test = scaler.transform(features.x_test[:, selected].astype(np.float32))

    y_train = prepared.train_y.values.astype(int)
    y_val = prepared.val_y.values.astype(int)
    y_test = prepared.test_y.values.astype(int)

    model = make_lgbm(seed, learner, y_train, alpha=alpha, gamma=gamma)
    model.fit(x_train, y_train)

    threshold = 0.5
    if learner == "threshold_tuned":
        val_probs = predict_probabilities(model, x_val)
        threshold = tune_threshold(y_val, val_probs)

    test_probs = predict_probabilities(model, x_test)
    metrics = compute_metrics(y_test, test_probs, threshold, prepared.test_urls)
    metrics.update(
        {
            "seed": seed,
            "selector": selector,
            "learner": learner,
            "feature_count": int(len(selected)),
            "feature_budget": int(budget),
            "alpha": float(alpha),
            "gamma": float(gamma),
            "mode": prepared.mode,
            "train_path": prepared.train_path,
            "test_path": prepared.test_path,
            "train_n": int(len(prepared.train_y)),
            "val_n": int(len(prepared.val_y)),
            "test_n": int(len(prepared.test_y)),
            "test_positive_rate": float(np.mean(y_test)),
        }
    )
    bundle = InferenceBundle(
        model=model,
        scaler=scaler,
        ngram_processor=features.ngram_processor,
        selected_indices=selected,
        feature_names=features.feature_names,
        selected_names=selected_names,
        threshold=threshold,
    )
    return metrics, bundle, meta


def summarise_results(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "pr_auc",
        "mcc",
        "brier",
        "ece",
        "recall_at_precision_90",
        "recall_at_precision_95",
        "hard_negative_fpr",
        "feature_count",
    ]
    available = [c for c in metric_cols if c in df.columns]
    rows = []
    for keys, group in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(len(group))
        for col in available:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            if len(values) > 1:
                row[f"{col}_ci95"] = float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))
            else:
                row[f"{col}_ci95"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, outdir: Path, name: str) -> Path:
    ensure_dir(outdir)
    path = outdir / name
    df.to_csv(path, index=False)
    LOGGER.info("Wrote %s", path)
    return path


def write_json(data: Any, outdir: Path, name: str) -> Path:
    ensure_dir(outdir)
    path = outdir / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, sort_keys=True)
    LOGGER.info("Wrote %s", path)
    return path


def to_jsonable(value: Any) -> Any:
    """Convert NumPy/pandas/path objects into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(to_jsonable(k)): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) and not isinstance(value, (str, bytes)):
        return None
    return value


def pairwise_tests(df: pd.DataFrame, outdir: Path, name: str, baseline_selector: str = "sdcs", metric: str = "f1") -> None:
    try:
        from scipy.stats import wilcoxon
    except Exception as exc:  # pragma: no cover - scipy should be available, but keep runnable.
        LOGGER.warning("Skipping pairwise tests because scipy is unavailable: %s", exc)
        return

    if "seed" not in df.columns or metric not in df.columns:
        return
    rows = []
    grouping = [c for c in ["learner", "feature_budget", "mode", "test_path"] if c in df.columns]
    for keys, group in df.groupby(grouping, dropna=False) if grouping else [((), df)]:
        base = group[group["selector"] == baseline_selector][["seed", metric]].rename(columns={metric: "baseline"})
        if base.empty:
            continue
        for selector in sorted(set(group["selector"]) - {baseline_selector}):
            other = group[group["selector"] == selector][["seed", metric]].rename(columns={metric: "other"})
            merged = base.merge(other, on="seed")
            if len(merged) < 2:
                continue
            stat, p_value = wilcoxon(merged["baseline"], merged["other"], zero_method="wilcox", alternative="two-sided")
            row = {
                "selector": selector,
                "baseline_selector": baseline_selector,
                "metric": metric,
                "n_pairs": int(len(merged)),
                "baseline_mean": float(merged["baseline"].mean()),
                "other_mean": float(merged["other"].mean()),
                "delta_baseline_minus_other": float(merged["baseline"].mean() - merged["other"].mean()),
                "wilcoxon_stat": float(stat),
                "p_value": float(p_value),
            }
            if grouping:
                key_tuple = keys if isinstance(keys, tuple) else (keys,)
                row.update(dict(zip(grouping, key_tuple)))
            rows.append(row)
    if rows:
        write_csv(pd.DataFrame(rows), outdir, name)


def run_sdcs_ablation(args: argparse.Namespace) -> None:
    outdir = ensure_dir(Path(args.outdir) / "sdcs_ablation")
    selectors = [
        "mi_only",
        "gain_only",
        "mi_gain_no_stability",
        "stability_only",
        "sdcs_no_redundancy",
        "ngram_preselection_only",
        "sdcs",
    ]
    rows: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    for seed in args.seeds:
        prepared = prepare_data(args, seed)
        features = build_features(prepared, args, seed)
        for selector in selectors:
            LOGGER.info("Running SDCS ablation selector=%s seed=%d", selector, seed)
            metrics, _, meta = fit_evaluate(prepared, features, selector, "focal", seed, args)
            rows.append(metrics)
            metadata[f"{seed}_{selector}"] = {
                "selected_top": [features.feature_names[i] for i in meta.get("ranked_indices", [])[:30]],
                "meta": {k: v for k, v in meta.items() if k not in {"stability_frequency"}},
            }
    df = pd.DataFrame(rows)
    write_csv(df, outdir, "per_run.csv")
    write_csv(summarise_results(df, ["selector", "learner", "mode", "test_path"]), outdir, "summary.csv")
    pairwise_tests(df, outdir, "pairwise_wilcoxon_vs_sdcs.csv")
    write_json(metadata, outdir, "selection_metadata.json")


def run_sdcs_loss_factorial(args: argparse.Namespace) -> None:
    outdir = ensure_dir(Path(args.outdir) / "sdcs_loss_factorial")
    selectors = ["no_sdcs", "mi_only", "gain_only", "sdcs"]
    learners = ["std", "weighted", "focal"]
    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        prepared = prepare_data(args, seed)
        features = build_features(prepared, args, seed)
        for selector, learner in itertools.product(selectors, learners):
            LOGGER.info("Running factorial selector=%s learner=%s seed=%d", selector, learner, seed)
            metrics, _, _ = fit_evaluate(prepared, features, selector, learner, seed, args)
            rows.append(metrics)
    df = pd.DataFrame(rows)
    write_csv(df, outdir, "per_run.csv")
    write_csv(summarise_results(df, ["selector", "learner", "mode", "test_path"]), outdir, "summary.csv")
    pairwise_tests(df, outdir, "pairwise_wilcoxon_vs_sdcs.csv")


def downsample_to_ratio(
    x: np.ndarray,
    y: pd.Series,
    urls: pd.Series,
    pos_to_neg_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, pd.Series, pd.Series]:
    rng = np.random.RandomState(seed)
    y_arr = y.values.astype(int)
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return x, y, urls
    keep_pos = min(len(pos_idx), max(1, int(round(len(neg_idx) * pos_to_neg_ratio))))
    pos_sample = rng.choice(pos_idx, keep_pos, replace=False)
    keep = np.concatenate([neg_idx, pos_sample])
    rng.shuffle(keep)
    return x[keep], y.iloc[keep].reset_index(drop=True), urls.iloc[keep].reset_index(drop=True)


def run_focal_imbalance(args: argparse.Namespace) -> None:
    outdir = ensure_dir(Path(args.outdir) / "focal_imbalance")
    learners = ["std", "weighted", "scale_pos_weight", "threshold_tuned", "focal"]
    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        prepared = prepare_data(args, seed)
        features = build_features(prepared, args, seed)

        for scenario in ["natural", "severe_1_to_10"]:
            scenario_prepared = prepared
            scenario_features = features
            if scenario == "severe_1_to_10":
                x_train, y_train, train_urls = downsample_to_ratio(
                    features.x_train, prepared.train_y, prepared.train_urls, 0.10, seed
                )
                x_test, y_test, test_urls = downsample_to_ratio(
                    features.x_test, prepared.test_y, prepared.test_urls, 0.10, seed + 1
                )
                scenario_prepared = PreparedData(
                    train_urls=train_urls,
                    val_urls=prepared.val_urls,
                    test_urls=test_urls,
                    train_y=y_train,
                    val_y=prepared.val_y,
                    test_y=y_test,
                    train_indices=prepared.train_indices,
                    val_indices=prepared.val_indices,
                    test_indices=prepared.test_indices,
                    mode=f"{prepared.mode}_{scenario}",
                    train_path=prepared.train_path,
                    test_path=prepared.test_path,
                )
                scenario_features = FeatureBlocks(
                    x_train=x_train,
                    x_val=features.x_val,
                    x_test=x_test,
                    feature_names=features.feature_names,
                    ngram_processor=features.ngram_processor,
                )

            for learner in learners:
                LOGGER.info("Running imbalance scenario=%s learner=%s seed=%d", scenario, learner, seed)
                metrics, _, _ = fit_evaluate(scenario_prepared, scenario_features, "sdcs", learner, seed, args)
                metrics["imbalance_scenario"] = scenario
                rows.append(metrics)
    df = pd.DataFrame(rows)
    write_csv(df, outdir, "per_run.csv")
    write_csv(summarise_results(df, ["imbalance_scenario", "learner", "selector", "mode"]), outdir, "summary.csv")


def run_gamma_sensitivity(args: argparse.Namespace) -> None:
    outdir = ensure_dir(Path(args.outdir) / "gamma_sensitivity")
    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        prepared = prepare_data(args, seed)
        features = build_features(prepared, args, seed)
        selected, meta = select_features(
            features.x_train,
            prepared.train_y.values,
            features.feature_names,
            "sdcs",
            seed,
            args.feature_budget,
            args,
        )
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features.x_train[:, selected].astype(np.float32))
        x_test = scaler.transform(features.x_test[:, selected].astype(np.float32))
        y_train = prepared.train_y.values.astype(int)
        y_test = prepared.test_y.values.astype(int)

        for alpha, gamma in itertools.product(args.alphas, args.gammas):
            LOGGER.info("Gamma sensitivity seed=%d alpha=%.3f gamma=%.3f", seed, alpha, gamma)
            model = make_lgbm(seed, "focal", y_train, alpha=alpha, gamma=gamma)
            model.fit(x_train, y_train)
            probs = predict_probabilities(model, x_test)
            metrics = compute_metrics(y_test, probs, 0.5, prepared.test_urls)
            metrics.update(
                {
                    "seed": seed,
                    "selector": "sdcs_fixed",
                    "learner": "focal",
                    "alpha": float(alpha),
                    "gamma": float(gamma),
                    "feature_count": int(len(selected)),
                    "mode": prepared.mode,
                    "train_path": prepared.train_path,
                    "test_path": prepared.test_path,
                }
            )
            rows.append(metrics)

        write_json(
            {"top_ranked_features": [features.feature_names[i] for i in meta.get("ranked_indices", [])[:50]]},
            outdir,
            f"selection_seed_{seed}.json",
        )

    df = pd.DataFrame(rows)
    write_csv(df, outdir, "per_run.csv")
    write_csv(summarise_results(df, ["alpha", "gamma", "mode", "test_path"]), outdir, "summary.csv")

    best_f1 = df["f1"].max()
    near = df[df["f1"] >= best_f1 - 0.005].copy()
    write_csv(near, outdir, "near_optimal_within_0p005_f1.csv")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        for metric in ["f1", "ece"]:
            pivot = df.groupby(["alpha", "gamma"])[metric].mean().reset_index().pivot(index="alpha", columns="gamma", values=metric)
            plt.figure(figsize=(10, 5))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
            plt.title(f"Focal-loss {metric.upper()} sensitivity")
            plt.tight_layout()
            path = outdir / f"{metric}_heatmap.png"
            plt.savefig(path, dpi=220)
            plt.close()
            LOGGER.info("Wrote %s", path)
    except Exception as exc:
        LOGGER.warning("Skipping heatmaps: %s", exc)


def bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int = 1000) -> Tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.RandomState(seed)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_bootstrap)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_repeated_runs(args: argparse.Namespace) -> None:
    outdir = ensure_dir(Path(args.outdir) / "repeated_runs")
    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        prepared = prepare_data(args, seed)
        features = build_features(prepared, args, seed)
        LOGGER.info("Running repeated full LitePhish seed=%d", seed)
        metrics, _, _ = fit_evaluate(prepared, features, "sdcs", "focal", seed, args)
        rows.append(metrics)
    df = pd.DataFrame(rows)
    write_csv(df, outdir, "per_run.csv")
    summary = summarise_results(df, ["selector", "learner", "mode", "test_path"])
    for metric in ["f1", "auc", "ece", "recall"]:
        if metric in df:
            lo, hi = bootstrap_ci(df[metric].to_numpy(dtype=float), args.seeds[0], args.bootstrap)
            summary[f"{metric}_bootstrap_ci_low"] = lo
            summary[f"{metric}_bootstrap_ci_high"] = hi
    write_csv(summary, outdir, "summary.csv")


def feature_category(name: str) -> str:
    if name in FeatureExtractor.FEATURE_NAMES:
        if "Entropy" in name:
            return "handcrafted_entropy"
        if any(token in name for token in ["Obfuscation", "Encoding", "Redirect", "Homoglyph", "Special"]):
            return "handcrafted_obfuscation"
        return "handcrafted_other"
    return "ngram"


def shap_ranks_for_bundle(bundle: InferenceBundle, x_selected: np.ndarray, sample_size: int, seed: int) -> Dict[str, int]:
    try:
        import shap
    except Exception as exc:
        LOGGER.warning("Skipping SHAP ranks because shap is unavailable: %s", exc)
        return {}
    rng = np.random.RandomState(seed)
    if x_selected.shape[0] > sample_size:
        idx = rng.choice(x_selected.shape[0], sample_size, replace=False)
        x_eval = x_selected[idx]
    else:
        x_eval = x_selected
    explainer = shap.TreeExplainer(bundle.model)
    values = explainer.shap_values(x_eval)
    if isinstance(values, list):
        values = values[-1]
    scores = np.mean(np.abs(values), axis=0)
    order = np.argsort(scores)[::-1]
    return {bundle.selected_names[int(i)]: int(rank + 1) for rank, i in enumerate(order)}


def run_feature_inventory(args: argparse.Namespace) -> None:
    outdir = ensure_dir(Path(args.outdir) / "feature_inventory")
    rows: Dict[str, Dict[str, Any]] = {}
    shap_rank_map: Dict[str, int] = {}

    if args.artifacts:
        artifacts = Path(args.artifacts)
        with (artifacts / "selected_features.pkl").open("rb") as f:
            selected = np.asarray(pickle.load(f), dtype=int)
        with (artifacts / "all_feature_names.pkl").open("rb") as f:
            names = list(pickle.load(f))
        if args.with_shap and (artifacts / "model.pkl").exists() and (artifacts / "test_features_scaled.pkl").exists():
            bundle = load_bundle_from_artifacts(artifacts)
            with (artifacts / "test_features_scaled.pkl").open("rb") as f:
                x_scaled = np.asarray(pickle.load(f))
            shap_rank_map = shap_ranks_for_bundle(bundle, x_scaled, min(args.score_sample_size, len(x_scaled)), args.seeds[0])
        selected_set = set(map(int, selected))
        for i, name in enumerate(names):
            rows[name] = {
                "feature": name,
                "feature_index": i,
                "category": feature_category(name),
                "is_novel_claimed_feature": name in NOVEL_FEATURES,
                "retained_in_artifacts": i in selected_set,
                "selection_frequency": float(i in selected_set),
                "mean_rank": float(np.where(selected == i)[0][0] + 1) if i in selected_set else float("nan"),
                "shap_rank": shap_rank_map.get(name, float("nan")),
            }

    per_seed_selected: List[set] = []
    per_seed_ranks: Dict[str, List[int]] = {}
    if not args.artifacts or args.run_inventory_selection:
        for seed in args.seeds:
            prepared = prepare_data(args, seed)
            features = build_features(prepared, args, seed)
            if args.with_shap and seed == args.seeds[0]:
                _, bundle, meta = fit_evaluate(prepared, features, "sdcs", "focal", seed, args)
                selected = bundle.selected_indices
                x_selected = bundle.scaler.transform(features.x_test[:, selected].astype(np.float32))
                shap_rank_map.update(
                    shap_ranks_for_bundle(bundle, x_selected, min(args.score_sample_size, len(x_selected)), seed)
                )
            else:
                selected, meta = select_features(
                    features.x_train,
                    prepared.train_y.values,
                    features.feature_names,
                    "sdcs",
                    seed,
                    args.feature_budget,
                    args,
                )
            selected_set = set(map(int, selected))
            per_seed_selected.append(selected_set)
            ranked = meta.get("ranked_indices", [])
            rank_map = {int(idx): rank + 1 for rank, idx in enumerate(ranked)}
            for i, name in enumerate(features.feature_names):
                if name not in rows:
                    rows[name] = {
                        "feature": name,
                        "feature_index": i,
                        "category": feature_category(name),
                        "is_novel_claimed_feature": name in NOVEL_FEATURES,
                        "retained_in_artifacts": False,
                    }
                rows[name].setdefault("selection_count", 0)
                rows[name]["selection_count"] += int(i in selected_set)
                if i in rank_map:
                    per_seed_ranks.setdefault(name, []).append(rank_map[i])

        for row in rows.values():
            if per_seed_selected:
                row["selection_frequency"] = row.get("selection_count", 0) / len(per_seed_selected)
            ranks = per_seed_ranks.get(row["feature"], [])
            row["mean_rank"] = float(np.mean(ranks)) if ranks else row.get("mean_rank", float("nan"))
            row["shap_rank"] = shap_rank_map.get(row["feature"], row.get("shap_rank", float("nan")))

    df = pd.DataFrame(rows.values())
    if not df.empty:
        if "selection_frequency" not in df.columns:
            df["selection_frequency"] = 0.0
        df["retained_final"] = df["selection_frequency"].fillna(0) > 0
        df = df.sort_values(["is_novel_claimed_feature", "selection_frequency", "mean_rank"], ascending=[False, False, True])
    write_csv(df, outdir, "feature_inventory.csv")
    if not df.empty:
        write_csv(df[df["is_novel_claimed_feature"] == True], outdir, "novel_feature_survival.csv")  # noqa: E712


def run_compression_curve(args: argparse.Namespace) -> None:
    outdir = ensure_dir(Path(args.outdir) / "compression_curve")
    rows: List[Dict[str, Any]] = []
    selector_for_representation = {
        "handcrafted_only": "handcrafted_only",
        "ngram_only": "ngram_only",
        "combined_no_sdcs": "mi_gain_no_stability",
        "combined_sdcs": "sdcs",
    }
    for seed in args.seeds:
        prepared = prepare_data(args, seed)
        features = build_features(prepared, args, seed)
        for budget in args.budgets:
            for representation, selector in selector_for_representation.items():
                if representation == "handcrafted_only" and budget != args.budgets[0]:
                    continue
                LOGGER.info("Compression representation=%s budget=%d seed=%d", representation, budget, seed)
                metrics, _, _ = fit_evaluate(prepared, features, selector, "focal", seed, args, budget=budget)
                metrics["representation"] = representation
                rows.append(metrics)
    df = pd.DataFrame(rows)
    write_csv(df, outdir, "per_run.csv")
    write_csv(summarise_results(df, ["representation", "feature_budget", "mode", "test_path"]), outdir, "summary.csv")


def mean_pairwise_jaccard(sets: Sequence[set]) -> float:
    if len(sets) < 2:
        return float("nan")
    scores = []
    for a, b in itertools.combinations(sets, 2):
        union = len(a | b)
        scores.append(len(a & b) / union if union else 1.0)
    return float(np.mean(scores))


def run_representation_stability(args: argparse.Namespace) -> None:
    outdir = ensure_dir(Path(args.outdir) / "representation_stability")
    representations = {
        "handcrafted_only": "handcrafted_only",
        "ngram_only": "ngram_only",
        "combined_no_sdcs": "mi_gain_no_stability",
        "combined_sdcs": "sdcs",
    }
    rows: List[Dict[str, Any]] = []
    selected_sets: Dict[str, List[set]] = {k: [] for k in representations}
    top_importance_sets: Dict[str, List[set]] = {k: [] for k in representations}
    for seed in args.seeds:
        prepared = prepare_data(args, seed)
        features = build_features(prepared, args, seed)
        for representation, selector in representations.items():
            LOGGER.info("Representation stability representation=%s seed=%d", representation, seed)
            metrics, bundle, _ = fit_evaluate(prepared, features, selector, "focal", seed, args)
            metrics["representation"] = representation
            rows.append(metrics)
            selected_sets[representation].append(set(bundle.selected_names))
            try:
                importances = np.asarray(bundle.model.feature_importances_, dtype=float)
                top_idx = np.argsort(importances)[::-1][:20]
                top_importance_sets[representation].append({bundle.selected_names[int(i)] for i in top_idx})
            except Exception:
                pass
    df = pd.DataFrame(rows)
    stability_rows = []
    for representation in representations:
        stability_rows.append(
            {
                "representation": representation,
                "selected_feature_jaccard": mean_pairwise_jaccard(selected_sets[representation]),
                "top20_importance_jaccard": mean_pairwise_jaccard(top_importance_sets[representation]),
            }
        )
    write_csv(df, outdir, "per_run.csv")
    write_csv(summarise_results(df, ["representation", "mode", "test_path"]), outdir, "summary.csv")
    write_csv(pd.DataFrame(stability_rows), outdir, "stability_indices.csv")


def load_bundle_from_artifacts(artifacts_dir: Path) -> InferenceBundle:
    with (artifacts_dir / "model.pkl").open("rb") as f:
        model = pickle.load(f)
    with (artifacts_dir / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)
    with (artifacts_dir / "ngram_processor.pkl").open("rb") as f:
        ngram_processor = pickle.load(f)
    with (artifacts_dir / "selected_features.pkl").open("rb") as f:
        selected = np.asarray(pickle.load(f), dtype=int)
    with (artifacts_dir / "all_feature_names.pkl").open("rb") as f:
        names = list(pickle.load(f))
    selected_names = [names[int(i)] for i in selected]
    return InferenceBundle(model, scaler, ngram_processor, selected, names, selected_names)


def train_default_bundle(args: argparse.Namespace, seed: int) -> Tuple[InferenceBundle, PreparedData, FeatureBlocks]:
    prepared = prepare_data(args, seed)
    features = build_features(prepared, args, seed)
    _, bundle, _ = fit_evaluate(prepared, features, "sdcs", "focal", seed, args)
    return bundle, prepared, features


def extract_with_bundle(urls: pd.Series, bundle: InferenceBundle, args: argparse.Namespace) -> np.ndarray:
    cache_dir = ensure_dir(Path(args.outdir) / "_cache")
    handcrafted = extract_handcrafted(urls.reset_index(drop=True), cache_dir, use_cache=not args.no_cache, n_jobs=args.n_jobs)
    ngram_matrix = bundle.ngram_processor.transform(urls.reset_index(drop=True))
    ngram_array = ngram_matrix.toarray().astype(np.float32) if hasattr(ngram_matrix, "toarray") else np.asarray(ngram_matrix, dtype=np.float32)
    return np.hstack([handcrafted, ngram_array]).astype(np.float32, copy=False)


def predict_urls(bundle: InferenceBundle, urls: pd.Series, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    full = extract_with_bundle(urls, bundle, args)
    selected = full[:, bundle.selected_indices]
    scaled = bundle.scaler.transform(selected.astype(np.float32))
    probs = predict_probabilities(bundle.model, scaled)
    preds = (probs >= bundle.threshold).astype(int)
    return preds, probs


def measure_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    train_norm = train_df["url"].map(normalize_url)
    test_norm = test_df["url"].map(normalize_url)
    train_root = train_df["url"].map(root_domain)
    test_root = test_df["url"].map(root_domain)

    train_norm_set = set(train_norm)
    train_root_set = set(train_root)
    overlap_mask = test_norm.isin(train_norm_set)
    root_overlap_mask = test_root.isin(train_root_set)
    overlap_rows = test_df[overlap_mask | root_overlap_mask].copy()
    overlap_rows["normalized_url"] = test_norm[overlap_mask | root_overlap_mask].values
    overlap_rows["root_domain"] = test_root[overlap_mask | root_overlap_mask].values
    overlap_rows["exact_url_overlap"] = overlap_mask[overlap_mask | root_overlap_mask].values
    overlap_rows["root_domain_overlap"] = root_overlap_mask[overlap_mask | root_overlap_mask].values

    summary = {
        "train_n": int(len(train_df)),
        "test_n": int(len(test_df)),
        "train_positive_rate": float(train_df["phishing"].mean()),
        "test_positive_rate": float(test_df["phishing"].mean()),
        "exact_normalized_url_overlap_count": int(overlap_mask.sum()),
        "exact_normalized_url_overlap_rate": float(overlap_mask.mean()),
        "root_domain_overlap_count": int(root_overlap_mask.sum()),
        "root_domain_overlap_rate": float(root_overlap_mask.mean()),
        "train_unique_root_domains": int(train_root.nunique()),
        "test_unique_root_domains": int(test_root.nunique()),
        "test_label_counts": {str(k): int(v) for k, v in test_df["phishing"].value_counts().to_dict().items()},
        "source_column_present": "source" in test_df.columns,
        "top_test_tlds": dict(test_df["url"].map(lambda u: extract_tld_parts(str(u)).suffix.lower()).value_counts().head(20)),
    }
    if "source" in test_df.columns:
        summary["test_source_counts"] = {str(k): int(v) for k, v in test_df["source"].value_counts().to_dict().items()}
    return summary, overlap_rows


def evaluate_external_dataframe(bundle: InferenceBundle, df: pd.DataFrame, args: argparse.Namespace, label: str) -> Dict[str, Any]:
    preds, probs = predict_urls(bundle, df["url"], args)
    metrics = compute_metrics(df["phishing"].values, probs, bundle.threshold, df["url"])
    metrics.update({"evaluation_set": label, "n": int(len(df)), "positive_rate": float(df["phishing"].mean())})
    return metrics


def analyze_external_cohort(args: argparse.Namespace) -> None:
    outdir = ensure_dir(Path(args.outdir) / "external_cohort")
    train_path = Path(args.train_dataset or args.dataset)
    live_path = Path(args.test_dataset or args.live_dataset or "dataset/PhishTank.csv")
    train_df = read_csv_dataset(train_path, sample=args.sample, seed=args.seeds[0])
    live_df = read_csv_dataset(live_path, sample=args.test_sample or args.sample, seed=args.seeds[0])
    summary, overlap_rows = measure_overlap(train_df, live_df)
    write_json(summary, outdir, "overlap_summary.json")
    write_csv(overlap_rows, outdir, "overlap_rows.csv")

    if args.artifacts:
        bundle = load_bundle_from_artifacts(Path(args.artifacts))
    else:
        bundle, _, _ = train_default_bundle(args, args.seeds[0])

    eval_rows = [evaluate_external_dataframe(bundle, live_df, args, "live_original")]
    if args.alternate_benign:
        alt_df = read_csv_dataset(Path(args.alternate_benign), sample=None, seed=args.seeds[0])
        live_phish = live_df[live_df["phishing"] == 1]
        alt_benign = alt_df[alt_df["phishing"] == 0]
        if len(alt_benign) > len(live_df[live_df["phishing"] == 0]):
            alt_benign = alt_benign.sample(n=len(live_df[live_df["phishing"] == 0]), random_state=args.seeds[0])
        shifted = pd.concat([live_phish, alt_benign], ignore_index=True).sample(frac=1.0, random_state=args.seeds[0])
        shifted_summary, shifted_overlap = measure_overlap(train_df, shifted)
        write_json(shifted_summary, outdir, "alternate_benign_overlap_summary.json")
        write_csv(shifted_overlap, outdir, "alternate_benign_overlap_rows.csv")
        eval_rows.append(evaluate_external_dataframe(bundle, shifted, args, "live_phishing_plus_alternate_benign"))
    write_csv(pd.DataFrame(eval_rows), outdir, "live_evaluation.csv")


def measure_latency(args: argparse.Namespace) -> None:
    outdir = ensure_dir(Path(args.outdir) / "latency")
    if args.artifacts:
        bundle = load_bundle_from_artifacts(Path(args.artifacts))
    else:
        bundle, _, _ = train_default_bundle(args, args.seeds[0])

    test_path = Path(args.test_dataset or args.live_dataset or args.dataset)
    df = read_csv_dataset(test_path, sample=args.test_sample or args.sample, seed=args.seeds[0])
    rows = []
    extractor = FeatureExtractor()
    rng = np.random.RandomState(args.seeds[0])
    for batch_size in args.batch_sizes:
        n = min(batch_size, len(df))
        urls = df["url"].sample(n=n, random_state=int(rng.randint(0, 1_000_000))).reset_index(drop=True)
        for repeat in range(args.repeats):
            t0 = time.perf_counter()
            ngram_matrix = bundle.ngram_processor.transform(urls)
            if hasattr(ngram_matrix, "toarray"):
                ngram_array = ngram_matrix.toarray().astype(np.float32)
            else:
                ngram_array = np.asarray(ngram_matrix, dtype=np.float32)
            t1 = time.perf_counter()
            handcrafted_rows = Parallel(n_jobs=args.n_jobs, prefer="threads")(
                delayed(extractor.generate_features)(url) for url in urls
            )
            handcrafted = np.asarray(handcrafted_rows, dtype=np.float32)
            t2 = time.perf_counter()
            full = np.hstack([handcrafted, ngram_array]).astype(np.float32, copy=False)
            selected = full[:, bundle.selected_indices]
            scaled = bundle.scaler.transform(selected.astype(np.float32))
            t3 = time.perf_counter()
            _ = predict_probabilities(bundle.model, scaled)
            t4 = time.perf_counter()
            rows.append(
                {
                    "batch_size": int(n),
                    "repeat": repeat,
                    "ngram_ms_per_url": (t1 - t0) * 1000.0 / n,
                    "handcrafted_ms_per_url": (t2 - t1) * 1000.0 / n,
                    "merge_scale_ms_per_url": (t3 - t2) * 1000.0 / n,
                    "classifier_ms_per_url": (t4 - t3) * 1000.0 / n,
                    "end_to_end_ms_per_url": (t4 - t0) * 1000.0 / n,
                    "urls_per_second": n / (t4 - t0) if (t4 - t0) > 0 else float("inf"),
                    "n_jobs": args.n_jobs,
                    "feature_count": int(len(bundle.selected_indices)),
                }
            )
    df_rows = pd.DataFrame(rows)
    write_csv(df_rows, outdir, "per_repeat.csv")
    summary = (
        df_rows.groupby("batch_size")
        .agg(
            end_to_end_p50=("end_to_end_ms_per_url", "median"),
            end_to_end_p95=("end_to_end_ms_per_url", lambda s: float(np.percentile(s, 95))),
            end_to_end_p99=("end_to_end_ms_per_url", lambda s: float(np.percentile(s, 99))),
            classifier_p50=("classifier_ms_per_url", "median"),
            handcrafted_p50=("handcrafted_ms_per_url", "median"),
            ngram_p50=("ngram_ms_per_url", "median"),
            urls_per_second_mean=("urls_per_second", "mean"),
        )
        .reset_index()
    )
    write_csv(summary, outdir, "summary.csv")


def replace_keywords(url: str) -> str:
    replacements = {
        "paypal": "paypa1",
        "login": "log-in",
        "verify": "review",
        "secure": "safe",
        "account": "profile",
        "password": "passcode",
        "update": "refresh",
        "bank": "portal",
        "office365": "office",
    }
    out = str(url)
    for old, new in replacements.items():
        out = re.sub(old, new, out, flags=re.IGNORECASE)
    return out


def append_benign_padding(url: str) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}utm_source=google&utm_medium=organic&ref=home"


def reduce_path_depth(url: str) -> str:
    text = str(url)
    prefix = ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        prefix = "http://"
    parsed = urllib.parse.urlsplit(prefix + text)
    segments = [s for s in parsed.path.split("/") if s]
    if len(segments) <= 2:
        return url
    new_path = "/" + "/".join([segments[0], segments[-1]])
    rebuilt = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, new_path, parsed.query, ""))
    return rebuilt[len(prefix) :] if prefix and rebuilt.startswith(prefix) else rebuilt


def remove_or_blur_query(url: str) -> str:
    text = str(url)
    prefix = ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        prefix = "http://"
    parsed = urllib.parse.urlsplit(prefix + text)
    query = "id=1001&view=home" if parsed.query else ""
    rebuilt = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))
    return rebuilt[len(prefix) :] if prefix and rebuilt.startswith(prefix) else rebuilt


def soften_tld(url: str) -> str:
    risky = [".xyz", ".top", ".club", ".info", ".biz", ".site", ".online", ".icu", ".click", ".work"]
    out = str(url)
    for tld in risky:
        out = re.sub(re.escape(tld) + r"(?=[:/?#]|$)", ".com", out, flags=re.IGNORECASE)
    return out


def perturbation_candidates(url: str) -> Dict[str, str]:
    return {
        "keyword_masking": replace_keywords(url),
        "benign_padding": append_benign_padding(url),
        "path_depth_reduction": reduce_path_depth(url),
        "query_blur": remove_or_blur_query(url),
        "tld_softening": soften_tld(url),
        "keyword_masking_plus_padding": append_benign_padding(replace_keywords(url)),
        "path_reduce_plus_keyword_masking": replace_keywords(reduce_path_depth(url)),
    }


def run_adversarial_stress(args: argparse.Namespace) -> None:
    outdir = ensure_dir(Path(args.outdir) / "adversarial_stress")
    if args.artifacts:
        bundle = load_bundle_from_artifacts(Path(args.artifacts))
    else:
        bundle, _, _ = train_default_bundle(args, args.seeds[0])

    test_path = Path(args.test_dataset or args.live_dataset or args.dataset)
    df = read_csv_dataset(test_path, sample=args.test_sample or args.sample, seed=args.seeds[0])
    phishing = df[df["phishing"] == 1].copy().reset_index(drop=True)
    if args.attack_sample and len(phishing) > args.attack_sample:
        phishing = phishing.sample(n=args.attack_sample, random_state=args.seeds[0]).reset_index(drop=True)

    orig_preds, orig_probs = predict_urls(bundle, phishing["url"], args)
    rows = []
    greedy_urls = []
    greedy_attack = []
    greedy_probs = []

    for idx, url in enumerate(phishing["url"]):
        candidates = perturbation_candidates(url)
        candidate_df = pd.Series(list(candidates.values()))
        _, candidate_probs = predict_urls(bundle, candidate_df, args)
        best_pos = int(np.argmin(candidate_probs))
        best_name = list(candidates.keys())[best_pos]
        best_url = list(candidates.values())[best_pos]
        greedy_urls.append(best_url)
        greedy_attack.append(best_name)
        greedy_probs.append(float(candidate_probs[best_pos]))
        for name, adv_url, adv_prob in zip(candidates.keys(), candidates.values(), candidate_probs):
            adv_pred = int(adv_prob >= bundle.threshold)
            rows.append(
                {
                    "row_id": idx,
                    "attack": name,
                    "original_url": url,
                    "perturbed_url": adv_url,
                    "original_prob": float(orig_probs[idx]),
                    "perturbed_prob": float(adv_prob),
                    "probability_drop": float(orig_probs[idx] - adv_prob),
                    "original_pred": int(orig_preds[idx]),
                    "perturbed_pred": adv_pred,
                    "attack_success": int(orig_preds[idx] == 1 and adv_pred == 0),
                    "changed": int(str(url) != str(adv_url)),
                }
            )

    greedy_preds = (np.asarray(greedy_probs) >= bundle.threshold).astype(int)
    summary_rows = []
    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        for attack, group in result_df.groupby("attack"):
            summary_rows.append(
                {
                    "attack": attack,
                    "n": int(len(group)),
                    "attack_success_rate": float(group["attack_success"].mean()),
                    "mean_probability_drop": float(group["probability_drop"].mean()),
                    "recall_after_attack": float(np.mean(group["perturbed_pred"] == 1)),
                    "changed_rate": float(group["changed"].mean()),
                }
            )
    summary_rows.append(
        {
            "attack": "greedy_min_probability",
            "n": int(len(phishing)),
            "attack_success_rate": float(np.mean((orig_preds == 1) & (greedy_preds == 0))) if len(phishing) else float("nan"),
            "mean_probability_drop": float(np.mean(orig_probs - np.asarray(greedy_probs))) if len(phishing) else float("nan"),
            "recall_after_attack": float(np.mean(greedy_preds == 1)) if len(phishing) else float("nan"),
            "changed_rate": float(np.mean([a != b for a, b in zip(phishing["url"], greedy_urls)])) if len(phishing) else float("nan"),
        }
    )

    clean_mask = phishing["url"].map(lambda u: any(term in str(u).lower() for term in HOSTING_CLEAN_LINK_TERMS))
    clean_df = phishing[clean_mask].reset_index(drop=True)
    if len(clean_df):
        clean_preds, clean_probs = predict_urls(bundle, clean_df["url"], args)
        clean_metrics = compute_metrics(np.ones(len(clean_df), dtype=int), clean_probs, bundle.threshold, clean_df["url"])
        clean_metrics.update({"attack": "clean_link_hosting_subset", "n": int(len(clean_df))})
        summary_rows.append(clean_metrics)

    write_csv(result_df, outdir, "per_url_attacks.csv")
    write_csv(
        pd.DataFrame(
            {
                "original_url": phishing["url"],
                "greedy_url": greedy_urls,
                "greedy_attack": greedy_attack,
                "original_prob": orig_probs,
                "greedy_prob": greedy_probs,
                "original_pred": orig_preds,
                "greedy_pred": greedy_preds,
            }
        ),
        outdir,
        "greedy_attack_urls.csv",
    )
    write_csv(pd.DataFrame(summary_rows), outdir, "summary.csv")


def run_all(args: argparse.Namespace) -> None:
    for experiment in [
        "sdcs_ablation",
        "sdcs_loss_factorial",
        "focal_imbalance",
        "gamma_sensitivity",
        "repeated_runs",
        "feature_inventory",
        "compression_curve",
        "representation_stability",
        "external_cohort",
        "latency",
        "adversarial_stress",
    ]:
        LOGGER.info("===== Running %s =====", experiment)
        dispatch(experiment, args)


def dispatch(experiment: str, args: argparse.Namespace) -> None:
    if experiment == "sdcs_ablation":
        run_sdcs_ablation(args)
    elif experiment == "sdcs_loss_factorial":
        run_sdcs_loss_factorial(args)
    elif experiment == "focal_imbalance":
        run_focal_imbalance(args)
    elif experiment == "gamma_sensitivity":
        run_gamma_sensitivity(args)
    elif experiment == "repeated_runs":
        run_repeated_runs(args)
    elif experiment == "feature_inventory":
        run_feature_inventory(args)
    elif experiment == "compression_curve":
        run_compression_curve(args)
    elif experiment == "representation_stability":
        run_representation_stability(args)
    elif experiment == "external_cohort":
        analyze_external_cohort(args)
    elif experiment == "latency":
        measure_latency(args)
    elif experiment == "adversarial_stress":
        run_adversarial_stress(args)
    elif experiment == "all":
        run_all(args)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "sdcs_ablation",
            "sdcs_loss_factorial",
            "focal_imbalance",
            "gamma_sensitivity",
            "repeated_runs",
            "feature_inventory",
            "compression_curve",
            "representation_stability",
            "external_cohort",
            "latency",
            "adversarial_stress",
            "all",
        ],
    )
    parser.add_argument("--dataset", default=default_data_file("PhishFusion.csv"))
    parser.add_argument("--train-dataset", default=None)
    parser.add_argument("--test-dataset", default=None)
    parser.add_argument("--live-dataset", default=default_data_file("PhishTank.csv"))
    parser.add_argument("--alternate-benign", default=None)
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--outdir", default=str(ROOT / "results" / "litephish"))
    parser.add_argument("--split", choices=["domain", "random"], default="domain")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--sample", type=int, default=None, help="Stratified sample size for quick/debug runs.")
    parser.add_argument("--test-sample", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Force a tiny run to validate code paths.")
    parser.add_argument("--dedup-cross-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument("--feature-budget", type=int, default=568)
    parser.add_argument("--budgets", nargs="+", type=int, default=[100, 250, 568, 1000, 2000, 5000])
    parser.add_argument("--score-sample-size", type=int, default=20000)
    parser.add_argument(
        "--max-ngram-features",
        type=int,
        default=None,
        help="Optional debug/smoke cap applied after stage-1 n-gram selection. Leave unset for paper runs.",
    )
    parser.add_argument("--selector-estimators", type=int, default=200)
    parser.add_argument("--stability-runs", type=int, default=20)
    parser.add_argument("--sample-fraction", type=float, default=0.905)
    parser.add_argument("--frequency-threshold", type=float, default=0.705)
    parser.add_argument("--regularization-strength", type=float, default=0.191)
    parser.add_argument("--mi-weight", type=float, default=0.318)
    parser.add_argument("--alpha-sdcs", type=float, default=0.767)
    parser.add_argument("--correlation-threshold", type=float, default=0.956)

    parser.add_argument("--focal-alpha", type=float, default=0.95)
    parser.add_argument("--focal-gamma", type=float, default=0.30)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.50, 0.65, 0.80, 0.90, 0.95])
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.0, 0.3, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])

    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 10, 100, 1000, 10000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--attack-sample", type=int, default=2000)
    parser.add_argument("--run-inventory-selection", action="store_true")
    parser.add_argument("--with-shap", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.seeds = args.seeds[:1]
    args.sample = min(args.sample or 600, 600)
    args.test_sample = min(args.test_sample or args.sample, args.sample)
    args.feature_budget = min(args.feature_budget, 40)
    args.max_ngram_features = min(args.max_ngram_features or 300, 300)
    args.budgets = [20, 40]
    args.score_sample_size = min(args.score_sample_size, 500)
    args.selector_estimators = min(args.selector_estimators, 20)
    args.stability_runs = min(args.stability_runs, 3)
    args.alphas = args.alphas[:2]
    args.gammas = args.gammas[:2]
    args.batch_sizes = [1, 10]
    args.repeats = 2
    args.attack_sample = min(args.attack_sample, 20)
    LOGGER.info("Smoke mode enabled: sample=%s seeds=%s budget=%s", args.sample, args.seeds, args.feature_budget)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    apply_smoke_defaults(args)
    ensure_dir(Path(args.outdir))
    write_json(vars(args), Path(args.outdir), f"{args.experiment}_args.json")
    dispatch(args.experiment, args)


if __name__ == "__main__":
    main()
