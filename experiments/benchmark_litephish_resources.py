#!/usr/bin/env python3
"""Reproducible deployment-resource benchmark for a trained LitePhish bundle.

The benchmark deliberately separates URL feature construction from numeric
preprocessing and model prediction.  It does not fit or modify the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PeakRSS:
    def __init__(self, interval: float = 0.005) -> None:
        self.process = psutil.Process()
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _rss(self) -> int:
        total = self.process.memory_info().rss
        for child in self.process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            self._stop.wait(self.interval)

    def start(self) -> None:
        self.peak = self._rss()
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._rss())
        return self.peak


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "standard_deviation": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def cgroup_v2_limits() -> dict[str, object]:
    """Return only effective numeric controls for this process, not cgroup IDs."""
    result: dict[str, object] = {"detected": False, "cpu_max": None, "memory_max_bytes": None}
    try:
        membership = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        unified = next(line.split(":", 2)[2] for line in membership if line.startswith("0::"))
        root = Path("/sys/fs/cgroup") / unified.lstrip("/")
        cpu_text = (root / "cpu.max").read_text(encoding="utf-8").strip()
        memory_text = (root / "memory.max").read_text(encoding="utf-8").strip()
        quota, period = cpu_text.split()
        result = {
            "detected": True,
            "cpu_max": cpu_text,
            "cpu_quota_cores": None if quota == "max" else float(quota) / float(period),
            "memory_max_bytes": None if memory_text == "max" else int(memory_text),
        }
    except (OSError, StopIteration, ValueError):
        pass
    return result


def extract_handcrafted(urls: pd.Series, feature_extractor_cls, jobs: int) -> np.ndarray:
    extractor = feature_extractor_cls()
    if jobs == 1:
        rows = [extractor.generate_features(url) for url in urls]
    else:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=jobs, prefer="threads")(
            delayed(extractor.generate_features)(url) for url in urls
        )
    return np.asarray(rows, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--declared-cpu-limit", type=float, default=None)
    parser.add_argument("--declared-memory-limit-mib", type=int, default=None)
    args = parser.parse_args()

    if args.sample_size < args.batch_size or args.repeats < 1:
        raise ValueError("sample-size must be >= batch-size and repeats must be positive")
    sys.path.insert(0, str(args.pipeline_root))
    sys.path.insert(0, str(args.pipeline_root / "experiments"))
    from feature_extraction import FeatureExtractor
    from experiments.run_litephish_experiments import predict_probabilities

    bundled_format = (args.artifact_dir / "LitePhish_model.joblib").is_file()
    if bundled_format:
        artifact_paths = {
            "model": args.artifact_dir / "LitePhish_model.joblib",
            "scaler": args.artifact_dir / "LitePhish_scaler.joblib",
            "preprocessing": args.artifact_dir / "LitePhish_preprocessing.pkl",
        }
    else:
        artifact_paths = {
            "model": args.artifact_dir / "model.pkl",
            "scaler": args.artifact_dir / "scaler.pkl",
            "ngram_processor": args.artifact_dir / "ngram_processor.pkl",
            "selected_features": args.artifact_dir / "selected_features.pkl",
            "all_feature_names": args.artifact_dir / "all_feature_names.pkl",
        }
    for path in artifact_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    monitor = PeakRSS()
    monitor.start()
    process = psutil.Process()
    rss_before_load = process.memory_info().rss
    load_start = time.perf_counter()
    # Some notebook-created legacy pickles recorded the custom class under
    # __main__; expose the released implementation there before deserializing.
    from model import FocalLossLGBM
    setattr(sys.modules["__main__"], "FocalLossLGBM", FocalLossLGBM)
    model = joblib.load(artifact_paths["model"])
    scaler = joblib.load(artifact_paths["scaler"])
    if bundled_format:
        with artifact_paths["preprocessing"].open("rb") as stream:
            preprocessing = pickle.load(stream)
    else:
        preprocessing = {
            "ngram_processor": joblib.load(artifact_paths["ngram_processor"]),
            "selected_indices": joblib.load(artifact_paths["selected_features"]),
            "feature_names": joblib.load(artifact_paths["all_feature_names"]),
        }
    load_seconds = time.perf_counter() - load_start
    rss_after_load = process.memory_info().rss

    ngram_processor = preprocessing["ngram_processor"]
    selected_indices = np.asarray(preprocessing["selected_indices"], dtype=int)
    feature_names = list(preprocessing["feature_names"])

    frame = pd.read_csv(args.dataset, encoding="latin1")
    url_column = "url" if "url" in frame.columns else "URL"
    if url_column not in frame:
        raise ValueError("dataset must contain url or URL")
    urls_all = frame[url_column].astype(str).dropna().reset_index(drop=True)
    if len(urls_all) < args.sample_size:
        raise ValueError(f"requested {args.sample_size} URLs but dataset has {len(urls_all)}")
    rng = np.random.RandomState(args.seed)
    sample_ids = rng.choice(len(urls_all), size=args.sample_size, replace=False)
    urls = urls_all.iloc[sample_ids].reset_index(drop=True)

    def run_batch(batch: pd.Series) -> dict[str, float]:
        start = time.perf_counter()
        handcrafted = extract_handcrafted(batch, FeatureExtractor, args.jobs)
        handcrafted_seconds = time.perf_counter() - start

        start = time.perf_counter()
        ngram_sparse = ngram_processor.transform(batch)
        ngram = ngram_sparse.toarray().astype(np.float32, copy=False)
        ngram_seconds = time.perf_counter() - start

        start = time.perf_counter()
        full = np.hstack([handcrafted, ngram]).astype(np.float32, copy=False)
        selected = full[:, selected_indices]
        scaled = scaler.transform(selected.astype(np.float32, copy=False))
        preprocessing_seconds = time.perf_counter() - start

        start = time.perf_counter()
        probability = predict_probabilities(model, scaled)
        prediction_seconds = time.perf_counter() - start
        if len(probability) != len(batch):
            raise RuntimeError("prediction length mismatch")
        return {
            "handcrafted": handcrafted_seconds,
            "ngram": ngram_seconds,
            "numeric_preprocessing": preprocessing_seconds,
            "model_prediction": prediction_seconds,
            "end_to_end": handcrafted_seconds + ngram_seconds + preprocessing_seconds + prediction_seconds,
        }

    for index in range(args.warmup_batches):
        start = (index * args.batch_size) % len(urls)
        run_batch(urls.iloc[start : start + args.batch_size])

    raw: dict[str, list[float]] = {
        "handcrafted": [], "ngram": [], "numeric_preprocessing": [],
        "model_prediction": [], "end_to_end": [],
    }
    measured_urls = 0
    for repeat in range(args.repeats):
        for start in range(0, len(urls), args.batch_size):
            batch = urls.iloc[start : min(start + args.batch_size, len(urls))]
            times = run_batch(batch)
            measured_urls += len(batch)
            for phase, seconds in times.items():
                raw[phase].append(seconds / len(batch) * 1000.0)

    peak_rss = monitor.stop()
    artifacts = {
        name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in artifact_paths.items()
    }
    result = {
        "benchmark": "LitePhish deployment resources",
        "profile_name": args.profile_name,
        "profile": {
            "declared_cpu_limit_cores": args.declared_cpu_limit,
            "declared_memory_limit_mib": args.declared_memory_limit_mib,
            "jobs": args.jobs,
            "batch_size": args.batch_size,
            "sample_size_unique": args.sample_size,
            "repeats": args.repeats,
            "measured_url_executions": measured_urls,
            "warmup_batches_excluded": args.warmup_batches,
        },
        "runtime": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "logical_cpu_visible": os.cpu_count(),
            "cpu_affinity_count": len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else None,
            "total_memory_mib": psutil.virtual_memory().total / (1024 ** 2),
            "cgroup_v2_effective_limits": cgroup_v2_limits(),
        },
        "dataset": {
            "path_name": args.dataset.name,
            "sha256": sha256(args.dataset),
            "available_rows": len(urls_all),
            "sampling_seed": args.seed,
        },
        "model": {
            "selected_feature_count": int(len(selected_indices)),
            "full_feature_count": int(len(feature_names)),
            "artifact_files": artifacts,
            "total_artifact_bytes": int(sum(item["bytes"] for item in artifacts.values())),
            "load_seconds": load_seconds,
        },
        "memory": {
            "rss_before_artifact_load_mib": rss_before_load / (1024 ** 2),
            "rss_after_artifact_load_mib": rss_after_load / (1024 ** 2),
            "artifact_load_rss_delta_mib": (rss_after_load - rss_before_load) / (1024 ** 2),
            "whole_process_peak_rss_mib": peak_rss / (1024 ** 2),
        },
        "latency_ms_per_url": {phase: describe(values) for phase, values in raw.items()},
        "feature_extraction_ms_per_url": describe([
            handcrafted + ngram for handcrafted, ngram in zip(raw["handcrafted"], raw["ngram"])
        ]),
        "throughput_urls_per_second_from_mean_end_to_end": 1000.0 / statistics.mean(raw["end_to_end"]),
        "definitions": {
            "feature_extraction": "86 handcrafted URL features plus selected character 2-3 gram transform",
            "numeric_preprocessing": "feature concatenation, trained-feature selection, and StandardScaler transform",
            "model_prediction": "trained focal-loss-inspired LightGBM probability prediction",
            "peak_ram": "maximum resident set size of benchmark process plus child processes sampled every 5 ms",
            "artifact_size": "sum of serialized model, scaler, and fitted preprocessing bundle",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
