#!/usr/bin/env python3
"""Measure resource use for the five inference pipelines under a common protocol."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import pickle
import platform
import statistics
import sys
import threading
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil


MODELS = ("Ebbu", "E2Phish", "MUDS", "TabNet", "LitePhish")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "standard_deviation": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def cgroup_v2_limits() -> dict[str, object]:
    result: dict[str, object] = {"detected": False, "cpu_max": None, "memory_max_bytes": None}
    try:
        membership = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        unified = next(line.split(":", 2)[2] for line in membership if line.startswith("0::"))
        root = Path("/sys/fs/cgroup") / unified.lstrip("/")
        cpu_text = (root / "cpu.max").read_text(encoding="utf-8").strip()
        memory_text = (root / "memory.max").read_text(encoding="utf-8").strip()
        quota, period = cpu_text.split()
        return {
            "detected": True,
            "cpu_max": cpu_text,
            "cpu_quota_cores": None if quota == "max" else float(quota) / float(period),
            "memory_max_bytes": None if memory_text == "max" else int(memory_text),
        }
    except (OSError, StopIteration, ValueError):
        return result


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


def force_single_thread(estimator) -> None:
    """Set fitted sklearn/LightGBM thread controls to one where exposed."""
    if not hasattr(estimator, "get_params") or not hasattr(estimator, "set_params"):
        return
    params = estimator.get_params(deep=True)
    updates = {key: 1 for key in params if key == "n_jobs" or key.endswith("__n_jobs")}
    if updates:
        try:
            estimator.set_params(**updates)
        except (TypeError, ValueError):
            pass


class Adapter:
    def __init__(self, name: str, artifact_dir: Path, pipeline_root: Path, baseline_root: Path) -> None:
        self.name = name
        self.artifact_dir = artifact_dir
        self.pipeline_root = pipeline_root
        self.baseline_root = baseline_root
        self.required_files: list[Path] = []
        self.model = None
        self.aux = {}
        self.feature_code_paths: list[Path] = []

    def load(self) -> None:
        if self.name == "MUDS":
            self.required_files = [self.artifact_dir / "PhishFusion_model.joblib"]
            code = self.baseline_root / "MUDS" / "features.py"
            self.feature_code_paths = [code]
            self.aux["features"] = load_module("table_vi_muds_features", code)
            self.model = joblib.load(self.required_files[0])
        elif self.name == "E2Phish":
            self.required_files = [
                self.artifact_dir / "PhishFusion_model.joblib",
                self.artifact_dir / "PhishFusion_selected_features.joblib",
            ]
            code = self.baseline_root / "E2Phish" / "feature_extractor.py"
            self.feature_code_paths = [code]
            self.aux["features"] = load_module("table_vi_e2phish_features", code)
            self.model = joblib.load(self.required_files[0])
            self.aux["selected"] = list(joblib.load(self.required_files[1]))
        elif self.name == "Ebbu":
            self.required_files = [self.artifact_dir / "PhishFusion_model.joblib"]
            feature_dir = self.baseline_root / "ebbu2017"
            code = feature_dir / "pdd_impl.py"
            support = [feature_dir / name for name in ("keywords.txt", "allbrands.txt", "alexa_tld.txt")]
            self.required_files.extend(path for path in support if path.is_file())
            self.feature_code_paths = [code]
            old_cwd = Path.cwd()
            try:
                os.chdir(feature_dir)
                self.aux["features"] = load_module("table_vi_ebbu_features", code)
            finally:
                os.chdir(old_cwd)
            self.model = joblib.load(self.required_files[0])
        elif self.name == "TabNet":
            self.required_files = [self.artifact_dir / "PhishFusion_artifacts.joblib"]
            code = self.baseline_root / "tabnet" / "data_preprocessing.py"
            self.feature_code_paths = [code]
            self.aux["features"] = load_module("table_vi_tabnet_features", code)
            bundle = joblib.load(self.required_files[0])
            self.model = bundle["model"]
            self.aux.update(bundle)
        elif self.name == "LitePhish":
            sys.path.insert(0, str(self.pipeline_root))
            sys.path.insert(0, str(self.pipeline_root / "experiments"))
            from model import FocalLossLGBM

            setattr(sys.modules["__main__"], "FocalLossLGBM", FocalLossLGBM)
            self.required_files = [
                self.artifact_dir / "model.pkl",
                self.artifact_dir / "scaler.pkl",
                self.artifact_dir / "ngram_processor.pkl",
                self.artifact_dir / "selected_features.pkl",
                self.artifact_dir / "all_feature_names.pkl",
            ]
            self.feature_code_paths = [
                self.pipeline_root / "feature_extraction.py",
                self.pipeline_root / "ngram_processing.py",
            ]
            self.model = joblib.load(self.required_files[0])
            self.aux["scaler"] = joblib.load(self.required_files[1])
            self.aux["ngram_processor"] = joblib.load(self.required_files[2])
            self.aux["selected"] = np.asarray(joblib.load(self.required_files[3]), dtype=int)
            self.aux["feature_names"] = list(joblib.load(self.required_files[4]))
            from feature_extraction import FeatureExtractor
            from experiments.run_litephish_experiments import predict_probabilities

            self.aux["FeatureExtractor"] = FeatureExtractor
            self.aux["predict_probabilities"] = predict_probabilities
        else:
            raise ValueError(self.name)
        for path in self.required_files + self.feature_code_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        force_single_thread(self.model)

    def extract(self, urls: pd.Series):
        frame = pd.DataFrame({"url": urls.astype(str).to_numpy(), "phishing": np.zeros(len(urls), dtype=int)})
        if self.name == "MUDS":
            return self.aux["features"].extract_features(frame)
        if self.name == "E2Phish":
            processed = self.aux["features"].extract_features_from_dataframe(frame)
            if len(processed) != len(frame):
                raise RuntimeError(f"E2Phish extractor retained {len(processed)}/{len(frame)} URLs")
            return processed.drop(columns=["labels"]).select_dtypes(include=[np.number])
        if self.name == "Ebbu":
            processed = self.aux["features"].extract_features_from_dataframe(frame)
            if len(processed) != len(frame):
                raise RuntimeError(f"Ebbu extractor retained {len(processed)}/{len(frame)} URLs")
            return processed.drop(columns=["labels"])
        if self.name == "TabNet":
            return frame["url"].apply(self.aux["features"].extract_features).apply(pd.Series).fillna(0)
        extractor = self.aux["FeatureExtractor"]()
        handcrafted = np.asarray([extractor.generate_features(url) for url in frame["url"]], dtype=np.float32)
        ngram_sparse = self.aux["ngram_processor"].transform(frame["url"])
        ngram = ngram_sparse.toarray().astype(np.float32, copy=False)
        return np.hstack([handcrafted, ngram]).astype(np.float32, copy=False)

    def preprocess(self, raw):
        if self.name == "E2Phish":
            return raw.loc[:, self.aux["selected"]]
        if self.name == "TabNet":
            imputed = self.aux["imputer"].transform(raw)
            scaled = self.aux["scaler"].transform(imputed)
            return scaled[:, np.asarray(self.aux["selected_indices"], dtype=int)]
        if self.name == "LitePhish":
            selected = raw[:, self.aux["selected"]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self.aux["scaler"].transform(selected.astype(np.float32, copy=False))
        return raw

    def predict(self, prepared) -> np.ndarray:
        if self.name == "LitePhish":
            return np.asarray(self.aux["predict_probabilities"](self.model, prepared), dtype=float)
        return np.asarray(self.model.predict_proba(prepared)[:, 1], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--declared-cpu-limit", type=float, default=None)
    parser.add_argument("--declared-memory-limit-mib", type=int, default=None)
    args = parser.parse_args()

    process = psutil.Process()
    monitor = PeakRSS()
    monitor.start()
    rss_before_load = process.memory_info().rss
    load_start = time.perf_counter()
    adapter = Adapter(args.model, args.artifact_dir, args.pipeline_root, args.baseline_root)
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        adapter.load()
    load_seconds = time.perf_counter() - load_start
    rss_after_load = process.memory_info().rss

    frame = pd.read_csv(args.dataset, encoding="latin1")
    url_column = "url" if "url" in frame else "URL"
    urls_all = frame[url_column].astype(str).reset_index(drop=True)
    if len(urls_all) < args.sample_size:
        raise ValueError("sample larger than dataset")
    rng = np.random.RandomState(args.seed)
    sample_ids = rng.choice(len(urls_all), size=args.sample_size, replace=False)
    urls = urls_all.iloc[sample_ids].reset_index(drop=True)

    first_pass_hash = hashlib.sha256()

    def run_batch(batch: pd.Series, capture_predictions: bool = False) -> dict[str, float]:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            start = time.perf_counter()
            raw = adapter.extract(batch)
            feature_seconds = time.perf_counter() - start
            start = time.perf_counter()
            prepared = adapter.preprocess(raw)
            preprocessing_seconds = time.perf_counter() - start
            start = time.perf_counter()
            probability = adapter.predict(prepared)
            prediction_seconds = time.perf_counter() - start
        if len(probability) != len(batch):
            raise RuntimeError("prediction length mismatch")
        if capture_predictions:
            first_pass_hash.update(np.asarray(probability, dtype="<f8").tobytes())
        return {
            "feature_extraction": feature_seconds,
            "numeric_preprocessing": preprocessing_seconds,
            "model_prediction": prediction_seconds,
            "end_to_end": feature_seconds + preprocessing_seconds + prediction_seconds,
        }

    for index in range(args.warmup_batches):
        start = (index * args.batch_size) % len(urls)
        run_batch(urls.iloc[start : start + args.batch_size])

    raw_times = {name: [] for name in ("feature_extraction", "numeric_preprocessing", "model_prediction", "end_to_end")}
    executions = 0
    for repeat in range(args.repeats):
        for start in range(0, len(urls), args.batch_size):
            batch = urls.iloc[start : min(start + args.batch_size, len(urls))]
            timings = run_batch(batch, capture_predictions=(repeat == 0))
            executions += len(batch)
            for phase, seconds in timings.items():
                raw_times[phase].append(seconds / len(batch) * 1000.0)

    peak_rss = monitor.stop()
    artifact_records = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in adapter.required_files
    }
    result = {
        "benchmark": "Common-protocol model resource comparison",
        "model": args.model,
        "profile_name": args.profile_name,
        "profile": {
            "declared_cpu_limit_cores": args.declared_cpu_limit,
            "declared_memory_limit_mib": args.declared_memory_limit_mib,
            "batch_size": args.batch_size,
            "sample_size_unique": args.sample_size,
            "repeats": args.repeats,
            "measured_url_executions": executions,
            "warmup_batches_excluded": args.warmup_batches,
        },
        "runtime": {
            "system": platform.system(), "machine": platform.machine(),
            "python": platform.python_version(), "logical_cpu_visible": os.cpu_count(),
            "cpu_affinity_count": len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else None,
            "cgroup_v2_effective_limits": cgroup_v2_limits(),
        },
        "dataset": {
            "name": args.dataset.name, "sha256": sha256(args.dataset),
            "available_rows": len(urls_all), "sampling_seed": args.seed,
            "sample_row_id_sha256": hashlib.sha256(np.asarray(sample_ids, dtype="<i8").tobytes()).hexdigest(),
        },
        "artifacts": artifact_records,
        "total_artifact_bytes": int(sum(item["bytes"] for item in artifact_records.values())),
        "feature_code_sha256": {path.name: sha256(path) for path in adapter.feature_code_paths},
        "load_seconds": load_seconds,
        "memory": {
            "rss_before_load_mib": rss_before_load / (1024 ** 2),
            "rss_after_load_mib": rss_after_load / (1024 ** 2),
            "artifact_load_rss_delta_mib": (rss_after_load - rss_before_load) / (1024 ** 2),
            "whole_process_peak_rss_mib": peak_rss / (1024 ** 2),
        },
        "latency_ms_per_url": {phase: describe(values) for phase, values in raw_times.items()},
        "throughput_urls_per_second": 1000.0 / statistics.mean(raw_times["end_to_end"]),
        "first_pass_probability_sha256_float64_le": first_pass_hash.hexdigest(),
        "definitions": {
            "feature_extraction": "released model-specific conversion from raw URL strings to the full numeric feature representation",
            "numeric_preprocessing": "trained-feature selection and any fitted imputation/scaling",
            "model_prediction": "probability prediction from the already trained PhishFusion-source artifact",
            "peak_ram": "process plus child-process RSS sampled every 5 ms, cross-checked with GNU time",
            "artifact_size": "serialized inference artifacts and required fitted/support data; source code excluded",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
