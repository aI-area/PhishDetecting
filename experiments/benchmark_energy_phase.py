#!/usr/bin/env python3
"""Run one phase-marked inference trial for local package-energy measurement."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from experiments.benchmark_model_resources import Adapter, sha256


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Ebbu", "E2Phish", "MUDS", "TabNet", "LitePhish"))
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("start_marker", type=Path)
    parser.add_argument("end_marker", type=Path)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--measured-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except ImportError:
        pass

    adapter = Adapter(args.model, args.artifact_dir, args.pipeline_root, args.baseline_root)
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        adapter.load()

    frame = pd.read_csv(args.dataset, encoding="latin1")
    url_column = "url" if "url" in frame else "URL"
    urls_all = frame[url_column].astype(str).reset_index(drop=True)
    rng = np.random.RandomState(args.seed)
    sample_ids = rng.choice(len(urls_all), size=args.sample_size, replace=False)
    urls = urls_all.iloc[sample_ids].reset_index(drop=True)

    def run_batch(batch: pd.Series) -> tuple[np.ndarray, dict[str, float]]:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            started = time.perf_counter()
            raw = adapter.extract(batch)
            feature_seconds = time.perf_counter() - started
            started = time.perf_counter()
            prepared = adapter.preprocess(raw)
            preprocessing_seconds = time.perf_counter() - started
            started = time.perf_counter()
            probability = adapter.predict(prepared)
            prediction_seconds = time.perf_counter() - started
        return np.asarray(probability, dtype=float), {
            "feature_extraction": feature_seconds,
            "numeric_preprocessing": preprocessing_seconds,
            "model_prediction": prediction_seconds,
        }

    for index in range(args.warmup_batches):
        start = (index * args.batch_size) % len(urls)
        run_batch(urls.iloc[start : start + args.batch_size])

    phase_totals = {
        "feature_extraction": 0.0,
        "numeric_preprocessing": 0.0,
        "model_prediction": 0.0,
    }
    probability_hash = hashlib.sha256()
    phase_start_ns = time.monotonic_ns()
    atomic_json(
        args.start_marker,
        {"event": "measured_phase_start", "monotonic_ns": phase_start_ns, "pid": os.getpid()},
    )
    for repeat in range(args.measured_repeats):
        for start in range(0, len(urls), args.batch_size):
            batch = urls.iloc[start : min(start + args.batch_size, len(urls))]
            probability, timing = run_batch(batch)
            if repeat == 0:
                probability_hash.update(np.asarray(probability, dtype="<f8").tobytes())
            for key, seconds in timing.items():
                phase_totals[key] += seconds
    phase_end_ns = time.monotonic_ns()
    atomic_json(
        args.end_marker,
        {"event": "measured_phase_end", "monotonic_ns": phase_end_ns, "pid": os.getpid()},
    )

    process = psutil.Process()
    result = {
        "status": "PASS",
        "model": args.model,
        "measured_phase": {
            "sample_size_unique": args.sample_size,
            "batch_size": args.batch_size,
            "measured_repeats": args.measured_repeats,
            "url_executions": args.sample_size * args.measured_repeats,
            "warmup_batches_excluded": args.warmup_batches,
            "start_monotonic_ns": phase_start_ns,
            "end_monotonic_ns": phase_end_ns,
            "duration_seconds": (phase_end_ns - phase_start_ns) / 1e9,
            "phase_seconds": phase_totals,
        },
        "runtime": {
            "pid": os.getpid(),
            "cpu_affinity": process.cpu_affinity(),
            "cpu_affinity_count": len(process.cpu_affinity()),
        },
        "dataset": {
            "path": str(args.dataset),
            "sha256": sha256(args.dataset),
            "rows": len(urls_all),
            "sampling_seed": args.seed,
            "sample_row_id_sha256": hashlib.sha256(
                np.asarray(sample_ids, dtype="<i8").tobytes()
            ).hexdigest(),
        },
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in adapter.required_files
        },
        "feature_code_sha256": {path.name: sha256(path) for path in adapter.feature_code_paths},
        "probability_sha256_float64_le": probability_hash.hexdigest(),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
