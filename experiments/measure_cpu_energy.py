#!/usr/bin/env python3
"""Measure CPU-package energy for the five inference pipelines."""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.wintypes as wt
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil


MODELS = ("Ebbu", "E2Phish", "MUDS", "TabNet", "LitePhish")
MEASURED_REPEATS = {"Ebbu": 1, "E2Phish": 3, "MUDS": 40, "TabNet": 1, "LitePhish": 12}
MAP_NAME = r"Global\HWiNFO_SENS_SM2"
FILE_MAP_READ = 0x0004
MAGIC = int.from_bytes(b"HWiS", "little")


class Header(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32), ("version", ctypes.c_uint32),
        ("interface_version", ctypes.c_uint32), ("last_update", ctypes.c_int64),
        ("sensor_offset", ctypes.c_uint32), ("sensor_size", ctypes.c_uint32),
        ("sensor_count", ctypes.c_uint32), ("reading_offset", ctypes.c_uint32),
        ("reading_size", ctypes.c_uint32), ("reading_count", ctypes.c_uint32),
    ]


class Reading(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("reading_type", ctypes.c_uint32), ("sensor_index", ctypes.c_uint32),
        ("reading_id", ctypes.c_uint32), ("name_original", ctypes.c_char * 128),
        ("name_user", ctypes.c_char * 128), ("unit", ctypes.c_char * 16),
        ("value", ctypes.c_double), ("value_min", ctypes.c_double),
        ("value_max", ctypes.c_double), ("value_avg", ctypes.c_double),
    ]


def decode(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace").strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PackagePowerReader:
    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._map_view = kernel32.MapViewOfFile
        self._map_view.argtypes = [wt.HANDLE, wt.DWORD, wt.DWORD, wt.DWORD, ctypes.c_size_t]
        self._map_view.restype = wt.LPVOID
        self._unmap = kernel32.UnmapViewOfFile
        self._unmap.argtypes = [wt.LPCVOID]
        self._unmap.restype = wt.BOOL
        self._close = kernel32.CloseHandle
        self._close.argtypes = [wt.HANDLE]
        self._close.restype = wt.BOOL
        open_mapping = kernel32.OpenFileMappingW
        open_mapping.argtypes = [wt.DWORD, wt.BOOL, wt.LPCWSTR]
        open_mapping.restype = wt.HANDLE
        self.handle = open_mapping(FILE_MAP_READ, False, MAP_NAME)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "HWiNFO shared memory unavailable")
        mapped = self._map_view(self.handle, FILE_MAP_READ, 0, 0, 0)
        if not mapped:
            error = ctypes.get_last_error()
            self._close(self.handle)
            raise OSError(error, "HWiNFO shared memory mapping failed")
        self.address = int(ctypes.cast(mapped, ctypes.c_void_p).value)
        header = self._header()
        if header.magic != MAGIC:
            raise RuntimeError("Invalid HWiNFO shared-memory signature")
        self.reading_offset = header.reading_offset
        self.reading_size = header.reading_size
        self.reading_index = self._find_package_power(header)

    def _header(self) -> Header:
        return Header.from_buffer_copy(ctypes.string_at(self.address, ctypes.sizeof(Header)))

    def _record(self, index: int) -> Reading:
        address = self.address + self.reading_offset + index * self.reading_size
        return Reading.from_buffer_copy(ctypes.string_at(address, ctypes.sizeof(Reading)))

    def _find_package_power(self, header: Header) -> int:
        matches = []
        for index in range(header.reading_count):
            record = self._record(index)
            if decode(bytes(record.name_original)) == "CPU Package Power" and decode(bytes(record.unit)) == "W":
                matches.append(index)
        if len(matches) != 1:
            raise RuntimeError(f"Expected one CPU Package Power reading, found {len(matches)}")
        return matches[0]

    def read(self) -> float:
        value = float(self._record(self.reading_index).value)
        if not math.isfinite(value) or value < 0 or value > 1000:
            raise RuntimeError(f"Invalid CPU Package Power reading: {value}")
        return value

    def close(self) -> None:
        if self.address:
            self._unmap(ctypes.c_void_p(self.address))
            self.address = 0
        if self.handle:
            self._close(self.handle)
            self.handle = None


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sample_for(reader: PackagePowerReader, seconds: float, interval: float) -> list[tuple[int, float]]:
    samples = []
    deadline = time.monotonic() + seconds
    while True:
        samples.append((time.monotonic_ns(), reader.read()))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))
    return samples


def integrate(samples: list[tuple[int, float]], start_ns: int, end_ns: int) -> float:
    if end_ns <= start_ns:
        raise ValueError("Non-positive measurement interval")
    ordered = sorted(samples)
    if not ordered or ordered[0][0] > start_ns or ordered[-1][0] < end_ns:
        raise RuntimeError("Power samples do not bracket the measured phase")

    def interpolated(at_ns: int) -> float:
        for left, right in zip(ordered, ordered[1:]):
            if left[0] <= at_ns <= right[0]:
                if right[0] == left[0]:
                    return left[1]
                weight = (at_ns - left[0]) / (right[0] - left[0])
                return left[1] + weight * (right[1] - left[1])
        return ordered[-1][1]

    points = [(start_ns, interpolated(start_ns))]
    points.extend((stamp, watts) for stamp, watts in ordered if start_ns < stamp < end_ns)
    points.append((end_ns, interpolated(end_ns)))
    joules = 0.0
    for left, right in zip(points, points[1:]):
        joules += (right[0] - left[0]) / 1e9 * (left[1] + right[1]) / 2.0
    return joules


def mean_power(samples: list[tuple[int, float]]) -> float:
    return statistics.mean(value for _, value in samples)


def artifact_dir(root: Path, model: str) -> Path:
    mapping = {
        "MUDS": root / "MUDS",
        "E2Phish": root / "E2Phish",
        "Ebbu": root / "Ebbu",
        "TabNet": root / "TabNet",
        "LitePhish": root / "LitePhish",
    }
    return mapping[model]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hwinfo-exe", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--idle-seconds", type=float, default=10.0)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--cpu-core", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    raw_root = args.output_root / "raw"
    raw_root.mkdir(exist_ok=True)

    reader = PackagePowerReader()
    environment = os.environ.copy()
    environment.update({
        "PYTHONHASHSEED": "42", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    order_rng = random.Random(args.seed)
    schedule = []
    for trial in range(1, args.trials + 1):
        trial_models = list(MODELS)
        order_rng.shuffle(trial_models)
        schedule.extend((trial, model) for model in trial_models)
    atomic_json(args.output_root / "schedule.json", {"seed": args.seed, "schedule": schedule})

    rows = []
    try:
        for sequence, (trial, model) in enumerate(schedule, start=1):
            stem = f"{sequence:02d}_{model}_trial{trial}"
            trial_root = raw_root / stem
            trial_root.mkdir(exist_ok=True)
            print(f"[{sequence}/{len(schedule)}] idle-before {model} trial {trial}", flush=True)
            pre = sample_for(reader, args.idle_seconds, args.sample_interval)

            start_marker = trial_root / "phase_start.json"
            end_marker = trial_root / "phase_end.json"
            result_path = trial_root / "runner_result.json"
            log_path = trial_root / "runner.log"
            command = [
                str(args.python), str(args.runner), model, str(artifact_dir(args.artifact_root, model)),
                str(args.dataset), str(result_path), str(start_marker), str(end_marker),
                "--pipeline-root", str(args.pipeline_root), "--baseline-root", str(args.baseline_root),
                "--sample-size", "5000", "--batch-size", "100", "--warmup-batches", "2", "--seed", "42",
                "--measured-repeats", str(MEASURED_REPEATS[model]),
            ]
            print(f"[{sequence}/{len(schedule)}] measuring {model} trial {trial}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                psutil.Process(process.pid).cpu_affinity([args.cpu_core])
                live_samples = []
                while process.poll() is None:
                    live_samples.append((time.monotonic_ns(), reader.read()))
                    time.sleep(args.sample_interval)
                live_samples.append((time.monotonic_ns(), reader.read()))
            if process.returncode != 0:
                raise RuntimeError(f"{model} trial {trial} failed with rc={process.returncode}; see {log_path}")

            print(f"[{sequence}/{len(schedule)}] idle-after {model} trial {trial}", flush=True)
            post = sample_for(reader, args.idle_seconds, args.sample_interval)
            start_ns = json.loads(start_marker.read_text(encoding="utf-8"))["monotonic_ns"]
            end_ns = json.loads(end_marker.read_text(encoding="utf-8"))["monotonic_ns"]
            runner_result = json.loads(result_path.read_text(encoding="utf-8"))
            gross_joules = integrate(live_samples, start_ns, end_ns)
            idle_watts = (mean_power(pre) + mean_power(post)) / 2.0
            duration = (end_ns - start_ns) / 1e9
            adjusted_joules = gross_joules - idle_watts * duration
            row = {
                "sequence": sequence, "trial": trial, "model": model,
                "duration_seconds": duration,
                "measured_repeats": MEASURED_REPEATS[model],
                "url_executions": runner_result["measured_phase"]["url_executions"],
                "gross_package_joules": gross_joules,
                "idle_pre_watts": mean_power(pre), "idle_post_watts": mean_power(post),
                "idle_reference_watts": idle_watts,
                "idle_adjusted_package_joules": adjusted_joules,
                "gross_millijoules_per_url": gross_joules / runner_result["measured_phase"]["url_executions"] * 1000,
                "idle_adjusted_millijoules_per_url": adjusted_joules / runner_result["measured_phase"]["url_executions"] * 1000,
                "idle_adjusted_urls_per_joule": runner_result["measured_phase"]["url_executions"] / adjusted_joules if adjusted_joules > 0 else None,
                "power_sample_count": len(live_samples),
                "runner_cpu_affinity_count": runner_result["runtime"]["cpu_affinity_count"],
                "sample_row_id_sha256": runner_result["dataset"]["sample_row_id_sha256"],
                "probability_sha256_float64_le": runner_result["probability_sha256_float64_le"],
            }
            rows.append(row)
            with (trial_root / "power_samples.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["segment", "monotonic_ns", "cpu_package_power_watts"])
                writer.writerows(("idle_pre", *item) for item in pre)
                writer.writerows(("workload", *item) for item in live_samples)
                writer.writerows(("idle_post", *item) for item in post)
            atomic_json(trial_root / "energy_result.json", row)
            with (args.output_root / "energy_trials.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    finally:
        reader.close()

    grouped = []
    t_critical = 2.7764451051977987 if args.trials == 5 else None
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        values = [row["idle_adjusted_millijoules_per_url"] for row in model_rows]
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        half = t_critical * sd / math.sqrt(len(values)) if t_critical else None
        grouped.append({
            "model": model, "trials": len(values), "mean_idle_adjusted_mj_per_url": mean,
            "sd_idle_adjusted_mj_per_url": sd,
            "ci95_low_mj_per_url": mean - half if half is not None else None,
            "ci95_high_mj_per_url": mean + half if half is not None else None,
            "mean_gross_mj_per_url": statistics.mean(row["gross_millijoules_per_url"] for row in model_rows),
            "mean_idle_watts": statistics.mean(row["idle_reference_watts"] for row in model_rows),
            "mean_duration_seconds": statistics.mean(row["duration_seconds"] for row in model_rows),
            "mean_idle_adjusted_urls_per_joule": 1000.0 / mean if mean > 0 else None,
        })
    with (args.output_root / "energy_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(grouped[0]))
        writer.writeheader()
        writer.writerows(grouped)
    audit = {
        "status": "PASS", "measurement_scope": "CPU package only; not whole-system wall energy",
        "sensor": "HWiNFO CPU Package Power", "sensor_unit": "W",
        "integration": "trapezoidal integration over phase-marked samples",
        "idle_adjustment": (
            f"mean of immediate {args.idle_seconds:g}-second pre/post idle blocks "
            "times workload duration"
        ),
        "hardware": {
            "platform": platform.platform(), "processor": platform.processor(),
            "physical_cpu_cores": psutil.cpu_count(logical=False), "logical_cpu_cores": psutil.cpu_count(),
            "pinned_cpu_core": args.cpu_core,
        },
        "protocol": {
            "models": MODELS, "trials_per_model": args.trials, "randomization_seed": args.seed,
            "sample_size_unique_per_trial": 5000, "batch_size": 100,
            "measured_repeats_by_model": MEASURED_REPEATS,
            "duration_control": "fixed pilot-derived repeats chosen to make each phase approximately 30 seconds or longer",
            "warmup_batches_excluded": 2, "dataset_sha256": sha256(args.dataset),
            "hwinfo_sensor_interval_ms": 250,
            "reader_sample_interval_ms": args.sample_interval * 1000,
            "expected_dataset_sha256": "925bab1b18e59bdf6bca803b8e559fc1f99593124ec546e9b1c6d779142cf356",
            "expected_sample_row_id_sha256": "66fcfd4badd3d32a5aa54555d68a912c9130f4edebc618e1ede0e150d365dd79",
        },
        "software": {
            "python": sys.version, "hwinfo_executable_sha256": sha256(args.hwinfo_exe),
            "runner_sha256": sha256(args.runner), "orchestrator_sha256": sha256(Path(__file__)),
        },
        "limitations": [
            "Package power includes other CPU activity; paired local idle blocks are subtracted.",
            "The Intel Core i7-9700 energy host differs from the Xeon resource-measurement host.",
            "HWiNFO/CPU telemetry is not a wall-socket measurement and excludes non-package system energy.",
        ],
    }
    if audit["protocol"]["dataset_sha256"] != audit["protocol"]["expected_dataset_sha256"]:
        raise RuntimeError("Dataset hash mismatch")
    if any(row["sample_row_id_sha256"] != audit["protocol"]["expected_sample_row_id_sha256"] for row in rows):
        raise RuntimeError("Sample cohort hash mismatch")
    if any(row["runner_cpu_affinity_count"] != 1 for row in rows):
        raise RuntimeError("A runner was not pinned to exactly one CPU core")
    if any(row["duration_seconds"] < 25 for row in rows):
        raise RuntimeError("A measured phase was shorter than the 25-second minimum")
    if any(row["power_sample_count"] < 80 for row in rows):
        raise RuntimeError("A measured phase has fewer than 80 package-power samples")
    for model in MODELS:
        hashes = {row["probability_sha256_float64_le"] for row in rows if row["model"] == model}
        if len(hashes) != 1:
            raise RuntimeError(f"Non-deterministic prediction hash for {model}: {sorted(hashes)}")
    atomic_json(args.output_root / "experiment_audit.json", audit)
    print(json.dumps({"status": "PASS", "summary": grouped}, indent=2))


if __name__ == "__main__":
    main()
