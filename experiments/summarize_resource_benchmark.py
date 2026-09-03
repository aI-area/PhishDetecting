#!/usr/bin/env python3
"""Create compact analysis summaries from resource benchmark JSON."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "results" / "resource_benchmark"
FILES = [
    "constrained_1cpu_256mb.json",
    "constrained_online_batch1.json",
    "constrained_1cpu_512mb.json",
    "constrained_2cpu_1gb.json",
    "server_16worker.json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


rows = []
source_hashes = {}
for name in FILES:
    path = ROOT / name
    data = json.loads(path.read_text(encoding="utf-8"))
    source_hashes[name] = sha256(path)
    rows.append({
        "profile": data["profile_name"],
        "cpu_limit_cores": data["runtime"]["cgroup_v2_effective_limits"].get("cpu_quota_cores"),
        "memory_limit_mib": (
            data["runtime"]["cgroup_v2_effective_limits"].get("memory_max_bytes") or 0
        ) / (1024 ** 2) or "",
        "batch_size": data["profile"]["batch_size"],
        "measured_url_executions": data["profile"]["measured_url_executions"],
        "artifact_size_mb_decimal": data["model"]["total_artifact_bytes"] / 1_000_000,
        "model_file_size_mb_decimal": data["model"]["artifact_files"]["model"]["bytes"] / 1_000_000,
        "selected_features": data["model"]["selected_feature_count"],
        "artifact_load_rss_delta_mib": data["memory"]["artifact_load_rss_delta_mib"],
        "peak_rss_mib": data["memory"]["whole_process_peak_rss_mib"],
        "feature_extraction_ms_per_url": data["feature_extraction_ms_per_url"]["mean"],
        "numeric_preprocessing_ms_per_url": data["latency_ms_per_url"]["numeric_preprocessing"]["mean"],
        "model_prediction_ms_per_url": data["latency_ms_per_url"]["model_prediction"]["mean"],
        "end_to_end_ms_per_url": data["latency_ms_per_url"]["end_to_end"]["mean"],
        "throughput_urls_per_second": data["throughput_urls_per_second_from_mean_end_to_end"],
    })

with (ROOT / "resource_benchmark_summary.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

compact_features = 568
raw_features = 56_601
audit = {
    "benchmark": "LitePhish resource-constrained deployment",
    "status": "complete",
    "primary_analysis_profiles": [
        "constrained_online_1cpu_256MiB_batch1",
        "constrained_1cpu_512MiB",
    ],
    "hardware": {
        "architecture": "x86_64",
        "processor": "Intel Xeon Gold 6240 at 2.60 GHz",
        "accelerator_used": False,
        "constraint_mechanism": "Linux cgroup v2 via systemd user scope",
        "arm_or_physical_embedded_device": False,
        "interpretation": "enforced constrained execution profile; not an ARM-device measurement",
    },
    "workload": {
        "dataset": "PhishTank.csv",
        "dataset_sha256": "925bab1b18e59bdf6bca803b8e559fc1f99593124ec546e9b1c6d779142cf356",
        "sampling_seed": 42,
        "warmup_excluded": True,
        "model_training_performed": False,
    },
    "representation": {
        "raw_features": raw_features,
        "compact_features": compact_features,
        "feature_reduction_percent": (1 - compact_features / raw_features) * 100,
        "raw_dense_float32_bytes_per_url": raw_features * 4,
        "compact_dense_float32_bytes_per_url": compact_features * 4,
        "dense_input_memory_reduction_percent": (1 - compact_features / raw_features) * 100,
    },
    "metric_definitions": {
        "artifact_size": "model + scaler + fitted n-gram processor + selected-index and feature-name metadata",
        "feature_extraction": "86 handcrafted URL features plus selected character 2-3 gram transform",
        "peak_ram": "process and child-process maximum RSS, sampled at 5 ms and cross-checked with GNU time",
        "online": "batch size 1",
        "batched": "batch size 100",
    },
    "source_json_sha256": source_hashes,
    "benchmark_script": "benchmark_litephish_resources.py",
}
(ROOT / "resource_benchmark_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps({"rows": len(rows), "audit": str(ROOT / "resource_benchmark_audit.json")}, indent=2))
