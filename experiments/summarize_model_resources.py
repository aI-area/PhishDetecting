#!/usr/bin/env python3
"""Aggregate the common-protocol model resource measurements."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "results" / "resource_benchmark"
MODELS = ["Ebbu", "E2Phish", "MUDS", "TabNet", "LitePhish"]
MODES = ["online", "batched"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


rows = []
records = {}
for model in MODELS:
    for mode in MODES:
        path = ROOT / f"{model}_{mode}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        records[(model, mode)] = data
        latency = data["latency_ms_per_url"]
        rows.append({
            "model": model,
            "profile": mode,
            "unique_urls": data["profile"]["sample_size_unique"],
            "repeats": data["profile"]["repeats"],
            "measured_url_executions": data["profile"]["measured_url_executions"],
            "batch_size": data["profile"]["batch_size"],
            "artifact_size_mb": data["total_artifact_bytes"] / 1_000_000,
            "artifact_load_rss_delta_mib": data["memory"]["artifact_load_rss_delta_mib"],
            "peak_rss_mib": data["memory"]["whole_process_peak_rss_mib"],
            "feature_extraction_ms_per_url": latency["feature_extraction"]["mean"],
            "numeric_preprocessing_ms_per_url": latency["numeric_preprocessing"]["mean"],
            "model_prediction_ms_per_url": latency["model_prediction"]["mean"],
            "end_to_end_ms_per_url": latency["end_to_end"]["mean"],
            "throughput_urls_per_second": data["throughput_urls_per_second"],
        })

with (ROOT / "model_resources_long.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

compact = []
for model in MODELS:
    online = next(row for row in rows if row["model"] == model and row["profile"] == "online")
    batched = next(row for row in rows if row["model"] == model and row["profile"] == "batched")
    compact.append({
        "model": model,
        "artifact_size_mb": online["artifact_size_mb"],
        "peak_rss_mib": max(online["peak_rss_mib"], batched["peak_rss_mib"]),
        "online_feature_ms_per_url": online["feature_extraction_ms_per_url"],
        "online_model_ms_per_url": online["model_prediction_ms_per_url"],
        "online_total_ms_per_url": online["end_to_end_ms_per_url"],
        "batched_total_ms_per_url": batched["end_to_end_ms_per_url"],
        "batched_urls_per_second": batched["throughput_urls_per_second"],
    })
with (ROOT / "model_resources_summary.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(compact[0]))
    writer.writeheader()
    writer.writerows(compact)

dataset_hashes = {data["dataset"]["sha256"] for data in records.values()}
assert len(dataset_hashes) == 1
profile_sample_hashes = {}
for mode in MODES:
    values = {records[(model, mode)]["dataset"]["sample_row_id_sha256"] for model in MODELS}
    assert len(values) == 1
    profile_sample_hashes[mode] = next(iter(values))
for data in records.values():
    limits = data["runtime"]["cgroup_v2_effective_limits"]
    assert data["runtime"]["cpu_affinity_count"] == 1
    assert limits["cpu_quota_cores"] == 1.0
    assert limits["memory_max_bytes"] == 512 * 1024 * 1024

audit = {
    "status": "PASS",
    "models": MODELS,
    "profiles": {
        "online": {"unique_urls": 1000, "batch_size": 1, "repeats": 3, "warmup_batches_excluded": 20},
        "batched": {"unique_urls": 5000, "batch_size": 100, "repeats": 5, "warmup_batches_excluded": 2},
    },
    "dataset": "PhishTank.csv",
    "dataset_rows": 93498,
    "dataset_sha256": next(iter(dataset_hashes)),
    "sample_row_id_sha256_by_profile": profile_sample_hashes,
    "execution": {
        "sequential": True,
        "pinned_cpu_core": True,
        "cpu_affinity_count": 1,
        "cpu_quota_cores": 1.0,
        "memory_hard_limit_mib": 512,
        "accelerator_used": False,
        "hardware": "x86-64 Intel Xeon Gold 6240 at 2.60 GHz",
    },
    "definitions": next(iter(records.values()))["definitions"],
    "raw_result_sha256": {
        path.name: sha256(path) for path in sorted(ROOT.glob("*.json"))
        if path.name not in {"model_resources_provenance.json"}
    },
}
(ROOT / "model_resources_provenance.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps({"status": "PASS", "rows": len(rows), "compact_rows": len(compact)}, indent=2))
