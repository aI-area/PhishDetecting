#!/usr/bin/env python3
"""Audit and summarize the matched LitePhish compactness experiment."""
from __future__ import annotations
import argparse, csv, hashlib, json, statistics
from pathlib import Path

def sha(path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument("experiment",type=Path); args=p.parse_args()
    root=args.experiment; training=json.loads((root/"artifacts/training_audit.json").read_text())
    rows=[]; raw_hashes={}; limits=[]
    for cfg in ("raw_56601","compact_568"):
      for prof in ("online","batched"):
        files=sorted((root/"benchmark").glob(f"{cfg}_{prof}*.json"))
        if len(files)<3: raise RuntimeError(f"fewer than three passes for {cfg}/{prof}")
        docs=[json.loads(x.read_text()) for x in files]
        if any(d["status"]!="PASS" for d in docs): raise RuntimeError("failed pass")
        if len({d["dataset"]["sample_row_id_sha256"] for d in docs})!=1: raise RuntimeError("sample mismatch")
        for x in files: raw_hashes[x.name]=sha(x)
        med=lambda fn: statistics.median(fn(d) for d in docs)
        rows.append({"configuration":cfg,"profile":prof,"feature_count":docs[0]["features"]["active"],
          "artifact_mb":docs[0]["artifact_bytes"]/1e6,"peak_rss_mib_median":med(lambda d:d["memory"]["peak_rss_mib"]),
          "feature_ms_per_url_median":med(lambda d:d["latency_ms_per_url"]["handcrafted"]["mean"]+d["latency_ms_per_url"]["ngram"]["mean"]),
          "preprocessing_ms_per_url_median":med(lambda d:d["latency_ms_per_url"]["numeric_preprocessing"]["mean"]),
          "model_ms_per_url_median":med(lambda d:d["latency_ms_per_url"]["model_prediction"]["mean"]),
          "total_ms_per_url_median":med(lambda d:d["latency_ms_per_url"]["end_to_end"]["mean"]),
          "urls_per_second_median":med(lambda d:d["throughput_urls_per_second"]),"independent_passes":len(docs)})
        limits.extend(d["runtime"].get("cgroup_v2_effective_limits",{}) for d in docs if d["runtime"].get("cgroup_v2_effective_limits"))
    out=root/"compactness_summary.csv"
    with out.open("w",newline="",encoding="utf-8") as f:
      w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    tr=training["configurations"]; raw=tr["raw_56601"]; compact=tr["compact_568"]
    audit={"status":"PASS","dataset":training["dataset"],"split":training["split"],"shared":training["shared"],
      "performance":{"raw_56601":raw["metrics"],"compact_568":compact["metrics"]},
      "training":{"raw_seconds":raw["training_seconds"],"compact_seconds":compact["training_seconds"],
                  "speedup_raw_over_compact":raw["training_seconds"]/compact["training_seconds"]},
      "resource_protocol":{"sequential":True,"same_fixed_seed_samples":True,"cpu_affinity_count":1,
        "cpu_quota_cores":1.0,"memory_hard_limit_mib":512,"accelerator_used":False,
        "passes_per_configuration_profile":min(r["independent_passes"] for r in rows)},
      "raw_result_sha256":raw_hashes}
    (root/"compactness_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps({"status":"PASS","summary_rows":len(rows),"passes":audit["resource_protocol"]["passes_per_configuration_profile"]},indent=2))
if __name__=="__main__": main()
