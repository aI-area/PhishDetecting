#!/usr/bin/env python3
"""Common-protocol inference benchmark for raw and compact LitePhish."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse


def cgroup_v2_limits():
    result = {"detected": False, "cpu_quota_cores": None, "memory_max_bytes": None}
    try:
        membership = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        unified = next(line.split(":", 2)[2] for line in membership if line.startswith("0::"))
        root = Path("/sys/fs/cgroup") / unified.lstrip("/")
        quota, period = (root / "cpu.max").read_text(encoding="utf-8").strip().split()
        memory = (root / "memory.max").read_text(encoding="utf-8").strip()
        return {"detected": True, "cpu_quota_cores": None if quota == "max" else float(quota) / float(period),
                "memory_max_bytes": None if memory == "max" else int(memory)}
    except (OSError, StopIteration, ValueError):
        return result


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(x):
    return {"mean": float(statistics.mean(x)), "median": float(statistics.median(x)),
            "standard_deviation": float(statistics.stdev(x)) if len(x) > 1 else 0.0,
            "minimum": float(min(x)), "maximum": float(max(x))}


class PeakRSS:
    def __init__(self):
        self.p = psutil.Process(); self.peak = 0; self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
    def rss(self):
        total = self.p.memory_info().rss
        for child in self.p.children(recursive=True):
            try: total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied): pass
        return total
    def _run(self):
        while not self.stop_event.is_set(): self.peak=max(self.peak,self.rss()); self.stop_event.wait(.005)
    def start(self): self.peak=self.rss(); self.thread.start()
    def stop(self): self.stop_event.set(); self.thread.join(); return max(self.peak,self.rss())


def main():
    p=argparse.ArgumentParser(); p.add_argument("configuration",choices=["raw_56601","compact_568"])
    p.add_argument("artifact_root",type=Path); p.add_argument("dataset",type=Path); p.add_argument("output",type=Path)
    p.add_argument("--pipeline-root",type=Path,required=True); p.add_argument("--sample-size",type=int,required=True)
    p.add_argument("--batch-size",type=int,required=True); p.add_argument("--repeats",type=int,required=True)
    p.add_argument("--warmup-batches",type=int,required=True); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--profile-name",required=True); args=p.parse_args()
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[key]="1"
    sys.path.insert(0,str(args.pipeline_root)); from feature_extraction import FeatureExtractor
    monitor=PeakRSS(); monitor.start(); process=psutil.Process(); rss0=process.memory_info().rss
    model_path=args.artifact_root/args.configuration/"model.joblib"; scaler_path=args.artifact_root/args.configuration/"scaler.joblib"
    processor_path=args.artifact_root/"ngram_processor.joblib"; selected_path=args.artifact_root/"selected_568.joblib"
    model=joblib.load(model_path); scaler=joblib.load(scaler_path); processor=joblib.load(processor_path)
    selected=np.asarray(joblib.load(selected_path),dtype=int); rss1=process.memory_info().rss
    frame=pd.read_csv(args.dataset,encoding="latin1"); col="url" if "url" in frame else "URL"
    urls_all=frame[col].astype(str).reset_index(drop=True); rng=np.random.RandomState(args.seed)
    ids=rng.choice(len(urls_all),args.sample_size,replace=False); urls=urls_all.iloc[ids].reset_index(drop=True)
    extractor=FeatureExtractor()
    def run(batch):
        t=time.perf_counter(); hc=np.asarray([extractor.generate_features(u) for u in batch],dtype=np.float32); a=time.perf_counter()-t
        t=time.perf_counter()
        if args.configuration=="raw_56601": ng=processor.vectorizer.transform(batch)
        else: ng=processor.transform(batch)
        b=time.perf_counter()-t
        t=time.perf_counter(); full=sparse.hstack([sparse.csr_matrix(hc),ng],format="csr",dtype=np.float32)
        prepared=scaler.transform(full if args.configuration=="raw_56601" else full[:,selected].toarray().astype(np.float32,copy=False)); c=time.perf_counter()-t
        t=time.perf_counter(); prob=model.predict_proba(prepared)[:,1]; d=time.perf_counter()-t
        if len(prob)!=len(batch): raise RuntimeError("prediction length mismatch")
        return a,b,c,d,a+b+c+d
    for i in range(args.warmup_batches): run(urls.iloc[(i*args.batch_size)%len(urls):][:args.batch_size])
    raw=[[] for _ in range(5)]; executions=0
    for _ in range(args.repeats):
        for start in range(0,len(urls),args.batch_size):
            batch=urls.iloc[start:start+args.batch_size]; values=run(batch); executions+=len(batch)
            for bucket,value in zip(raw,values): bucket.append(value/len(batch)*1000)
    peak=monitor.stop(); names=["handcrafted","ngram","numeric_preprocessing","model_prediction","end_to_end"]
    artifacts=[model_path,scaler_path,processor_path]+([selected_path] if args.configuration=="compact_568" else [])
    result={"status":"PASS","configuration":args.configuration,"profile":args.profile_name,
      "dataset":{"name":args.dataset.name,"rows":len(urls_all),"sha256":sha256(args.dataset),"sample_row_id_sha256":hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()},
      "protocol":{"unique_urls":args.sample_size,"batch_size":args.batch_size,"repeats":args.repeats,"warmup_batches_excluded":args.warmup_batches,"measured_executions":executions,"seed":args.seed},
      "runtime":{"cpu_affinity_count":len(process.cpu_affinity()),"accelerator_used":False,
                 "cgroup_v2_effective_limits":cgroup_v2_limits()},
      "features":{"raw":56601,"compact":568,"active":56601 if args.configuration.startswith("raw") else 568},
      "artifact_bytes":sum(x.stat().st_size for x in artifacts),"artifact_sha256":{x.name:sha256(x) for x in artifacts},
      "memory":{"rss_before_load_mib":rss0/2**20,"rss_after_load_mib":rss1/2**20,"peak_rss_mib":peak/2**20},
      "latency_ms_per_url":{n:describe(v) for n,v in zip(names,raw)},"throughput_urls_per_second":1000/statistics.mean(raw[-1])}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
