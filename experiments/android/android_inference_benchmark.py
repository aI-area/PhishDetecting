"""Frozen LitePhish inference on Android; verify equivalence before profiling."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python_packages"))
sys.path.insert(0, str(ROOT / "bundle"))
import tldextract
import feature_extraction
import utils


def telemetry():
    values = {}
    for name in ("status", "current_now", "current_avg", "voltage_now", "charge_counter", "temp", "capacity"):
        try:
            values[name] = (Path("/sys/class/power_supply/battery") / name).read_text().strip()
        except OSError as e:
            values[name] = str(e)
    try:
        values["memory"] = Path("/proc/self/status").read_text()
    except OSError:
        pass
    return values


class Runtime:
    def __init__(self):
        bundle = ROOT / "bundle"
        self.meta = json.loads((bundle / "metadata.json").read_text())
        extractor = tldextract.TLDExtract(cache_dir=str(ROOT / "suffix_cache"),
                    suffix_list_urls=((bundle / "public_suffix_list.dat").as_uri(),),
                    fallback_to_snapshot=False, include_psl_private_domains=False)
        extractor("example.com")
        feature_extraction.extract_tld_parts = extractor
        utils.extract_tld_parts = extractor
        self.extractor = feature_extraction.FeatureExtractor()
        self.ngram_index = {word: i for i, word in enumerate(self.meta["ngrams"])}
        self.selected = np.asarray(self.meta["selected"], dtype=np.int64)
        self.mean = np.asarray(self.meta["mean"], dtype=np.float64)
        self.scale = np.asarray(self.meta["scale"], dtype=np.float64)
        library = list((ROOT / "lightgbm-4.5.0").rglob("lib_lightgbm.so"))
        if not library:
            library = list((ROOT / "build").rglob("lib_lightgbm.so"))
        self.lib = ct.CDLL(str(library[0]))
        self.lib.LGBM_GetLastError.restype = ct.c_char_p
        self.lib.LGBM_BoosterCreateFromModelfile.argtypes = [ct.c_char_p, ct.POINTER(ct.c_int), ct.POINTER(ct.c_void_p)]
        self.lib.LGBM_BoosterPredictForMat.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int,
            ct.c_int32, ct.c_int32, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_char_p,
            ct.POINTER(ct.c_int64), ct.POINTER(ct.c_double)]
        self.handle = ct.c_void_p()
        niter = ct.c_int()
        self.check(self.lib.LGBM_BoosterCreateFromModelfile(str(bundle / "model.txt").encode(), ct.byref(niter), ct.byref(self.handle)))

    def check(self, code):
        if code != 0:
            raise RuntimeError(self.lib.LGBM_GetLastError().decode())

    def extract(self, urls):
        handcrafted = np.asarray([self.extractor.generate_features(u) for u in urls], dtype=np.float32)
        ngrams = np.zeros((len(urls), len(self.ngram_index)), dtype=np.float32)
        for row, url in enumerate(urls):
            text = url.lower() if self.meta["lowercase"] else url
            text = re.sub(r"\s\s+", " ", text)
            for n in range(self.meta["ngram_range"][0], self.meta["ngram_range"][1] + 1):
                for word, count in Counter(text[i:i+n] for i in range(len(text)-n+1)).items():
                    index = self.ngram_index.get(word)
                    if index is not None:
                        ngrams[row, index] = count
        return np.hstack([handcrafted, ngrams])

    def prepare(self, raw):
        x = raw[:, self.selected].astype(np.float32, copy=True)
        if self.meta["scaler_with_mean"]:
            x -= self.mean
        if self.meta["scaler_with_std"]:
            x /= self.scale
        return np.ascontiguousarray(x)

    def predict(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        logits = np.empty(len(x), dtype=np.float64)
        outlen = ct.c_int64()
        self.check(self.lib.LGBM_BoosterPredictForMat(self.handle, x.ctypes.data_as(ct.c_void_p),
            0, len(x), x.shape[1], 1, 1, 0, -1, b"num_threads=1",
            ct.byref(outlen), logits.ctypes.data_as(ct.POINTER(ct.c_double))))
        assert outlen.value == len(x)
        return logits, 1.0 / (1.0 + np.exp(-logits))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["validate", "benchmark"])
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--count", type=int, default=5000)
    p.add_argument("--trial-index", type=int)
    p.add_argument("--batch-size", type=int, choices=[1, 100])
    args = p.parse_args()
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    cohort = json.loads((ROOT / "bundle/cohort.json").read_text())
    urls = cohort["urls"][:args.count]
    if args.mode == "benchmark" and args.trial_index is None:
        validation = json.loads((output / "validation.json").read_text())
        assert validation["count"] == 5000 and validation["checks"]["identical_labels"]
        assert all(c["passed"] for c in validation["checks"].values() if isinstance(c, dict))
        order = [(repeat, batch) for repeat in range(args.repeat) for batch in (1, 100)]
        np.random.RandomState(20261019).shuffle(order)
        (output / "trial_order.json").write_text(json.dumps(order))
        for repeat, batch in order:
            subprocess.run([sys.executable, __file__, "benchmark", "--count", str(args.count),
                            "--trial-index", str(repeat), "--batch-size", str(batch)], check=True)
            time.sleep(10)
        (output / "benchmark_complete.json").write_text(json.dumps(dict(trials=len(order), count=len(urls))))
        return
    start = time.perf_counter()
    runtime = Runtime()
    load_s = time.perf_counter() - start
    if args.mode == "validate":
        references = np.load(ROOT / "bundle/reference.npz")
        raw = runtime.extract(urls)
        prepared = runtime.prepare(raw)
        logits, probs = runtime.predict(prepared)
        checks = {}
        for name, actual in (("selected_raw", raw[:, runtime.selected]), ("prepared", prepared),
                             ("logits", logits), ("probabilities", probs)):
            expected = references[name][:args.count]
            checks[name] = dict(max_absolute_error=float(np.max(np.abs(actual-expected))),
                                passed=bool(np.allclose(actual, expected, rtol=1e-7, atol=1e-7)))
        checks["identical_labels"] = bool(np.array_equal(probs >= 0.5, references["probabilities"][:args.count] >= 0.5))
        result = dict(count=len(urls), checks=checks, load_s=load_s, telemetry=telemetry(),
                      platform=platform.platform(), python=sys.version, numpy=np.__version__)
        (output / "validation.json").write_text(json.dumps(result, indent=2))
        np.savez_compressed(output / "android_predictions.npz", probabilities=probs, logits=logits)
        print(json.dumps(result, indent=2), flush=True)
        assert all(c["passed"] for c in checks.values() if isinstance(c, dict)) and checks["identical_labels"]
        return
    validation = json.loads((output / "validation.json").read_text())
    assert validation["count"] == 5000 and validation["checks"]["identical_labels"]
    assert all(c["passed"] for c in validation["checks"].values() if isinstance(c, dict))
    reference = np.load(ROOT / "bundle/reference.npz")["probabilities"][:args.count]
    runtime.predict(runtime.prepare(runtime.extract(urls[:20])))
    order = [(args.trial_index, args.batch_size)]
    for repeat, batch in order:
        path = output / f"trial_{repeat}_batch_{batch}.json"
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite {path}")
        before = telemetry()
        cpu_before = time.process_time()
        stages = np.zeros((len(urls), 4))
        predictions = []
        wall = time.perf_counter()
        for i in range(0, len(urls), batch):
            part = urls[i:i+batch]
            t0 = time.perf_counter()
            raw = runtime.extract(part)
            t1 = time.perf_counter()
            prepared = runtime.prepare(raw)
            t2 = time.perf_counter()
            _, probabilities = runtime.predict(prepared)
            t3 = time.perf_counter()
            stages[i:i+len(part), :] = np.asarray([t1-t0, t2-t1, t3-t2, t3-t0]) / len(part)
            predictions.extend(probabilities.tolist())
        duration = time.perf_counter()-wall
        cpu = time.process_time()-cpu_before
        predictions = np.asarray(predictions)
        assert np.allclose(predictions, reference, rtol=1e-7, atol=1e-7)
        record = dict(repeat=repeat, batch_size=batch, count=len(urls), load_s=load_s,
            duration_s=duration, throughput_urls_s=len(urls)/duration,
            mean_ms=dict(zip(("feature_extraction", "preprocessing", "inference", "end_to_end"), (stages.mean(axis=0)*1000).tolist())),
            p50_end_to_end_ms=float(np.percentile(stages[:, 3], 50)*1000),
            p95_end_to_end_ms=float(np.percentile(stages[:, 3], 95)*1000),
            cpu_process_s=cpu, cpu_utilization_one_core_percent=cpu/duration*100,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            prediction_max_absolute_error=float(np.max(np.abs(predictions-reference))),
            telemetry_before=before, telemetry_after=telemetry(),
            note="Native ARM runtime; USB connected. Battery-current readings are not whole-device energy measurements.")
        path.write_text(json.dumps(record, indent=2))
        np.savez_compressed(output / f"trial_{repeat}_batch_{batch}.npz", timings_s=stages, probabilities=predictions)
        print(json.dumps({k: record[k] for k in ("repeat", "batch_size", "duration_s", "mean_ms", "peak_rss_kib")}), flush=True)


if __name__ == "__main__":
    main()
