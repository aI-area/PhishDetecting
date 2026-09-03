#!/usr/bin/env python3
"""Compute paired significance tests for internal, transfer, and external evaluations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    aliases = {"label": "phishing", "y_true": "phishing", "y_pred": "prediction",
               "score": "probability", "prob": "probability"}
    d = d.rename(columns={k: v for k, v in aliases.items() if k in d and v not in d})
    required = {"url", "phishing", "prediction", "probability"}
    if not required.issubset(d.columns):
        raise ValueError(f"{path}: missing {sorted(required - set(d.columns))}")
    out = d[["url", "phishing", "prediction", "probability"]].copy()
    out["url"] = out["url"].astype(str)
    out["phishing"] = out["phishing"].astype(int)
    out["prediction"] = out["prediction"].astype(int)
    out["probability"] = out["probability"].astype(float)
    return out


def align(reference: pd.DataFrame, candidate: pd.DataFrame, path: Path) -> pd.DataFrame:
    if len(reference) == len(candidate) and reference["url"].equals(candidate["url"]):
        if not np.array_equal(reference.phishing, candidate.phishing):
            raise ValueError(f"{path}: labels differ in identical URL order")
        return candidate.reset_index(drop=True)
    if (len(reference) == len(candidate) and np.array_equal(reference.phishing, candidate.phishing)
            and float((reference["url"].to_numpy() == candidate["url"].to_numpy()).mean()) >= 0.99):
        # Some released evaluators normalize a handful of URL strings while preserving
        # the validated reference row order. Require both exact label order and >=99% URL identity.
        return candidate.reset_index(drop=True)
    left = reference.copy(); right = candidate.copy()
    left["occ"] = left.groupby("url").cumcount(); right["occ"] = right.groupby("url").cumcount()
    right = right.set_index(["url", "occ"])
    keys = pd.MultiIndex.from_frame(left[["url", "occ"]])
    if not keys.isin(right.index).all() or len(left) != len(right):
        raise ValueError(f"{path}: cannot establish one-to-one URL/occurrence alignment")
    result = right.loc[keys].reset_index()
    if not np.array_equal(left.phishing, result.phishing):
        raise ValueError(f"{path}: labels differ after occurrence-aware alignment")
    return result[["url", "phishing", "prediction", "probability"]]


def midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x); sorted_x = x[order]; n = len(x); ranks = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]: j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, float); out[order] = ranks
    return out


def paired_delong(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> tuple[float, float, float, float]:
    order = np.argsort(-y); m = int(y.sum()); n = len(y) - m
    predictions = np.vstack((score_a, score_b))[:, order]
    positive = predictions[:, :m]; negative = predictions[:, m:]
    tx = np.vstack([midrank(row) for row in positive])
    ty = np.vstack([midrank(row) for row in negative])
    tz = np.vstack([midrank(row) for row in predictions])
    auc = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    covariance = np.cov(v01) / m + np.cov(v10) / n
    variance = max(float(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]), 0.0)
    delta = float(auc[0] - auc[1]); se = math.sqrt(variance)
    p = 1.0 if se == 0 and delta == 0 else (0.0 if se == 0 else 2 * norm.sf(abs(delta / se)))
    return delta, delta - 1.959963984540054 * se, delta + 1.959963984540054 * se, float(p)


def paired_binary(y: np.ndarray, a: np.ndarray, b: np.ndarray, repetitions: int, seed: int) -> dict:
    category = y * 4 + a * 2 + b
    counts = np.bincount(category, minlength=8); n = len(y)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, counts / n, size=repetitions)
    indices = np.arange(8); yy = indices // 4; aa = (indices % 4) // 2; bb = indices % 2
    def metrics(matrix: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tp = matrix[:, (yy == 1) & (pred == 1)].sum(1); fp = matrix[:, (yy == 0) & (pred == 1)].sum(1)
        fn = matrix[:, (yy == 1) & (pred == 0)].sum(1)
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        acc = matrix[:, yy == pred].sum(1) / n
        return acc, f1
    acc_a, f1_a = metrics(draws, aa); acc_b, f1_b = metrics(draws, bb)
    result = {}
    for name, delta in (("accuracy", acc_a - acc_b), ("f1", f1_a - f1_b)):
        point = (accuracy_score(y, a) - accuracy_score(y, b)) if name == "accuracy" else (f1_score(y, a) - f1_score(y, b))
        p = min(1.0, 2 * min((np.sum(delta <= 0) + 1) / (repetitions + 1),
                             (np.sum(delta >= 0) + 1) / (repetitions + 1)))
        result[name] = (point, *np.quantile(delta, [0.025, 0.975]), p)
    correct_a = a == y; correct_b = b == y
    b_only = int((~correct_a & correct_b).sum()); a_only = int((correct_a & ~correct_b).sum())
    result["mcnemar"] = {"lite_only_correct": a_only, "baseline_only_correct": b_only,
                          "p": float(binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue) if a_only + b_only else 1.0}
    return result


def holm(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float); order = np.argsort(p); adjusted = np.empty(len(p)); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx])); adjusted[idx] = running
    return adjusted


def analyse(table: str, scenario: str, lite_path: Path, baselines: dict[str, Path], repetitions: int) -> tuple[list[dict], list[dict]]:
    lite = load(lite_path); y = lite.phishing.to_numpy(); lp = lite.prediction.to_numpy(); ls = lite.probability.to_numpy()
    rows = []; files = [{"table": table, "scenario": scenario, "model": "LitePhish", "path": str(lite_path), "sha256": sha256(lite_path), "n": len(lite)}]
    for index, (name, path) in enumerate(baselines.items()):
        base = align(lite, load(path), path); bp = base.prediction.to_numpy(); bs = base.probability.to_numpy()
        binary = paired_binary(y, lp, bp, repetitions, 42000 + index)
        auc = paired_delong(y, ls, bs)
        files.append({"table": table, "scenario": scenario, "model": name, "path": str(path), "sha256": sha256(path), "n": len(base)})
        for metric, values in (("accuracy", binary["accuracy"]), ("f1", binary["f1"]), ("roc_auc", auc)):
            primary_p = binary["mcnemar"]["p"] if metric == "accuracy" else values[3]
            rows.append({"table": table, "scenario": scenario, "baseline": name, "metric": metric,
                         "n": len(y), "lite_value": {"accuracy": accuracy_score(y, lp), "f1": f1_score(y, lp), "roc_auc": roc_auc_score(y, ls)}[metric],
                         "baseline_value": {"accuracy": accuracy_score(y, bp), "f1": f1_score(y, bp), "roc_auc": roc_auc_score(y, bs)}[metric],
                         "delta_lite_minus_baseline": values[0], "ci95_low": values[1], "ci95_high": values[2],
                         "p_raw": primary_p,
                         "test": ("exact McNemar; paired multinomial bootstrap CI" if metric == "accuracy" else
                                  ("paired multinomial bootstrap" if metric == "f1" else "paired DeLong")),
                         "bootstrap_p_raw": values[3] if metric in {"accuracy", "f1"} else np.nan,
                         "mcnemar_exact_p": binary["mcnemar"]["p"] if metric == "accuracy" else np.nan,
                         "lite_only_correct": binary["mcnemar"]["lite_only_correct"] if metric == "accuracy" else np.nan,
                         "baseline_only_correct": binary["mcnemar"]["baseline_only_correct"] if metric == "accuracy" else np.nan})
    return rows, files


def first(root: Path, pattern: str) -> Path:
    found = sorted(root.glob(pattern))
    if len(found) != 1: raise ValueError(f"expected one match for {pattern}, found {found}")
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("root", type=Path); parser.add_argument("output_dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000); parser.add_argument("--skip-internal", action="store_true")
    args = parser.parse_args(); root = args.root; args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []; files = []; missing = []
    internal_exp = {"MUDS":"03_muds", "E2Phish":"04_e2phish", "Ebbu":"05_ebbu", "TabNet":"06_tabnet",
                    "StealthPhisher":"07_stealth", "CNN-Fusion":"08_cnn_fusion", "DEPHIDES":"09_deephides", "GramBeddings":"10_grambeddings"}
    if not args.skip_internal:
        for ds in ("PhishStorm", "Ebbu2017", "PhishFusion"):
            if ds in {"PhishStorm", "Ebbu2017"}:
                lite = root / f"statistical_comparisons/internal_litephish_reuse/{ds}/LitePhish_predictions.csv"
            else:
                lite = root / f"statistical_comparisons/internal_litephish/{ds}_source/{ds}_to_{ds}/LitePhish_predictions.csv"
            bases = {name: first(root, f"experiment_{exp}_shared_split/{ds}/{ds}_test_predictions.csv") for name, exp in internal_exp.items()}
            bases["URLNet"] = root / f"experiment_urlnet_shared_split/{ds}/test/URLNet_results.csv"
            result = analyse("III", ds, lite, bases, args.bootstrap); rows += result[0]; files += result[1]

    scenarios = ("PhishStorm_to_PhishFusion", "PhishStorm_to_Ebbu2017", "Ebbu2017_to_PhishFusion", "Ebbu2017_to_PhishStorm")
    transfer_exp = {"MUDS":"12_muds", "E2Phish":"13_e2phish", "Ebbu":"14_ebbu", "TabNet":"15_tabnet", "StealthPhisher":"16_stealth",
                    "CNN-Fusion":"17_cnn_fusion", "DEPHIDES":"18_deephides", "GramBeddings":"19_grambeddings", "URLNet":"20_urlnet"}
    for scenario in scenarios:
        source, target = scenario.split("_to_")
        lite = root / f"cross_dataset/LitePhish/{source}_source/{scenario}/LitePhish_predictions.csv"
        bases = {}
        for name, exp in transfer_exp.items():
            if target == "PhishFusion":
                if name in {"MUDS","E2Phish","Ebbu","TabNet"}: path = first(root, f"cross_dataset/full_phishfusion_baselines/{name}/{scenario}/*_results.csv")
                elif name == "GramBeddings": path = first(root, f"cross_dataset/full_phishfusion_baselines/GramBeddings/{scenario}/*predictions.csv")
                elif name == "URLNet": path = root / f"cross_dataset/full_phishfusion_baselines/URLNet/{scenario}/test/URLNet_results.csv"
                else: path = first(root, f"cross_dataset/full_phishfusion_deep_models/{name}/{source}/*predictions.csv")
            else:
                if name == "URLNet": path = first(root, f"experiment_{exp}_cross_transfer/{scenario}/test/URLNet_results.csv")
                else: path = first(root, f"experiment_{exp}_cross_transfer/{scenario}/*test_predictions.csv")
            bases[name] = path
        result = analyse("IV", scenario, lite, bases, args.bootstrap); rows += result[0]; files += result[1]

    external_root = root.parents[1] / "baselines"
    lite = external_root / "ourfull_results.csv"
    external_names = {"E2Phish":"E2Phishfull_results.csv", "MUDS":"MUDSfull_results.csv", "StealthPhisher":"StealthPhisherfull_results.csv",
                      "TabNet":"TabNetfull_results.csv", "Ebbu":"Ebbu2017full_results.csv", "DEPHIDES":"DeepHidesfull_results.csv", "CNN-Fusion":"CNNfull_results.csv"}
    bases = {name: external_root / filename for name, filename in external_names.items()}
    generated_external = root / "statistical_comparisons/external_predictions"
    gram_path = generated_external / "GramBeddings/GramBeddings_results.csv"
    urlnet_path = generated_external / "URLNet/URLNet_results.csv"
    if gram_path.is_file(): bases["GramBeddings"] = gram_path
    else: missing.append({"table":"V", "scenario":"external_live", "model":"GramBeddings", "reason":"row-level prediction file not found"})
    if urlnet_path.is_file(): bases["URLNet"] = urlnet_path
    else: missing.append({"table":"V", "scenario":"external_live", "model":"URLNet", "reason":"row-level prediction file not found"})
    result = analyse("V", "external_live_93788", lite, bases, args.bootstrap); rows += result[0]; files += result[1]

    frame = pd.DataFrame(rows)
    frame["p_holm_within_table_metric"] = frame.groupby(["table", "metric"])["p_raw"].transform(holm)
    frame["significant_holm_0.05"] = frame["p_holm_within_table_metric"] < 0.05
    frame.to_csv(args.output_dir / "paired_significance_results.csv", index=False)
    pd.DataFrame(files).to_csv(args.output_dir / "prediction_file_audit.csv", index=False)
    pd.DataFrame(missing).to_csv(args.output_dir / "missing_prediction_evidence.csv", index=False)
    audit = {"status":"PASS_WITH_DECLARED_GAPS" if missing else "PASS", "bootstrap_repetitions":args.bootstrap,
             "positive_class":"phishing=1", "tests":{"accuracy":"exact McNemar plus paired multinomial bootstrap CI",
             "f1":"paired multinomial bootstrap", "roc_auc":"paired DeLong"},
             "multiplicity":"Holm correction within each table and metric family", "scope":"test-row uncertainty; not retraining-seed uncertainty",
             "comparisons":len(frame)//3, "missing":missing}
    (args.output_dir / "experiment_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
