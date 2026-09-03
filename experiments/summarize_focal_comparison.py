#!/usr/bin/env python3
"""Paired seed-wise inference for the LitePhish focal imbalance ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro, t as student_t, ttest_rel, wilcoxon


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holm(values: list[float]) -> list[float]:
    count = len(values)
    order = np.argsort(values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("detailed_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.detailed_csv)
    expected = {"1_to_3", "1_to_5", "1_to_10"}
    if set(data.loc[data.imbalance_ratio != "natural", "imbalance_ratio"]) != expected:
        raise RuntimeError("unexpected imbalance ratios")

    results = []
    for ratio in sorted(expected):
        for strategy in ("fixed_0_5", "validation_f1_tuned"):
            focal = data[(data.imbalance_ratio == ratio) & (data.method == "focal") &
                         (data.threshold_strategy == strategy)].set_index("seed")["f1"]
            for baseline in ("standard", "class_weight_balanced"):
                other = data[(data.imbalance_ratio == ratio) & (data.method == baseline) &
                             (data.threshold_strategy == strategy)].set_index("seed")["f1"]
                joined = pd.concat([focal.rename("focal"), other.rename("baseline")], axis=1).dropna()
                if len(joined) != 10:
                    raise RuntimeError(f"expected 10 paired seeds for {ratio}/{strategy}/{baseline}")
                difference = (joined.focal - joined.baseline).to_numpy(dtype=float)
                n = len(difference); mean = float(difference.mean()); sd = float(difference.std(ddof=1))
                half = float(student_t.ppf(0.975, n - 1) * sd / math.sqrt(n))
                t_result = ttest_rel(joined.focal, joined.baseline)
                w_result = wilcoxon(difference, zero_method="pratt", alternative="two-sided")
                normal = shapiro(difference)
                results.append({
                    "imbalance_ratio": ratio, "threshold_strategy": strategy,
                    "comparison": f"focal_minus_{baseline}", "paired_seeds": n,
                    "focal_f1_mean": float(joined.focal.mean()),
                    "baseline_f1_mean": float(joined.baseline.mean()),
                    "mean_f1_difference": mean, "difference_sd": sd,
                    "difference_ci95_low": mean - half, "difference_ci95_high": mean + half,
                    "paired_cohens_dz": mean / sd if sd else float("inf"),
                    "shapiro_w": float(normal.statistic), "shapiro_p": float(normal.pvalue),
                    "paired_t": float(t_result.statistic), "paired_t_df": n - 1,
                    "paired_t_p_raw": float(t_result.pvalue),
                    "wilcoxon_w": float(w_result.statistic), "wilcoxon_p_raw": float(w_result.pvalue),
                })
    frame = pd.DataFrame(results)
    frame["paired_t_p_holm"] = holm(frame.paired_t_p_raw.tolist())
    frame["wilcoxon_p_holm"] = holm(frame.wilcoxon_p_raw.tolist())
    frame["normality_flag_p_lt_0_05"] = frame.shapiro_p < 0.05
    output = args.output_dir / "focal_paired_comparisons.csv"
    frame.to_csv(output, index=False)
    audit = {
        "status": "PASS", "analysis": "paired seed-wise F1 comparisons",
        "input": str(args.detailed_csv.resolve()), "input_sha256": sha256(args.detailed_csv),
        "independent_unit": "seeded phishing subsample within each fixed imbalance ratio",
        "pairing": "same ratio, seed, split, features, and test cohort",
        "primary_test": "two-sided paired t-test with Student-t 95% CI on mean paired difference",
        "assumption_check": "Shapiro-Wilk test on paired differences",
        "sensitivity_test": "two-sided Wilcoxon signed-rank test",
        "multiplicity": "Holm family-wise correction across all 12 planned focal-versus-baseline F1 comparisons",
        "comparisons": len(frame), "normality_flags": int(frame.normality_flag_p_lt_0_05.sum()),
        "output": str(output.resolve()), "output_sha256": sha256(output),
    }
    audit_path = args.output_dir / "paired_analysis_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
