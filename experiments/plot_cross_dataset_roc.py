#!/usr/bin/env python3
"""Generate the four-panel cross-dataset ROC figure from validated predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve


SCENARIOS = [
    ("PhishStorm_to_PhishFusion", "PhishStorm $\\rightarrow$ PhishFusion"),
    ("PhishStorm_to_Ebbu2017", "PhishStorm $\\rightarrow$ Ebbu2017"),
    ("Ebbu2017_to_PhishFusion", "Ebbu2017 $\\rightarrow$ PhishFusion"),
    ("Ebbu2017_to_PhishStorm", "Ebbu2017 $\\rightarrow$ PhishStorm"),
]

MODELS = [
    ("MUDS", 12, "test"),
    ("E2Phish", 13, "test"),
    ("Ebbu", 14, "test"),
    ("TabNet", 15, "test"),
    ("StealthPhisher", 16, "target"),
    ("CNN-Fusion", 17, "target"),
    ("DEPHIDES", 18, "target"),
    ("GramBeddings", 19, "target"),
    ("URLNet", 20, "urlnet"),
    ("LitePhish", 21, "litephish"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_prediction(root: Path, model: str, experiment: int, kind: str, scenario: str) -> Path:
    source = scenario.split("_to_", 1)[0]
    if scenario.endswith("_to_PhishFusion"):
        full_root = root / "cross_dataset/full_phishfusion_baselines"
        classical = {
            "MUDS": "MUDS_results.csv", "E2Phish": "E2Phish_results.csv",
            "Ebbu": "Ebbu_results.csv", "TabNet": "TabNet_results.csv",
        }
        if model in classical:
            return full_root / model / scenario / classical[model]
        if model in {"StealthPhisher", "CNN-Fusion", "DEPHIDES"}:
            return (root / "cross_dataset/full_phishfusion_deep_models" /
                    model / source / f"{source}_to_full_PhishFusion_predictions.csv")
        if model == "GramBeddings":
            return (full_root / model / scenario /
                    f"{scenario}_repository_full_predictions.csv")
        if model == "URLNet":
            return full_root / model / scenario / "test" / "URLNet_results.csv"
        if model == "LitePhish":
            return (root / "cross_dataset/LitePhish" /
                    f"{source}_source" / scenario / "LitePhish_predictions.csv")
    if model == "LitePhish":
        return (root / "cross_dataset/LitePhish" /
                f"{source}_source" / scenario / "LitePhish_predictions.csv")
    experiment_dirs = {
        12: "cross_dataset/MUDS",
        13: "cross_dataset/E2Phish",
        14: "cross_dataset/Ebbu",
        15: "cross_dataset/TabNet",
        16: "cross_dataset/StealthPhisher",
        17: "cross_dataset/CNN-Fusion",
        18: "cross_dataset/DEPHIDES",
        19: "cross_dataset/GramBeddings",
        20: "cross_dataset/URLNet",
        21: "cross_dataset/LitePhish",
    }
    base = root / experiment_dirs[experiment]
    if kind == "test":
        return base / scenario / f"{scenario}_test_predictions.csv"
    if kind == "target":
        return base / scenario / f"{scenario}_target_test_predictions.csv"
    if kind == "urlnet":
        return base / scenario / "test" / "URLNet_results.csv"
    if kind == "litephish":
        return base / f"{source}_source" / scenario / "LitePhish_predictions.csv"
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("output_stem", type=Path)
    args = parser.parse_args()
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)

    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
              "#D55E00", "#882255", "#332288", "#999933", "#000000"]
    styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)),
              (0, (1, 1)), (0, (5, 2, 1, 2)), (0, (2, 2)), "-"]
    curve_rows, sources = [], []
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.35), sharex=True, sharey=True,
                             layout="constrained")
    for panel, ((scenario, title), ax) in enumerate(zip(SCENARIOS, axes.flat), start=1):
        for idx, (model, experiment, kind) in enumerate(MODELS):
            path = resolve_prediction(args.results_root, model, experiment, kind, scenario)
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path, usecols=["phishing", "probability"])
            if frame[["phishing", "probability"]].isna().any().any():
                raise ValueError(f"missing label/probability in {path}")
            fpr, tpr, thresholds = roc_curve(frame["phishing"].astype(int), frame["probability"].astype(float),
                                             pos_label=1, drop_intermediate=True)
            score = auc(fpr, tpr)
            width = 2.2 if model == "LitePhish" else 1.15
            zorder = 5 if model == "LitePhish" else 2
            ax.plot(fpr, tpr, color=colors[idx], linestyle=styles[idx], linewidth=width,
                    label=model, zorder=zorder)
            curve_rows.extend({"scenario": scenario, "model": model, "fpr": float(x),
                               "tpr": float(y), "threshold": float(t), "auc": float(score)}
                              for x, y, t in zip(fpr, tpr, thresholds))
            sources.append({"scenario": scenario, "model": model, "path": str(path),
                            "sha256": sha256(path), "rows": int(len(frame)), "auc": float(score)})
        ax.plot([0, 1], [0, 1], color="#777777", linestyle=(0, (2, 2)), linewidth=0.8,
                label="Chance" if panel == 1 else None, zorder=1)
        ax.set_title(f"({chr(96 + panel)}) {title}", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#D9D9D9", linewidth=0.45)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("True-positive rate")
    for ax in axes[-1, :]:
        ax.set_xlabel("False-positive rate")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False,
               handlelength=3.0, columnspacing=1.1)

    pdf_path = args.output_stem.with_suffix(".pdf")
    png_path = args.output_stem.with_suffix(".png")
    csv_path = args.output_stem.with_name(args.output_stem.name + "_source_data.csv.gz")
    manifest_path = args.output_stem.with_name(args.output_stem.name + "_manifest.json")
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, dpi=300, facecolor="white")
    plt.close(fig)
    pd.DataFrame(curve_rows).to_csv(csv_path, index=False, compression="gzip")
    manifest = {
        "figure": "Cross-dataset ROC curves",
        "positive_class": {"value": 1, "name": "phishing"},
        "transformation": "sklearn.metrics.roc_curve with drop_intermediate=True; no smoothing",
        "axes": {"x": "false-positive rate", "y": "true-positive rate", "limits": [0, 1]},
        "styles": {
            "colors": {model[0]: colors[index] for index, model in enumerate(MODELS)},
            "line_styles": {model[0]: str(styles[index]) for index, model in enumerate(MODELS)},
            "redundant_encoding": "model identity is encoded by both color and line style; LitePhish also uses a thicker line",
        },
        "layout_inches": [7.16, 5.35], "outputs": [str(pdf_path), str(png_path), str(csv_path)],
        "sources": sources,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"pdf": str(pdf_path), "png": str(png_path),
                      "source_data": str(csv_path), "curves": len(sources)}, indent=2))


if __name__ == "__main__":
    main()
