#!/usr/bin/env python3
"""Convert URLNet's legacy tab-separated output to the common prediction schema."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--urlnet-results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cohort = pd.read_csv(args.test_csv)[["url", "phishing"]].copy()
    legacy = pd.read_csv(args.urlnet_results, sep="\t")
    if len(cohort) != len(legacy):
        raise ValueError("URLNet output length does not match the test cohort")

    legacy_labels = legacy["label"].astype(int).replace({-1: 0})
    labels = cohort["phishing"].astype(int).reset_index(drop=True)
    if not legacy_labels.reset_index(drop=True).equals(labels):
        raise ValueError("URLNet output labels do not match test.csv order")

    output = cohort.reset_index(drop=True)
    output["phishing"] = labels
    output["prediction"] = legacy["predict"].astype(int).replace({-1: 0}).to_numpy()
    output["probability"] = legacy["score"].astype(float).to_numpy()
    if not output["probability"].between(0.0, 1.0).all():
        raise ValueError("URLNet positive-class score falls outside [0, 1]")

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print("Wrote {} predictions to {}".format(len(output), destination))


if __name__ == "__main__":
    main()
