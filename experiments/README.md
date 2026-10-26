# Experiments

Run commands from the repository root. Replace `your_path` with your own location. Dataset paths are user-supplied; no server account or workstation path is required.

## Shared protocol

Fit vocabulary, feature selection, scalers, resampling rules, and calibration on source training/validation data only. Keep the reported test rows fixed. Precision, recall, and F1 refer to the phishing class. The exact PhishFusion assignments are in `manifests/PhishFusion.csv`; a newly generated split is not automatically the same partition.

## Saved-result checks

```bash
python -m experiments.validate_public_results --root .
python -m experiments.summarize_phone_tradeoffs --results results --output local_results/phone_tradeoffs
```

These commands inspect retained measurements and predictions, not new training or hardware trials.

## Roy manual-feature Random Forest

```bash
python -m experiments.run_roy_baseline --mode internal --source-dataset your_path/PhishFusion.csv --source-manifest manifests/PhishFusion.csv --output-dir local_results/roy --name PhishFusion --jobs 8
```

This is the all-manual-feature RF configuration from Table II of Roy et al. (2024), DOI `10.1109/TCE.2024.3404459`: 17 descriptors, 100 trees, seed 42, standard RF defaults for unspecified parameters, and an explicit shortener list. It does not combine those features with TF–IDF or reproduce every published variant. Transfer mode accepts separate target dataset/manifest arguments; see `--help`.

## Matched-budget selectors

```bash
python -m experiments.compare_feature_selectors your_path/PhishFusion.csv manifests/PhishFusion.csv local_results/selectors --pipeline-root . --seeds 42 43 44 45 46 47 48 49 50 51 --jobs 8
```

SDCS, MI, ANOVA-F, L1-logistic, LightGBM gain, and mRMR-Pearson use the same 5,086-column candidate pool, common 90.5% training subsamples, and equal per-seed budgets. This study uses `max_df=.95`, gamma .30, alpha .95, and 500 trees; these are intentionally distinct from the primary model settings. Every downstream fit uses the complete training partition and the same held-out 11,376 rows.

## Objective and calibration comparison

First build or supply the frozen 568-feature cache and paired baseline records using the existing focal runner:

```bash
python -m experiments.run_focal_imbalance_ablation your_path/PhishFusion.csv artifacts/litephish local_results/focal_reference --pipeline-root . --feature-jobs 8 --outer-workers 1 --model-threads 4
python -m experiments.compare_probability_objectives . local_results/focal_reference local_results/probability/full --dataset your_path/PhishFusion.csv
python -m experiments.verify_probability_objectives local_results/probability/full local_results/focal_reference
```

The second command checks its standard, balanced, and focal metrics against the paired reference and adds alpha-only weighting. It stops on a hash, split, or metric mismatch. Do not remove a check to force agreement. The reported experiment uses the fixed historical representation; refitting a selector defines a different experiment. Four objectives × three training ratios × ten seeds produce 120 models and 360 metric rows. Sigmoid calibration fits validation only. For inferential summaries, copy `experiments/ANALYSIS_PLAN.md` to `local_results/probability/ANALYSIS_PLAN.md` before running the optional analysis module on that base directory.

## Source-only distribution-shift comparison

```bash
python -m experiments.fit_shift_representations local_results/shift/full_source --pipeline-root . --dataset your_path/PhishFusion.csv --full
python -m experiments.verify_shift_representations local_results/shift/full_source --runner experiments/fit_shift_representations.py --pipeline-root . --dataset your_path/PhishFusion.csv
python -m experiments.audit_shift_datasets local_results/shift/audit_v2 --source your_path/PhishFusion.csv --snapshot your_path/snapshot.csv --phishtank your_path/Phishtank/PhishTank.csv
python -m experiments.evaluate_shift_conditions local_results/shift local_results/shift/evaluation --pipeline-root . --dataset your_path/PhishFusion.csv --reference-root your_path/reference_experiments
```

The eligibility stage requires the same annotated snapshot inputs as the reported study; substitute datasets are new evaluations, not reproductions. The evaluator also expects the temporal and DNS-proxy reference cohorts beneath the supplied reference root. It does not retrain on targets. All three conditions and common denominators are retained. See the supplement for nuisance strata, clipped weights, population overlap, and inherited-label limitations.

## Android runtime and battery analysis

[`android/README.md`](android/README.md) describes the portable inference runtime, frozen model bundles, native-library requirements, and numerical qualification. Runtime and battery-side measurements are separate protocols. Energy analysis is offline by default:

```bash
python -m experiments.analyze_phone_energy your_path/phone_energy
```

That command requires an archived measurement package with its original schedule, protocol, shell telemetry, and app workload events. It does not estimate energy from TDP or from a charging-phone snapshot. Fresh discharge measurements require physical operator control, a continuity pilot, known signed-current polarity, continuous boot-time telemetry, and a safe battery/temperature limit.

## Existing comparison runners

Shared-split and cross-dataset runners remain available for CNN-Fusion, DEPHIDES, E2Phish, Ebbu, GramBeddings, MUDS, StealthPhisher, TabNet stacking, URLNet, and LitePhish. Use each module's `--help` for its input/output arguments. The compactness, temporal, natural-failure, desktop-resource, and CPU-energy runners retain their separate protocols.

## Environments

- Core classical runs: Python 3.9.20, NumPy 2.0.2, pandas 2.2.3, scikit-learn 1.6.1, SciPy 1.13.1, LightGBM 4.5.0.
- Neural baselines: a compatible TensorFlow/Keras environment specific to each original architecture.
- URLNet: Python 3.7.12, TensorFlow 1.15.0, NumPy 1.21.6, tflearn 0.3.2, dual branch `emb_mode=3`.
- Android: Termux Python 3.14.6, NumPy 2.4.4, native LightGBM 4.5.0 without OpenMP, and compiled ensemble kernels. These phone packages differ from the server environment; cross-platform timings are not attributable to hardware alone.
