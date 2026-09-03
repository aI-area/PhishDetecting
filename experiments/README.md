# LitePhish experiment workflows

This directory contains the workflows used for the domain-disjoint, cross-dataset, external, statistical, temporal, failure-case, compactness, resource, and CPU-energy analyses reported for LitePhish. All dataset tables are expected to contain `url` and binary `phishing` columns, where phishing is class 1. Scripts that accept a split manifest require the `row_id`, `label`, and `split` fields produced by `create_domain_split_manifest.py`.

The complete model and feature-selection settings are listed in [`LITEPHISH_CONFIGURATION.md`](LITEPHISH_CONFIGURATION.md). [`RESULTS_REPRODUCTION_MAP.md`](RESULTS_REPRODUCTION_MAP.md) maps each reported analysis to its computational outputs.

## Reproducibility invariants

- Fit vocabularies, scalers, selectors, resampling operations, and thresholds on source training data only.
- Keep effective-second-level domains disjoint across internal train, validation, and test partitions.
- Remove exact source-training/target intersections before cross-dataset evaluation while retaining the declared target cohort order.
- Report precision, recall, and F1 as binary class-1 metrics. Compute F1 as `2 * precision * recall / (precision + recall)`.
- Use seed 42 for the primary split unless a script explicitly evaluates the ten declared seeds: 11, 22, 33, 44, 55, 66, 77, 88, 99, and 110.
- Preserve generated provenance JSON/CSV files and SHA-256 values with every reported output.

## Workflow map

| Analysis | Entry points |
| --- | --- |
| Domain-disjoint manifests | `create_domain_split_manifest.py` |
| Common internal baseline training and metrics | `train_cnn_fusion_shared_split.py`, `train_deephides_shared_split.py`, `train_e2phish_shared_split.py`, `train_ebbu_shared_split.py`, `train_grambeddings_shared_split.py`, `train_muds_shared_split.py`, `train_stealth_shared_split.py`, `train_tabnet_shared_split.py`, `urlnet_prepare_shared_split.py`, `urlnet_train_shared_split.py`, `urlnet_test_shared_split.py`, `evaluate_litephish_source_test.py`, `evaluate_saved_predictions.py`, `aggregate_shared_split_metrics.py` |
| Cross-dataset cohorts and evaluation | `create_cross_dataset_transfer_cohort.py`, `infer_classical_full_target.py`, `train_cnn_fusion_cross_dataset.py`, `train_deephides_cross_dataset.py`, `train_stealthphisher_cross_dataset.py`, `train_grambeddings_transfer.py`, `train_litephish_cross_transfer.py`, `urlnet_materialize_transfer_target.py`, `aggregate_cross_transfer_metrics.py` |
| External prediction harmonization | `evaluate_saved_predictions.py`, `evaluate_grambeddings_predictions.py`, `urlnet_convert_predictions.py` |
| McNemar, paired bootstrap, DeLong, and Holm tests | `compare_model_predictions.py` |
| Ten-seed stability and LitePhish component analyses | `run_litephish_experiments.py` |
| Focal/class-imbalance isolation | `run_focal_imbalance_ablation.py`, `summarize_focal_comparison.py` |
| Temporal OpenPhish cohort | `prepare_temporal_generalization.py`, `evaluate_litephish_temporal.py` |
| Natural and transformed failure cases | `evaluate_litephish_failure_cases.py`, `analyze_litephish_fp_fn_cases.py` |
| Raw-versus-compact comparison | `train_litephish_compactness_pair.py`, `benchmark_litephish_compactness.py`, `summarize_litephish_compactness.py` |
| Common resource measurements | `benchmark_model_resources.py`, `benchmark_models_sequential_server.sh`, `summarize_model_resources.py` |
| CPU-package energy | `measure_cpu_energy.py`, `benchmark_energy_phase.py`, `validate_energy_telemetry.py` |

Use `python -m experiments.<module> --help` for the exact path and output arguments. The modules perform row, label, and hash checks where applicable and write provenance metadata alongside predictions or summaries.

## Environments

- Core LitePhish and classical baselines: Python 3.9 with the packages in the repository-level `requirements.txt`; additional baseline packages include `imbalanced-learn` and `pytorch-tabnet`.
- CNN-Fusion, DEPHIDES, StealthPhisher, and GramBeddings: Python 3.9 with a baseline-compatible TensorFlow/Keras environment. The shared and transfer runners record the active library versions in their provenance metadata.
- URLNet: Python 3.7.12, TensorFlow 1.15.0, NumPy 1.21.6, and tflearn 0.3.2. URLNet is evaluated with dual-branch `emb_mode=3`.

CPU-package energy measurement requires compatible local hardware telemetry access. The energy workflow records phase timestamps, power samples, randomized trial order, duration controls, and idle-adjusted estimates; it does not estimate energy from processor TDP.
