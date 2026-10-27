# Results map

Table numbers refer to the 12-page main manuscript. The supporting methods and additional tables are in `Supplementary.tex`.

| Display | Analysis | Inputs and outputs |
|---|---|---|
| II | Dataset composition | Four datasets and the descriptive union; `results/dataset_composition.csv`, `results/union_flow.csv` |
| III | Internal detection | Shared domain-disjoint partitions; model-specific runners and phishing-class metrics. The Roy implementation is its 17-manual-feature Random Forest configuration, not the separate published TF–IDF configuration. |
| IV / Fig. 3 | Four source–target transfers | Source-trained classifiers; original target row order; `results/cross_dataset_roc_points.csv.gz` contains all 44 curves. |
| V | Six matched-budget selectors | Same 5,086 source-trained candidates and identical downstream model; ten selector seeds 42–51; common 11,376 test rows; `results/selector_summary.csv`, `selector_per_seed.csv`, `selector_predictions.csv.gz` |
| VI | Fixed-threshold imbalance | Ten paired seeds, fixed 568-column representation, three training ratios; `results/objective_per_seed.csv`, `objective_summary.csv`, `objective_predictions/`, `objective_partitions.npz` |
| VII(a) | External PhishTank detection | 93,788 PhishTank/Common Crawl URLs; `results/roy_external_predictions.csv`; external evaluators |
| VII(b) | External PhreshPhish detection | `dataset/PhreshPhish/`; `results/phreshphish/`; `experiments/phreshphish/` |
| VIII | Physical ARM smartphone | Six models; 60 fresh-process runtime trials and 30 separate battery trials; `results/phone_predictions.csv`, `phone_runtime_trials.csv`, `phone_energy_trials.csv`, `phone_energy_summary.csv`, `phone_quality_resources.csv` |
| IX | Distribution shift | Three source-trained conditions; common source test and eligible full-URL targets; `results/distribution_shift_summary.csv` |
| Fig. 5 | Detection–resource analysis | Unrounded common-phone quality and cost; `results/pareto_membership.csv`, `pareto_pairwise_dominance.csv` |
| X | Raw/compact pair | Seed 42, fixed source split and model settings; `results/compactness_resources.csv` |

## Supporting analyses

- Probability comparison: four objectives × three training ratios × ten seeds = 120 classifiers, with raw, validation-calibrated, and validation-threshold variants. `analyze_probability_objectives.py` reports effect sizes and the declared correction families. A ten-pair exact signed-rank test cannot reject at .05 in the 54-contrast Holm family; non-rejection is not equivalence.
- Representation shifts: original, hostname-identity-reduced, and source-nuisance-balanced conditions use source-only fitting. Their feature counts differ from the frozen 568-column objective/resource representation.
- Temporal/failure analyses: all-positive temporal recall and DNS-homepage flag rates are not mixed-class detection metrics. Dataset annotations are not independently verified browsing behavior.
- Resource measurements: online means batch size one. Raw-to-compact reduction lowers fitting time and storage, but increases online latency in that protocol. Desktop CPU-package energy is not phone battery-side energy.
- Android measurements: numerical ports preserve the frozen inference functions. Roy is a best-effort RF reproduction. TabNet denotes descriptor selection followed by stacking, not standalone neural TabNet inference. The five-model-plus-LitePhish runtime uses the same frozen 5,000-URL pool; duration-limited energy trials use actual completed-URL denominators.

## What can be checked directly

`validate_public_results.py` recomputes classification metrics, selector summaries, selected objective metrics, dataset counts, and mean-coordinate Pareto membership from retained records. It does not retrain models or recreate hardware measurements. Model refitting additionally requires the appropriate source datasets, fixed manifests, serialized representation, dependency versions, and environment. The workflow guide identifies those requirements instead of treating a saved-result check as a complete retraining run.

## Component and operating-point records

`results/sdcs_components` contains all 40 component fits, feature selections and
held-out probabilities. `results/low_fpr` contains all 480 objective operating
points and 60 shift evaluations, including their underlying source-validation
and held-out probabilities. `validate_extended_results.py` independently
recomputes these records and the 44 PhreshPhish metric rows.
