# Results reproduction map

This document maps the reported analyses to their corresponding computational outputs.

| Analysis | Computational outputs |
|---|---|
| Domain-disjoint internal comparison | `results/statistical_comparisons/paired_significance_results.csv` (Table III prediction-level comparisons) |
| Four cross-dataset transfers with binary phishing-class metrics | `results/cross_dataset/summary/cross_dataset_metrics_long.csv`; `cross_dataset_provenance.json` |
| External live-URL comparison and paired significance | `results/statistical_comparisons/paired_significance_results.csv` (Table V, 93,788 URLs) |
| Ten-run Student-t confidence intervals | `repeated_run_confidence_intervals.json` ($n=10$, $df=9$, $t=2.262157$) |
| Common model size, peak RAM, feature cost, and latency | `results/resource_benchmark/model_resources_summary.csv` |
| Raw 56,601-feature versus compact 568-feature resources | `results/litephish_compactness/compactness/compactness_summary.csv` |
| Temporal post-freeze phishing detection | `results/temporal/evaluation/LitePhish_temporal_metrics.json` |
| Focal/standard/class-weighted imbalance comparison | `results/focal_imbalance/focal_paired_comparisons.csv`; `focal_imbalance_summary.csv` |
| Natural failure strata and counterfactual URL transformations | `results/failure_cases/fp_fn_category_rates.csv`; `counterfactual_comparisons.csv` |
| Paired McNemar/bootstrap/DeLong inference | `results/statistical_comparisons/experiment_audit.json`; `paired_significance_results.csv` |
| Popular-DNS likely-benign proxy | `results/popular_dns/frozen_released/experiment_audit.json` |
| Duration-controlled CPU-package energy | `results/cpu_energy/measurement_duration_controlled/energy_summary.csv`; accompanying audit |

Claim boundary: resource evidence is limited to controlled x86-64 execution and CPU-package energy. It does not establish performance, thermal behavior, battery consumption, or whole-device energy on physical ARM, smartphone, Raspberry Pi, gateway, or microcontroller hardware.

## Lightweight-detector comparison

| Study | Methodological evidence |
|---|---|
| Roy et al. (IEEE TCE 2024) | Primary article: Table I (17 manual URL features), Fig. 1 and Table II (manual features plus TF--IDF; DT/LR/RF), methodology (420,464 URLs), and experimental setup (3:1 within-corpus split; DT/LR/RF/CNN). DOI: `10.1109/TCE.2024.3404459`. |
| Sahingoz et al. (2019) | Article record and reported dataset: NLP/word-vector/hybrid representations, seven classifiers, and 73,575-URL constructed corpus. DOI: `10.1016/j.eswa.2018.09.029`. |
| Bustio-Martinez et al. (2022) | Article record and accompanying dataset description: 46 URL features reduced to 9 using the joint IG/chi-squared/ReliefF selector; Random Forest; 52,000 Alexa/PhishTank URLs. DOI: `10.1016/j.ins.2022.04.059`. |
| MUDS (2024) | Primary article: engineered URL features, IG/correlation/PCA, tree-based stacking and CL K-means, ISCX-URL-2016 with 651,191 URLs, and held-out-class zero-day simulations. DOI: `10.3390/jtaer19040141`. |

Comparison rule: “cross-data” is marked yes only when a trained source model is evaluated on an independently collected target dataset. Random partitions, cross-validation, and held-out attack classes constructed from the same source corpus are not counted as independent cross-dataset tests.

## Consumer-oriented security context

| Study | Scope in the comparative analysis | Verified record |
|---|---|---|
| Kumar et al. (2024) | SHAP-based explainability in consumer-IoT authentication and intrusion detection; cited as contextual motivation, not a phishing baseline. | Crossref metadata and institutional publication record; DOI: `10.1109/TCE.2023.3320157`, vol. 70, no. 1, pp. 1145--1154. |
| Khan et al. (2025) | Distributed weighted boosting for imbalanced consumer-IoT cyberattack detection; cited as a complementary resource-aware security strategy. | Crossref metadata; DOI: `10.1109/TCE.2024.3499942`, vol. 71, no. 2, pp. 6340--6347. |
| B. G. Roy et al. (2026) | Lightweight stacked-ensemble and density-based anomaly detection in a zero-trust consumer-IoT architecture; cited as network-level context. | Crossref metadata; DOI: `10.1109/TCE.2025.3635619`, vol. 72, no. 1, pp. 2007--2015. |

Scope boundary: these three studies address authentication or network-level intrusion detection. They are treated as contextual studies rather than URL-phishing baselines, and their reported metrics are not directly compared with LitePhish.
