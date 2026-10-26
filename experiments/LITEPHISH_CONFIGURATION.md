# Experimental configuration

| Analysis | Representation / classifier settings |
|---|---|
| Primary source models | Character 2-/3-grams, `min_df=5`, `max_df=1.0`, variance threshold `1e-5`, MI/L1 equal weighting, 5,000 N-gram candidates; 86 handcrafted descriptors |
| N-gram L1 scorer | Logistic regression inverse-regularization `C=.5` |
| SDCS | 20 subsamples; fraction .905; inverse-regularization `C=.191`; stability threshold .705; MI weight .318; recurrence weight .767; Pearson cutoff .956; at most 1,000 features |
| Source-specific focal pairs | PhishStorm gamma .55 / alpha .90; Ebbu2017 .30 / .95; primary PhishFusion .55 / .95 |
| Matched selectors | Ten seeds 42–51; `max_df=.95`; shared 5,086 candidates; equal per-seed SDCS budget; downstream gamma .30 / alpha .95; 500 trees; seed 42 |
| Objective isolation | Frozen 568 features; training ratios 1:3, 1:5, 1:10; ten paired seeds 42–51; 500 trees; focal gamma .55 / alpha .95; alpha-only gamma 0 / alpha .95 |
| Objective validation/test | Original prevalence unchanged; test phishing prevalence 55.60%; calibration and threshold selection use validation only |
| Controlled shifts | Original/balanced 573 features; hostname-identity-reduced 487; gamma .55 / alpha .95; 500 trees; seed 42 |
| Raw/compact pair | Same fixed seed-42 split and classifier settings; 56,601 versus 568 features |
| Phone workload | 5,000 PhishTank URLs sampled with `RandomState(42)`; 2,705 benign and 2,295 phishing; threshold .5 |
| Runtime / energy | 60 fresh-process runtime trials; separate 30 discharge-only energy trials; one handset |

The focal-inspired update applies class and difficulty weights to the logistic gradient/curvature; it is not the exact derivative of the reference focal-loss expression. Logits initialize at zero. Probability clipping is `1e-15`; the Hessian floor is `1e-6`.

Feature counts depend on source data and fitting protocol. L1 `C` denotes the implementation's inverse-regularization parameter; no unsupported conversion to the generic mathematical penalty coefficient is assumed.
