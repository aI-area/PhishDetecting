# Statistical and figure analysis plan

Declared after verification of model fitting, before calculating the new paired
inferential summaries. This is an extension of an existing experiment, not a
prospective registration. No conditions or outcomes are excluded based on scores.

Unit: paired training seed (ten seeds), conditional on the fixed representation,
domain split and shared validation/test cohorts. No per-URL significance test.

Report all four methods, three training ratios and all three probability/threshold
variants. Means, SD, medians and pointwise Student-t 95% intervals describe seed
variability. For each paired comparison retain the ten differences, their mean,
pointwise Student-t interval (df9), median, SD and paired standardized effect dz.
Shapiro-Wilk diagnostics and Q-Q plots identify nonnormality; they do not select
which favorable test to report. Two-sided Wilcoxon-Pratt signed-rank is the
primary sensitivity-resistant paired test for all comparisons; also report
paired-t statistics and p-values consistently. All-zero differences give p=1.

Holm correction is applied separately, for each test, to four declared families:

1. Raw fixed-threshold: focal versus each of three comparators, at each ratio,
   for F1, AP, ROC-AUC, Brier, log loss and ECE (54 comparisons).
2. Sigmoid-calibrated fixed-threshold: the same 54 comparisons.
3. Raw validation-F1 threshold: focal versus three comparators at three ratios,
   F1 only (9 comparisons).
4. Within-method sigmoid minus raw: four methods, three ratios, for F1, Brier,
   log loss and ECE (48 comparisons).

All differences are left minus right; negative probability-error differences
favor the left method. Pointwise CIs are not multiplicity-adjusted; report them
alongside the family-adjusted p-values, not as familywise confidence bounds.
Non-significance is not equivalence. Keep exact values, not only rounded gains.

Figures are provisional research outputs, not inserted into the manuscript or
certified against journal export rules. Precision-recall plots show every raw
seed curve (not averaged probability predictions) in full-range panels and an
explicitly marked high-performance view. AP is the mean per-seed AP, not the
area of a mean curve. Calibration plots pool the ten seed-run predictions in
the predeclared 15 bins solely for descriptive visualization; they do not treat
repeated URLs as independent or determine reported uncertainty. Retain bin
counts and mean values for each seed. Raw and calibrated figures use identical
scales. Raw metric point plots show all ten seeds; Q-Q diagnostics show paired
differences. No smoothing or selectively chosen seed.
