# Validation-selected low-FPR operating points

The operating-point analysis covers 480 objective settings and 60 controlled-shift
evaluations. Thresholds use source-validation benign observations only, with
prespecified FPR budgets of 1%, 0.5% and 0.1%, alongside the fixed 0.5 threshold.
Training imbalance changes the training population while validation and test
prevalence remain fixed. Every assigned objective, seed and operating point is
retained. A validation budget is not a guarantee of the external false-positive
rate. Positive-only temporal and homepage-proxy populations retain their distinct
metric definitions.

Run `python -m experiments.low_fpr_operating_points --help` for the required
objective-probability and distribution-shift inputs. The released CSV tables
retain all 540 held-out confusion matrices and their validation thresholds.
