# SDCS component comparison

Four conditions over ten paired seeds (42–51): relevance only, relevance plus
stability, relevance plus correlation pruning, and full SDCS. Each condition
uses the same per-seed feature budget and the fixed 86,098/8,278/11,376-row
PhishFusion partition. All 40 fitted conditions and their predictions are retained.

`summary.csv` and `paired_comparisons.csv` report the complete descriptive and
paired results. `all_per_seed_metrics.csv` contains the individual fits.
`fits/seed_*/` retains selected indices/names, scores, models, scalers and held-out
probabilities. Feature-selection times exclude the shared cached representation
and therefore do not measure total raw-URL-to-trained-model cost.
