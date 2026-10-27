# PhreshPhish external evaluation

Run commands from the repository root. The pinned URL-only release is in
`dataset/PhreshPhish`; the upstream attribution and license are in its README.
Evaluation never visits the URLs or retrieves webpage content.

## Pretrained models

```bash
python -m experiments.phreshphish.evaluate --source your_path/PhishFusion.csv --output local_results/phreshphish
```

The default comparison includes LitePhish, MUDS, Roy et al., E2Phish, TabNet stacking
and Ebbu at threshold 0.5. Vocabulary, feature selection, preprocessing and model
parameters remain fixed. The source dataset hash and release counts are checked;
each inference path must reproduce 100 prespecified reference rows before full
evaluation. `--qualify-only` runs only this reference check.

The controlled representation comparison can be evaluated with:

```bash
python -m experiments.phreshphish.evaluate --source your_path/PhishFusion.csv --output local_results/phreshphish_representations --models frozen_568 original source_nuisance_balanced hostname_independent
```

Outputs include row-indexed probabilities and metrics for the same 166,227
URL-disjoint observations and the 141,623-observation source-domain-disjoint
subset. The representation comparison also uses thresholds selected from source
validation data at 1%, 0.5% and 0.1% benign-FPR budgets. External labels never
determine model fitting or operating-point selection. Full-precision source
validation probabilities and thresholds are retained in `results/phreshphish`.

## Result analysis

```bash
python -m experiments.validate_extended_results --root .
python -m experiments.phreshphish.bootstrap_fpr --results results/phreshphish --output local_results/phreshphish_bootstrap
```

The bootstrap resamples registered domains, retaining every URL within each
sampled domain. Its 2,000 paired replicates describe fixed-model variation within
the observed populations, rather than training variability or the prevalence of
consumer browsing. `results/phreshphish` contains all nine probability columns,
the complete baseline comparison, controlled operating points, cohort accounting
and domain-bootstrap intervals. The `Roy` column key denotes Roy et al.; it is
kept as a stable machine-readable model identifier.

## URL metadata acquisition

The release is already included. To acquire the same URL metadata independently,
use Python 3.12 and the optional DuckDB dependency:

```bash
pip install -r experiments/phreshphish/requirements-acquisition.txt
python -m experiments.phreshphish.acquire_urls --output your_path/PhreshPhish
```

The collector projects only metadata columns from the pinned Parquet shards.
DuckDB's official `httpfs` extension must be available. No HTML columns or live
dataset URLs are requested. The reference inference environment remains the
documented Python 3.9/scikit-learn/LightGBM environment.
