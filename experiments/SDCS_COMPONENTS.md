# SDCS component experiment

The four-condition study isolates relevance ranking, stability screening and
correlation pruning on the canonical PhishFusion partition. All conditions use
the same training-only candidate representation and the SDCS feature budget of
each seed. Downstream classifiers share all hyperparameters and training rows.

## Inputs

The component runner expects an existing verified selector reference beneath
`your_path/selector_reference/selectors_full`: its N-gram processor, selected
indices, per-seed predictions and `verification.json`. This is the output of
the matched-selector runner and verifier described in the workflow guide. The
canonical source assignments are supplied directly with `--manifest`.

```bash
python -m experiments.compare_sdcs_components --base local_results/sdcs_components --anchor your_path/selector_reference --manifest manifests/PhishFusion.csv --dataset your_path/PhishFusion.csv --output local_results/sdcs_components/fits --seeds 42 43 44 45 46 47 48 49 50 51 --jobs 8
python -m experiments.verify_sdcs_components --base local_results/sdcs_components --anchor your_path/selector_reference --dataset your_path/PhishFusion.csv --output local_results/sdcs_components/fits
python -m experiments.analyze_sdcs_components --base local_results/sdcs_components --output local_results/sdcs_components/summary
```

The first command constructs the shared cache and fits all 40 conditions. The
complete SDCS condition must reproduce the reference indices and probabilities.
Selectors and scalers fit source training observations only. Cached feature
construction is excluded from reported selection time; those times are not total
offline training costs. The retained 40-fit data are in `results/sdcs_components`.

To reanalyze the retained results without training:

```bash
python -m experiments.analyze_sdcs_components --base results/sdcs_components --output local_results/sdcs_component_summary
```
