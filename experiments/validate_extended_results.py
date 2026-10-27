"""Recompute the retained external, component-ablation and low-FPR results."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def check_metrics(row, y, p, threshold=.5):
    pred = p >= threshold
    tp = int(((y == 1) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
    fp = int(((y == 0) & pred).sum()); tn = int(((y == 0) & ~pred).sum())
    for key, value in dict(tp=tp, fn=fn, fp=fp, tn=tn, n=len(y), benign=(y == 0).sum(), phishing=(y == 1).sum()).items():
        if key in row:
            assert int(row[key]) == value, (key, row)
    values = dict(recall=tp / (tp+fn) if tp+fn else None,
                  fpr=fp / (fp+tn) if fp+tn else None,
                  fpr_or_proxy_flag_rate=fp / (fp+tn) if fp+tn else None)
    if len(np.unique(y)) == 2:
        values.update(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                      f1=f1_score(y, pred), roc_auc=roc_auc_score(y, p),
                      average_precision=average_precision_score(y, p), mcc=matthews_corrcoef(y, pred))
    for key, value in values.items():
        if key in row and row[key] != '' and value is not None:
            assert abs(float(row[key])-value) < 1e-10, (key, row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args(); results = args.root / 'results'
    ext = read(results / 'phreshphish/predictions.csv.gz')
    y = np.array([int(r['label']) for r in ext])
    domain = np.array([r['eligible_domain_disjoint'].lower() in ('true', '1') for r in ext])
    assert len(ext) == 166227 and (y == 0).sum() == 89686 and domain.sum() == 141623
    names = ['frozen_568', 'MUDS', 'Roy', 'E2Phish', 'TabNet', 'Ebbu', 'original', 'source_nuisance_balanced', 'hostname_independent']
    probabilities = {n: np.array([float(r[n+'_probability']) for r in ext]) for n in names}
    for name, p in probabilities.items():
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        assert np.array_equal(p >= .5, [int(r[name+'_prediction']) for r in ext])
    external_count = 0
    for filename in ['baseline_summary.csv', 'operating_points.csv']:
        for row in read(results / 'phreshphish' / filename):
            mask = np.ones(len(y), dtype=bool) if row['population'] == 'url_disjoint' else domain
            name = row.get('model', row.get('condition'))
            check_metrics(row, y[mask], probabilities[name][mask], float(row['threshold']))
            external_count += 1
    validation = read(results / 'phreshphish/source_validation_probabilities.csv.gz')
    vy = np.array([int(r['label']) for r in validation])
    for row in read(results / 'phreshphish/thresholds.csv'):
        scores = np.array([float(r[row['condition']+'_probability']) for r in validation])
        if row['operating_point'] == 'fixed_0.5':
            expected = .5
        else:
            benign = np.sort(scores[vy == 0])[::-1]
            expected = np.nextafter(benign[math.floor(float(row['target_validation_fpr'])*len(benign))], np.inf)
        assert float(row['threshold']) == expected
    component = read(results / 'sdcs_components/all_per_seed_metrics.csv')
    assert len(component) == 40
    sets = {}
    for row in component:
        folder = results / 'sdcs_components/fits' / ('seed_'+row['seed']) / row['method']
        pred = read(folder / 'test_predictions.csv.gz')
        assert len(pred) == 11376
        yy = np.array([int(r['label']) for r in pred]); pp = np.array([float(r['probability']) for r in pred])
        check_metrics(row, yy, pp)
        chosen = read(folder / 'selected_features.csv')
        indices = {int(r['feature_index']) for r in chosen}
        assert len(indices) == len(chosen) == int(row['feature_count'])
        sets[row['seed'], row['method']] = indices
    for seed in map(str, range(42, 52)):
        assert len({len(s) for (ss, _), s in sets.items() if ss == seed}) == 1
    objective = read(results / 'low_fpr/objective_operating_points.csv')
    partition = np.load(results / 'objective_partitions.npz')
    for row in objective:
        path = results / 'objective_predictions' / (row['ratio']+'_seed'+row['seed']) / (row['method']+'_probabilities.npz')
        with np.load(path) as a:
            vy = partition['y_validation']; benign = np.sort(a['validation'][vy == 0])[::-1]
            if row['operating_point'] == 'fixed_0.5': expected = .5
            else: expected = np.nextafter(benign[math.floor(float(row['target_validation_fpr'])*len(benign))], np.inf)
            assert float(row['threshold']) == expected
            check_metrics(row, partition['y_test'], a['test'], expected)
    shift_rows = read(results / 'low_fpr/shift_operating_points.csv')
    cache = {}
    for row in shift_rows:
        if row['cohort'] not in cache:
            data = read(results / 'low_fpr/source_reference/evaluation_full' / (row['cohort']+'_paired_predictions.csv.gz'))
            cache[row['cohort']] = data
        data = cache[row['cohort']]
        yy = np.array([int(r['label']) for r in data])
        pp = np.array([float(r[row['condition']+'_probability']) for r in data])
        check_metrics(row, yy, pp, float(row['threshold']))
    assert external_count == 44 and len(objective) == 480 and len(shift_rows) == 60
    print(json.dumps(dict(status='EXTENDED_RESULTS_VERIFIED',external_metric_rows=external_count,
                          component_models=40,objective_operating_points=480,shift_operating_points=60), indent=2))


if __name__ == '__main__':
    main()
