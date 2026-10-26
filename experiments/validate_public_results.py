"""Check retained scientific results without training models or opening URLs."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .summarize_phone_tradeoffs import membership


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args()
    root = args.root; results = root/'results'; checks = 0
    expected = {'PhishFusion.csv': {0:52995, 1:52757}, 'ebbu2017.csv': {0:36400, 1:37175},
                'phishstorm.csv': {0:48009, 1:47902}, 'PhishTank.csv': {0:51453, 1:42335},
                'Phishtank/PhishTank.csv': {0:51453, 1:42045}, 'merged.csv': {0:130899, 1:125510}}
    for name, counts in expected.items():
        with (root/'dataset'/name).open(encoding='latin1', newline='') as stream:
            actual = Counter(int(r['phishing']) for r in csv.DictReader(stream))
        assert actual == counts, (name, actual)
        checks += 1
    partition = read(root/'manifests/PhishFusion.csv')
    assert Counter(r['split'] for r in partition) == {'train':86098, 'validation':8278, 'test':11376}
    checks += 1
    phone = read(results/'phone_predictions.csv')
    assert len(phone) == 5000 and Counter(r['label'] for r in phone) == {'0':2705, '1':2295}
    quality = [r for r in read(results/'phone_quality_resources.csv') if int(r['batch_size']) == 1]
    y = np.array([int(r['label']) for r in phone])
    for row in quality:
        p = np.array([float(r[row['model']]) for r in phone]); pred = p >= .5
        for key, value in [('accuracy',accuracy_score(y,pred)), ('precision',precision_score(y,pred)),
                           ('recall',recall_score(y,pred)), ('f1',f1_score(y,pred))]:
            assert abs(float(row[key])-value) < 1e-12, (row['model'],key)
            checks += 1
    for name in ('roy_phishfusion_predictions.csv', 'roy_external_predictions.csv'):
        data = read(results/name)
        label = 'phishing' if 'phishing' in data[0] else 'label'
        y = np.array([int(r[label]) for r in data]); p = np.array([float(r['probability']) for r in data])
        assert np.isfinite(p).all() and set(y) == {0,1}
        print(name, 'rows=',len(data), 'F1=',round(100*f1_score(y,p>=.5),5), 'AUC=',round(100*roc_auc_score(y,p),5))
        checks += 1
    detailed = read(results/'selector_per_seed.csv'); summary = read(results/'selector_summary.csv')
    assert len(detailed) == 60
    for row in summary:
        group = [r for r in detailed if r['method'] == row['method']]
        assert len(group) == 10
        for metric in ('accuracy','f1','roc_auc'):
            assert abs(np.mean([float(r[metric]) for r in group])-float(row[metric+'_mean'])) < 1e-12
            checks += 1
    for seed in range(42,52):
        assert len({r['feature_count'] for r in detailed if int(r['seed'])==seed}) == 1
        checks += 1
    stored = read(results/'pareto_membership.csv')
    for energy in ('gross_mj','incremental_mj'):
        actual = membership(quality, ('accuracy','fpr','latency_ms','peak_rss_mib',energy))
        scenario = 'batch1_accuracy_'+energy+'_5d'
        for row in actual:
            reference = next(r for r in stored if r['scenario']==scenario and r['model']==row['model'])
            assert row['non_dominated'] == (reference['non_dominated']=='True')
            checks += 1
    manifest = results/'SHA256SUMS.json'
    if manifest.is_file():
        for name, digest in json.loads(manifest.read_text()).items():
            assert hashlib.sha256((results/name).read_bytes()).hexdigest() == digest, name
            checks += 1
    print(json.dumps(dict(status='RETAINED_RESULTS_CHECKED', checks=checks,
                         scope='Saved results and counts only; no retraining or hardware measurement.'), indent=2))


if __name__ == '__main__':
    main()
