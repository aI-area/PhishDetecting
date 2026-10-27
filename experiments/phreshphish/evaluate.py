"""Evaluate pretrained models on common, source-overlap-filtered URL cohorts."""
import argparse
import hashlib
import html
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
import tldextract
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
from .inference import ROOT, MODELS, predictor, digest

SOURCE_SHA256 = 'c8af8da76bb6a521fcb7515eab802fec450dd10eafdfb49593096dcc07f1de8c'


def cohort(metadata, source):
    suffix = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)
    def parse(value):
        try:
            value = html.unescape(str(value).strip())
            p = urlsplit(value if '://' in value else '//' + value)
            host = (p.hostname or '').lower().rstrip('.')
            ext = suffix(host)
            domain = getattr(ext, 'top_domain_under_public_suffix', None) or getattr(ext, 'registered_domain', '') or host
            normalized = urlunsplit(('', host + ((':' + str(p.port)) if p.port else ''), p.path or '/', p.query, ''))
            return normalized, domain, p.scheme in ('http', 'https') and bool(host), bool(p.path.strip('/') or p.query)
        except (ValueError, TypeError):
            return '', '', False, False
    source_urls = set(pd.read_csv(source).url.astype(str)) | set(pd.read_csv(source, encoding='latin1').url.astype(str))
    source_keys = [parse(u) for u in source_urls]
    source_normalized = {v[0] for v in source_keys if v[0]}
    source_domains = {v[1] for v in source_keys if v[1]}
    data = pd.read_csv(metadata, keep_default_na=False)
    assert len(data) == 168060
    assert all(hashlib.sha256(u.encode()).hexdigest() == h for u, h in zip(data.url, data.url_sha256))
    data.insert(0, 'source_row_id', np.arange(len(data)))
    data['label'] = data.source_label.map({'benign': 0, 'phish': 1})
    assert data.label.value_counts().to_dict() == {0: 91260, 1: 76800}
    parsed = [parse(u) for u in data.url]
    data['normalized_url'] = [v[0] for v in parsed]
    data['registered_domain'] = [v[1] for v in parsed]
    data['valid_http'] = [v[2] for v in parsed]
    data['path_or_query'] = [v[3] for v in parsed]
    conflicts = set(data.groupby('normalized_url').label.nunique().loc[lambda s: s > 1].index)
    eligible = data.valid_http & data.normalized_url.ne('') & data.registered_domain.ne('')
    eligible &= ~data.normalized_url.isin(conflicts) & ~data.normalized_url.duplicated(keep='first')
    eligible &= ~data.url.isin(source_urls) & ~data.normalized_url.isin(source_normalized)
    data = data.loc[eligible].copy().reset_index(drop=True)
    data['eligible_domain_disjoint'] = ~data.registered_domain.isin(source_domains)
    assert len(data) == 166227 and data.label.value_counts().to_dict() == {0: 89686, 1: 76541}
    assert data.eligible_domain_disjoint.sum() == 141623
    return data


def metrics(y, scores, threshold):
    p = scores >= threshold
    tn = int(((y == 0) & ~p).sum()); fp = int(((y == 0) & p).sum())
    fn = int(((y == 1) & ~p).sum()); tp = int(((y == 1) & p).sum())
    return dict(n=len(y), benign=int((y == 0).sum()), phishing=int((y == 1).sum()), tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=accuracy_score(y, p), precision=precision_score(y, p, zero_division=0), recall=recall_score(y, p),
                f1=f1_score(y, p), roc_auc=roc_auc_score(y, scores), average_precision=average_precision_score(y, scores),
                mcc=matthews_corrcoef(y, p), fpr=fp / (fp + tn))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='your_path/PhishFusion.csv')
    p.add_argument('--metadata', type=Path, default=ROOT / 'dataset/PhreshPhish/test_url_metadata.csv.gz')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--models', nargs='+', choices=MODELS, default=MODELS[:6])
    p.add_argument('--batch-size', type=int, default=500)
    p.add_argument('--qualify-only', action='store_true', help='Reproduce 100 prespecified reference rows per model')
    a = p.parse_args(); assert a.batch_size > 0
    assert digest(a.source) == SOURCE_SHA256
    a.output.mkdir(exist_ok=False, parents=True)
    data = cohort(a.metadata, a.source)
    reference = pd.read_csv(ROOT / 'results/phreshphish/predictions.csv.gz', float_precision='round_trip', keep_default_na=False)
    assert np.array_equal(data.source_row_id, reference.source_row_id)
    assert np.array_equal(data.label, reference.label)
    assert np.array_equal(data.eligible_domain_disjoint, reference.eligible_domain_disjoint)
    sample = np.random.RandomState(20261026).choice(len(data), 100, replace=False)
    checks = {}; models = {}
    for name in a.models:
        model = predictor(name); models[name] = model
        scores = model.predict(data.url.iloc[sample].tolist())
        error = float(np.max(np.abs(scores - reference[name + '_probability'].to_numpy()[sample])))
        assert error < 1e-10, (name, error)
        checks[name] = error
        print('REFERENCE_MATCH', name, error, flush=True)
    (a.output / 'qualification.json').write_text(json.dumps(dict(rows_per_model=100, maximum_probability_errors=checks), indent=2) + '\n')
    if a.qualify_only:
        return
    for name, model in models.items():
        values = []
        for start in range(0, len(data), a.batch_size):
            scores = np.asarray(model.predict(data.url.iloc[start:start+a.batch_size].tolist()))
            assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
            values.extend(scores)
            if start % 10000 == 0:
                print('PREDICT', name, start, len(data), flush=True)
        data[name + '_probability'] = values
        data[name + '_prediction'] = (np.asarray(values) >= .5).astype(int)
    summary = []
    thresholds = pd.read_csv(ROOT / 'results/phreshphish/thresholds.csv', float_precision='round_trip')
    for population in ['url_disjoint', 'domain_disjoint']:
        subset = data if population == 'url_disjoint' else data.loc[data.eligible_domain_disjoint]
        for name in a.models:
            points = [('fixed_0.5', .5)]
            points += [(r.operating_point, r.threshold) for r in thresholds.itertuples() if r.condition == name and r.operating_point != 'fixed_0.5']
            for rule, threshold in points:
                summary.append(dict(population=population, model=name, operating_point=rule, threshold=threshold,
                                    **metrics(subset.label.to_numpy(), subset[name + '_probability'].to_numpy(), threshold)))
    columns = ['source_row_id', 'label', 'registered_domain', 'eligible_domain_disjoint', 'path_or_query']
    columns += [n + '_' + k for n in a.models for k in ['probability', 'prediction']]
    data[columns].to_csv(a.output / 'predictions.csv.gz', index=False, compression='gzip')
    pd.DataFrame(summary).to_csv(a.output / 'summary.csv', index=False)
    (a.output / 'complete.json').write_text(json.dumps(dict(models=a.models, rows=len(data), source_sha256=SOURCE_SHA256, target_fitting=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
