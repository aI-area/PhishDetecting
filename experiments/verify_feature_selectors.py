"""Independently check row bindings, preprocessing and persisted predictions."""
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('base', type=Path)
parser.add_argument('folder', type=Path)
parser.add_argument('--dataset', type=Path, default=Path('your_path/PhishFusion.csv'))
args = parser.parse_args()
sys.path.insert(0, str(args.root.resolve()))
from feature_extraction import FeatureExtractor

class ErrorGuard(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            raise RuntimeError(record.getMessage())
logging.getLogger().addHandler(ErrorGuard())

folder = args.folder
anchor = json.loads((args.base/'partitions/verification.json').read_text())
audit = json.loads((folder/'audit.json').read_text())
manifest_path = args.base/'partitions/PhishFusion_matched_split.csv'
dataset_path = args.dataset
assert digest(dataset_path) == anchor['dataset_sha256'] == audit['dataset']['sha256']
assert digest(manifest_path) == anchor['manifest_sha256'] == audit['split_manifest']['sha256']
assert digest(folder/'runner_snapshot.py') == audit['runner_sha256']
source = (folder/'runner_snapshot.py').read_text()
assert 'processor.fit_vectorizer(urls.iloc[train_idx])' in source
assert 'processor.select_features(urls.iloc[train_idx], pd.Series(y_all[train_idx]))' in source
assert 'x_select, y_select = x_train[subset], y_train[subset]' in source
assert 'scaler.fit_transform(x_train[:, chosen])' in source
log_path = args.base/(folder.name+'.log')
log = log_path.read_text(errors='replace').lower()
assert not any(term in log for term in ('traceback', 'error in ', 'error fitting', 'transform error', 'failed'))
manifest = pd.read_csv(manifest_path)
data = pd.read_csv(dataset_path, encoding='latin1')
label = 'phishing' if 'phishing' in data else 'label'
assert np.array_equal(data[label], manifest.label)
idx = {part:np.flatnonzero(manifest.split.eq(part)) for part in ('train','validation','test')}
for part, rows in idx.items():
    assert hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest() == anchor['partitions'][part]['index_sha256']
assert np.array_equal(pd.read_csv(folder/'training_rows.csv').row_id, idx['train'])
assert np.array_equal(pd.read_csv(folder/'test_rows.csv').row_id, idx['test'])
processor = joblib.load(folder/'ngram_processor.joblib')
vocab = clone(processor.vectorizer).fit(data.url.iloc[idx['train']].astype(str))
assert vocab.vocabulary_ == processor.vectorizer.vocabulary_
print('TRAINING_ONLY_VOCABULARY_VERIFIED', flush=True)
del vocab
extractor = FeatureExtractor()
names = list(extractor.FEATURE_NAMES) + list(processor.get_feature_names())
metrics = pd.read_csv(folder/'per_seed_metrics.csv')
predictions = pd.read_csv(folder/'row_level_predictions.csv.gz')
methods = {'sdcs','mutual_information','anova_f','l1_logistic','lgbm_gain','mrmr_pearson'}
seeds = audit['design']['seeds']
assert len(metrics) == len(seeds)*6
assert len(predictions) == len(seeds)*6*len(idx['test'])
assert not metrics.duplicated(['seed','method']).any()
assert not predictions.duplicated(['seed','method','row_id']).any()
for seed in seeds:
    group = metrics[metrics.seed.eq(seed)]
    assert set(group.method) == methods and group.feature_count.nunique() == 1

# Reconstruct features without retaining a multi-gigabyte cache on disk.
def features(rows):
    urls = data.url.iloc[rows].astype(str)
    hc = np.asarray(Parallel(n_jobs=8, prefer='threads')(
        delayed(extractor.generate_features)(u) for u in urls), dtype=np.float32)
    ng = processor.transform(urls).toarray().astype(np.float32, copy=False)
    return np.hstack([hc,ng]).astype(np.float32, copy=False)

train = features(idx['train'])
print('TRAINING_FEATURES_RECONSTRUCTED', flush=True)
test = features(idx['test'])
print('TEST_FEATURES_RECONSTRUCTED', flush=True)
max_error = 0.0
reports = []
for record in metrics.to_dict('records'):
    seed, method = int(record['seed']), record['method']
    method_dir = folder/'selected_features'/('seed_'+str(seed))
    # Literal character N-grams such as 'n/a' are feature names, not missing values.
    selected = pd.read_csv(method_dir/(method+'.csv'), keep_default_na=False)
    chosen = selected.feature_index.to_numpy(dtype=int)
    assert len(chosen) == int(record['feature_count']) == len(np.unique(chosen))
    assert min(chosen) >= 0 and max(chosen) < len(names)
    assert list(selected.feature_name) == [names[i] for i in chosen]
    scaler = joblib.load(method_dir/(method+'_scaler.joblib'))
    rebuilt = StandardScaler().fit(train[:,chosen])
    assert np.allclose(scaler.mean_, rebuilt.mean_, rtol=0, atol=1e-10)
    assert np.allclose(scaler.var_, rebuilt.var_, rtol=0, atol=1e-10)
    assert np.array_equal(scaler.n_samples_seen_, rebuilt.n_samples_seen_)
    booster = lgb.Booster(model_file=str(method_dir/(method+'_booster.txt')))
    p_model = expit(booster.predict(scaler.transform(test[:,chosen]), raw_score=True, num_threads=8))
    frame = predictions[predictions.seed.eq(seed) & predictions.method.eq(method)]
    assert np.array_equal(frame.row_id, idx['test'])
    assert np.array_equal(frame.label, data[label].iloc[idx['test']])
    p = frame.probability.to_numpy()
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert np.array_equal(frame.prediction, (p>=.5).astype(int))
    error = float(np.max(np.abs(p_model-p)))
    max_error = max(max_error,error)
    assert error < 1e-12
    y, pred = frame.label, frame.prediction
    values = dict(accuracy=accuracy_score(y,pred), precision=precision_score(y,pred), recall=recall_score(y,pred), f1=f1_score(y,pred), roc_auc=roc_auc_score(y,p), mcc=matthews_corrcoef(y,pred))
    assert all(abs(value-record[key]) < 1e-12 for key,value in values.items())
    reports.append(dict(seed=seed, method=method, features=len(chosen), max_probability_error=error, **values))
    print('MODEL_VERIFIED',seed,method,flush=True)
result = dict(status='MATCHED_SELECTORS_VERIFIED', models=len(reports), prediction_rows=len(predictions), maximum_probability_error=max_error, dataset_sha256=digest(dataset_path), manifest_sha256=digest(manifest_path), verifier_sha256=digest(__file__), models_verified=reports, scope='fixed partition and training-only preprocessing; selector fitting scope checked in frozen runner')
(folder/'verification.json').write_text(json.dumps(result,indent=2))
files = {str(p.relative_to(folder)):digest(p) for p in folder.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
(folder/'SHA256SUMS.json').write_text(json.dumps(files,indent=2))
print(json.dumps(result,indent=2),flush=True)
