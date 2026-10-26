import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from run_roy_baseline import extract_features

base = Path(sys.argv[1])
manifest = pd.read_csv(base / 'partitions/PhishFusion_matched_split.csv')
anchor = json.loads((base / 'partitions/verification.json').read_text())
folder = base / 'roy_phishfusion'
record = json.loads((folder / 'metrics.json').read_text())
assert record['source_dataset_sha256'] == anchor['dataset_sha256']
assert record['source_manifest_sha256'] == anchor['manifest_sha256']
assert record['training_rows'] == 86098
model = joblib.load(folder / 'roy_random_forest.joblib')
report = {}
for split in ('validation', 'test'):
    frame = pd.read_csv(folder / (split + '_predictions.csv'))
    expected = manifest[manifest.split.eq(split)]
    assert np.array_equal(frame.row_id, expected.row_id)
    assert np.array_equal(frame.phishing, expected.label)
    assert hashlib.sha256(frame.row_id.to_numpy(dtype=np.int64).tobytes()).hexdigest() == anchor['partitions'][split]['index_sha256']
    p = frame.probability.to_numpy()
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert np.array_equal(frame.prediction, (p >= .5).astype(int))
    regenerated = model.predict_proba(extract_features(frame.url))[:, 1]
    error = float(np.max(np.abs(regenerated - p)))
    assert error < 1e-12
    y, pred = frame.phishing, frame.prediction
    values = dict(accuracy=accuracy_score(y,pred), precision=precision_score(y,pred), recall=recall_score(y,pred), f1=f1_score(y,pred), roc_auc=roc_auc_score(y,p))
    for name, value in values.items():
        assert abs(value - record['metrics'][split][name]) < 1e-12, name
    report[split] = dict(rows=len(frame), metrics=values, max_probability_error=error)
result = dict(status='MATCHED_ROY_PREDICTIONS_VERIFIED', checks=report)
(folder / 'verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
