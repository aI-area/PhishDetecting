#!/usr/bin/env python3
"""Independently check paired rows, probabilities, calibration and metric records."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score, precision_recall_curve, roc_curve)
from sklearn.preprocessing import StandardScaler


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('output', type=Path)
    ap.add_argument('legacy', type=Path)
    args = ap.parse_args()
    out = args.output
    if (out/'verification.json').exists():
        raise RuntimeError('Verification already exists; refusing overwrite')
    config = json.loads((out/'configuration.json').read_text())
    complete = json.loads((out/'complete.json').read_text())
    assert complete['models'] == (12 if config['qualification'] else 120)
    assert sha(out/'runner_snapshot.py') == config['runner_sha256']
    assert sha(args.legacy/'compact_568_split_cache.joblib') == config['cache_sha256']
    cache = joblib.load(args.legacy/'compact_568_split_cache.joblib')
    legacy_audit = json.loads((args.legacy/'experiment_audit.json').read_text())
    partitions = np.load(out/'partitions.npz')
    for split in ('train','validation','test'):
        ix = partitions[split+'_indices']
        digest = hashlib.sha256(np.ascontiguousarray(ix).view(np.uint8)).hexdigest()
        assert digest == legacy_audit['split'][split+'_idx_sha256']
    assert set(partitions['train_indices']).isdisjoint(partitions['validation_indices'])
    assert set(partitions['train_indices']).isdisjoint(partitions['test_indices'])
    assert set(partitions['validation_indices']).isdisjoint(partitions['test_indices'])
    yt,yv,ye = [partitions['y_'+s] for s in ('train','validation','test')]
    summary = pd.read_csv(out/'metrics.csv')
    assert len(summary) == complete['models']*3
    benign,phishing = np.flatnonzero(yt==0),np.flatnonzero(yt==1)
    seen = 0
    max_error = 0.
    for denominator in (3,5,10):
        ratio = '1_to_'+str(denominator)
        for seed in config['seeds']:
            task = out/(ratio+'_seed'+str(seed))
            assert json.loads((task/'complete.json').read_text())['models']==4
            rng = np.random.default_rng(seed)
            subset = np.concatenate([benign,rng.choice(phishing,len(benign)//denominator,replace=False)])
            rng.shuffle(subset)
            assert np.array_equal(np.load(task/'training_indices.npy'),partitions['train_indices'][subset])
            saved_scaler = joblib.load(task/'scaler.joblib')
            rebuilt = StandardScaler().fit(cache['x_train'][subset])
            np.testing.assert_array_equal(saved_scaler.mean_,rebuilt.mean_)
            np.testing.assert_array_equal(saved_scaler.scale_,rebuilt.scale_)
            for method in config['methods']:
                probs = np.load(task/(method+'_probabilities.npz'))
                model = lgb.Booster(model_file=str(task/(method+'_booster.txt')))
                for split,key in [('validation','validation'),('test','test')]:
                    raw = model.predict(rebuilt.transform(cache['x_'+split]),raw_score=True,num_threads=4)
                    error = float(np.max(np.abs(expit(raw)-probs[key])))
                    max_error = max(max_error,error)
                    assert error < 1e-10
                calibrator = joblib.load(task/(method+'_calibrator.joblib'))
                fitted = LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000).fit(
                    logit(np.clip(probs['validation'],1e-12,1-1e-12)).reshape(-1,1),yv)
                np.testing.assert_allclose(calibrator.coef_,fitted.coef_,rtol=0,atol=1e-12)
                np.testing.assert_allclose(calibrator.intercept_,fitted.intercept_,rtol=0,atol=1e-12)
                calibrated = fitted.predict_proba(logit(np.clip(probs['test'],1e-12,1-1e-12)).reshape(-1,1))[:,1]
                np.testing.assert_allclose(probs['test_sigmoid'],calibrated,rtol=0,atol=1e-12)
                records = json.loads((task/(method+'_metrics.json')).read_text())
                assert len(records)==3
                for r in records:
                    p = probs['test_sigmoid'] if r['variant']=='sigmoid_fixed' else probs['test']
                    pred = p>=r['threshold']
                    tn,fp,fn,tp = confusion_matrix(ye,pred,labels=[0,1]).ravel()
                    calculated = dict(accuracy=accuracy_score(ye,pred),precision=precision_score(ye,pred),
                        recall=recall_score(ye,pred),f1=f1_score(ye,pred),mcc=matthews_corrcoef(ye,pred),
                        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),roc_auc=roc_auc_score(ye,p),
                        average_precision=average_precision_score(ye,p),brier=brier_score_loss(ye,p),
                        log_loss=log_loss(ye,np.clip(p,1e-12,1-1e-12),labels=[0,1]))
                    counts,edges = np.histogram(p,bins=np.linspace(0,1,16))
                    calibration_error = 0.
                    if r['variant']!='raw_validation_f1':
                        bins = pd.read_csv(task/(method+'_'+r['variant']+'_reliability.csv'))
                        assert np.array_equal(counts,bins['count'].to_numpy())
                    for b,count in enumerate(counts):
                        mask = (p>=edges[b]) & ((p<edges[b+1]) if b<14 else (p<=edges[b+1]))
                        if count:
                            calibration_error += count/len(ye)*abs(p[mask].mean()-ye[mask].mean())
                    calculated['ece']=calibration_error
                    selected = summary[(summary.imbalance_ratio==ratio)&(summary.seed==seed)&
                        (summary.method==method)&(summary.variant==r['variant'])]
                    assert len(selected)==1
                    for k,v in calculated.items():
                        assert abs(float(v)-float(r[k])) < 1e-10, (task,method,k)
                        assert abs(float(v)-float(selected.iloc[0][k])) < 1e-10
                    if r['variant']!='raw_validation_f1':
                        curves = np.load(task/(method+'_'+r['variant']+'_curves.npz'))
                        expected = precision_recall_curve(ye,p)+roc_curve(ye,p)
                        for k,v in zip(('precision','recall','pr_thresholds','fpr','tpr','roc_thresholds'),expected):
                            np.testing.assert_array_equal(curves[k],v)
                    else:
                        precision,recall,thresholds = precision_recall_curve(yv,probs['validation'])
                        f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
                        best = np.flatnonzero(np.isclose(f1,np.max(f1),rtol=0,atol=1e-12))
                        threshold = thresholds[best[np.argmin(np.abs(thresholds[best]-.5))]]
                        assert r['threshold']==float(threshold)
                seen+=1
                print(str(task.name)+' '+method+' VERIFIED',flush=True)
    assert seen==complete['models']
    manifest = {str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
    report = dict(status='PROBABILITY_EVIDENCE_VERIFIED',models=seen,metrics=len(summary),
        max_booster_probability_error=max_error,legacy_max_metric_error=complete['legacy_max_metric_error'],
        verification_runner_sha256=sha(Path(__file__)),training_draws_reproduced=True,
        scalers_rebuilt=True,validation_only_calibration_rebuilt=True,metrics_recomputed=True)
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
