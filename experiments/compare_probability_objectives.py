#!/usr/bin/env python3
"""Paired objective comparison with retained probability and calibration evidence."""
import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def arrsha(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def reliability(y, p):
    bins = np.minimum((p * 15).astype(int), 14)
    rows = []
    for b in range(15):
        mask = bins == b
        n = int(mask.sum())
        rows.append(dict(bin=b, lower=b/15, upper=(b+1)/15, count=n,
                         mean_probability=float(p[mask].mean()) if n else None,
                         observed_positive_rate=float(y[mask].mean()) if n else None))
    ece = sum(r['count']/len(y) * abs(r['mean_probability']-r['observed_positive_rate'])
              for r in rows if r['count'])
    return float(ece), rows


class FailOnModelError(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            raise RuntimeError(record.getMessage())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('legacy', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--dataset', type=Path, default=Path('your_path/PhishFusion.csv'))
    ap.add_argument('--qualification', action='store_true')
    args = ap.parse_args()
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite an existing evidence directory')
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.root))
    sys.path.insert(0, str(args.root / 'experiments'))
    from main import PhishingDetector
    from model import FocalLossLGBM
    from run_focal_imbalance_ablation import common_params, metrics, tuned_f1_threshold
    import model as model_module
    model_module.logging.getLogger(model_module.__name__).addHandler(FailOnModelError())
    audit = json.loads((args.legacy/'experiment_audit.json').read_text())
    dataset = args.dataset
    cachepath = args.legacy/'compact_568_split_cache.joblib'
    assert sha(dataset) == audit['dataset_sha256']
    assert sha(cachepath) == audit['features']['cache_sha256']
    cache = joblib.load(cachepath)
    detector = PhishingDetector()
    urls, labels = detector.load_data(str(dataset))
    indices = detector.split_data_indices(urls, labels)
    y = labels.to_numpy(dtype=int)
    ys = [y[ix] for ix in indices]
    for name, ix in zip(('train','validation','test'), indices):
        assert arrsha(ix) == audit['split'][name+'_idx_sha256']
        assert arrsha(ix) == cache[name+'_idx_sha256']
    assert cache['processor_sha256'] == audit['features']['processor_sha256']
    assert cache['selected_sha256'] == audit['features']['selected_indices_sha256']
    xs = [cache['x_'+name] for name in ('train','validation','test')]
    assert [x.shape for x in xs] == [(86098,568),(8278,568),(11376,568)]
    assert all(np.isfinite(x).all() for x in xs)
    selected = audit['focal_parameter_selection']['selected']
    assert selected == dict(gamma=.55, alpha=.95)
    previous = pd.read_csv(args.legacy/'focal_imbalance_detailed.csv')
    configuration = dict(status='RUNNING', qualification=args.qualification,
        runner_sha256=sha(Path(__file__)), model_source_sha256=sha(args.root/'model.py'),
        legacy_runner_sha256=sha(args.root/'experiments/run_focal_imbalance_ablation.py'),
        dataset_sha256=sha(dataset), cache_sha256=sha(cachepath),
        focal_parameters=selected, representation='fixed historical 568-column classifier ablation',
        train_rows=len(ys[0]), validation_rows=len(ys[1]), test_rows=len(ys[2]),
        validation_prevalence=float(ys[1].mean()), test_prevalence=float(ys[2].mean()),
        seeds=[42] if args.qualification else list(range(42,52)),
        methods=['standard','class_weight_balanced','alpha_only','focal'])
    write_json(args.output/'configuration.json', configuration)
    shutil.copy2(__file__, args.output/'runner_snapshot.py')
    np.savez_compressed(args.output/'partitions.npz', train_indices=indices[0],
        validation_indices=indices[1], test_indices=indices[2],
        y_train=ys[0], y_validation=ys[1], y_test=ys[2])
    benign = np.flatnonzero(ys[0] == 0)
    positive = np.flatnonzero(ys[0] == 1)
    all_rows = []
    comparison_errors = []
    for denominator in (3,5,10):
        ratio = '1_to_'+str(denominator)
        for seed in configuration['seeds']:
            if shutil.disk_usage(args.output).free < 2*1024**3:
                raise RuntimeError('Disk reserve below 2GiB')
            rng = np.random.default_rng(seed)
            subset = np.concatenate([benign, rng.choice(positive,len(benign)//denominator,replace=False)])
            rng.shuffle(subset)
            task = args.output/(ratio+'_seed'+str(seed))
            task.mkdir()
            np.save(task/'training_indices.npy', indices[0][subset])
            scaler = StandardScaler().fit(xs[0][subset])
            train, val, test = scaler.transform(xs[0][subset]), scaler.transform(xs[1]), scaler.transform(xs[2])
            joblib.dump(scaler, task/'scaler.joblib', compress=3)
            methods = list(configuration['methods'])
            np.random.default_rng(seed+denominator*1000).shuffle(methods)
            for method in methods:
                params = common_params(seed,500,4)
                if method in ('focal','alpha_only'):
                    model = FocalLossLGBM(alpha=.95, gamma=.55 if method=='focal' else 0., **params)
                else:
                    model = LGBMClassifier(objective='binary', **params,
                        **({'class_weight':'balanced'} if method=='class_weight_balanced' else {}))
                start = time.perf_counter()
                model.fit(train, ys[0][subset])
                pv, pt = model.predict_proba(val)[:,1], model.predict_proba(test)[:,1]
                elapsed = time.perf_counter()-start
                for p in (pv,pt):
                    assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.ptp(p)>0
                calibrator = LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000)
                calibrator.fit(logit(np.clip(pv,1e-12,1-1e-12)).reshape(-1,1),ys[1])
                assert calibrator.n_iter_[0] < 1000
                pc = calibrator.predict_proba(logit(np.clip(pt,1e-12,1-1e-12)).reshape(-1,1))[:,1]
                joblib.dump(calibrator,task/(method+'_calibrator.joblib'),compress=3)
                model.booster_.save_model(str(task/(method+'_booster.txt')))
                np.savez_compressed(task/(method+'_probabilities.npz'),validation=pv,test=pt,test_sigmoid=pc)
                raw_tuned = tuned_f1_threshold(ys[1],pv)
                records = []
                for variant,p,threshold in [('raw_fixed',pt,.5),('sigmoid_fixed',pc,.5),('raw_validation_f1',pt,raw_tuned)]:
                    ece,bins = reliability(ys[2],p)
                    record = dict(imbalance_ratio=ratio, seed=seed, method=method,variant=variant,
                        train_rows=len(subset),train_phishing=int(ys[0][subset].sum()),
                        fit_predict_seconds=elapsed, **metrics(ys[2],p,threshold),
                        brier=float(brier_score_loss(ys[2],p)),
                        log_loss=float(log_loss(ys[2],np.clip(p,1e-12,1-1e-12),labels=[0,1])),ece=ece)
                    records.append(record)
                    if variant != 'raw_validation_f1':
                        pd.DataFrame(bins).to_csv(task/(method+'_'+variant+'_reliability.csv'),index=False)
                        pr,re,th = precision_recall_curve(ys[2],p)
                        fpr,tpr,rth = roc_curve(ys[2],p)
                        np.savez_compressed(task/(method+'_'+variant+'_curves.npz'),
                            precision=pr,recall=re,pr_thresholds=th,fpr=fpr,tpr=tpr,roc_thresholds=rth)
                if method != 'alpha_only':
                    for variant,strategy in [('raw_fixed','fixed_0_5'),('raw_validation_f1','validation_f1_tuned')]:
                        old = previous[(previous.imbalance_ratio==ratio)&(previous.seed==seed)&
                            (previous.method==method)&(previous.threshold_strategy==strategy)]
                        assert len(old)==1
                        row = next(r for r in records if r['variant']==variant)
                        keys = ['f1','precision','recall','accuracy','roc_auc','average_precision','tn','fp','fn','tp']
                        errors = {k:abs(float(row[k])-float(old.iloc[0][k])) for k in keys}
                        comparison_errors.append(max(errors.values()))
                        if max(errors.values()) > 1e-10:
                            write_json(task/(method+'_legacy_mismatch.json'),errors)
                            raise RuntimeError('Legacy reproduction mismatch; evidence preserved')
                write_json(task/(method+'_metrics.json'), records)
                all_rows.extend(records)
                print(json.dumps(dict(ratio=ratio,seed=seed,method=method,status='SAVED',seconds=elapsed)),flush=True)
            write_json(task/'complete.json',dict(status='COMPLETE',models=4,training_rows_sha256=arrsha(indices[0][subset])))
    pd.DataFrame(all_rows).to_csv(args.output/'metrics.csv',index=False)
    assert len(all_rows)==(36 if args.qualification else 360)
    write_json(args.output/'complete.json',dict(status='QUALIFICATION_COMPLETE' if args.qualification else 'FULL_RUN_COMPLETE',
        models=len(all_rows)//3, legacy_max_metric_error=max(comparison_errors),
        note='Independent probability verification required before promotion or reporting'))


if __name__=='__main__':
    main()
