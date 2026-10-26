"""Qualify source-only fits before evaluation; never choose conditions by scores."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('fit_dir',type=Path)
    p.add_argument('--pipeline-root',type=Path,default=Path('.'))
    p.add_argument('--dataset',type=Path,default=Path('your_path/PhishFusion.csv'))
    p.add_argument('--runner',type=Path,required=True)
    args=p.parse_args()
    out=args.fit_dir
    assert not (out/'verification.json').exists(), 'Do not replace evidence'
    root=args.pipeline_root.resolve(); sys.path.insert(0,str(root))
    from feature_extraction import FeatureExtractor
    from model import FocalLossLGBM
    config=json.loads((out/'configuration.json').read_text())
    complete=json.loads((out/'complete.json').read_text())
    assert complete['status'] in ('SOURCE_ONLY_PILOT_COMPLETE','FULL_SOURCE_TRAINING_COMPLETE')
    assert config['source_code_sha256']==sha(args.runner)
    for name,digest in config['training_pipeline_sources'].items(): assert sha(root/name)==digest
    source=args.dataset
    assert sha(source)==config['source_sha256']
    manifest=pd.read_csv(out/'reconstructed_source_split.csv')
    assert sha(out/'reconstructed_source_split.csv')==config['split_sha256']
    assert manifest.groupby('esld').split.nunique().max()==1
    testidx=manifest.loc[manifest.split.eq('test'),'row_id'].to_numpy(dtype=int)
    assert hashlib.sha256(np.ascontiguousarray(testidx).view(np.uint8)).hexdigest()=='b37ff11d8baf705026dbe833e65a9425c3d2822cabcebc5ed9ed300504082921'
    selected=pd.read_csv(out/'source_pilot_rows.csv')
    assert not selected.source_row_id.duplicated().any()
    assert set(selected.partition)=={'train','validation'}
    assert np.array_equal(manifest.iloc[selected.source_row_id].split.to_numpy(),selected.partition.to_numpy())
    train=selected.loc[selected.partition.eq('train'),'source_row_id'].to_numpy(dtype=int)
    validation=selected.loc[selected.partition.eq('validation'),'source_row_id'].to_numpy(dtype=int)
    assert len(train)==config['train_rows'] and len(validation)==config['validation_rows']
    assert set(train).isdisjoint(validation) and set(selected.source_row_id).isdisjoint(testidx)
    data=pd.read_csv(source)
    train_urls=data.url.iloc[train].astype(str).reset_index(drop=True)
    ytrain=data.phishing.iloc[train].to_numpy(dtype=int)
    spec=importlib.util.spec_from_file_location('audited_fit_runner',args.runner)
    runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    draws,rule=runner.balance_indices(train_urls.tolist(),ytrain)
    repeat,_=runner.balance_indices(train_urls.tolist(),ytrain)
    assert np.array_equal(draws,repeat)
    assert rule==json.loads((out/'balancing_rule.json').read_text())
    assert np.array_equal(np.bincount(ytrain),np.bincount(ytrain[draws]))
    metrics=pd.read_csv(out/'validation_summary.csv')
    conditions=['original','source_nuisance_balanced','hostname_independent']
    assert set(metrics.condition)==set(conditions) and len(metrics)==3
    extractor=FeatureExtractor()
    selected_names={}
    maxerr=0
    artifacts={}
    for condition in conditions:
        bundle=joblib.load(out/(condition+'_pipeline.joblib'))
        representation=bundle['representation']
        processor=bundle['processor']; keep=bundle['keep_handcrafted']; chosen=bundle['selected']
        assert len(chosen)==len(set(chosen))==len(bundle['selected_names'])
        assert len(chosen)==int(metrics.loc[metrics.condition.eq(condition),'selected_features'].iloc[0])
        assert np.isfinite(bundle['scaler'].mean_).all()
        transform=lambda u:u if representation=='original' else (urlsplit(u).path or '/')+('?' + urlsplit(u).query if urlsplit(u).query else '')
        train_text=train_urls.map(transform)
        vectorizer=CountVectorizer(**processor.vectorizer.get_params()).fit(train_text)
        assert vectorizer.vocabulary_==processor.vectorizer.vocabulary_, 'Vocabulary not training-only'
        names=[extractor.FEATURE_NAMES[i] for i in keep]+list(processor.get_feature_names())
        assert [names[i] for i in chosen]==bundle['selected_names']
        if representation=='hostname_independent':
            assert not set(extractor.FEATURE_NAMES[i] for i in keep)&set(runner.IDENTITY_DESCRIPTORS)
        preds=pd.read_csv(out/(condition+'_validation_predictions.csv'))
        assert np.array_equal(preds.source_row_id.to_numpy(),validation)
        y=data.phishing.iloc[validation].to_numpy(dtype=int)
        assert np.array_equal(preds.label.to_numpy(),y)
        prob=preds.probability.to_numpy(); decision=prob>=.5
        assert np.isfinite(prob).all() and np.ptp(prob)>0
        assert np.array_equal(decision.astype(int),preds.prediction.to_numpy())
        tn,fp,fn,tp=confusion_matrix(y,decision,labels=[0,1]).ravel()
        recomputed=dict(accuracy=accuracy_score(y,decision),precision=precision_score(y,decision),
            recall=recall_score(y,decision),f1=f1_score(y,decision),auc=roc_auc_score(y,prob),fpr=fp/(fp+tn),fnr=fn/(tp+fn))
        expected=metrics.loc[metrics.condition.eq(condition)].iloc[0]
        assert all(abs(float(v)-expected[k])<1e-12 for k,v in recomputed.items())
        sample=data.url.iloc[validation[:40]].astype(str).tolist()
        hc=np.asarray([extractor.generate_features(u) for u in sample],dtype=np.float32)[:,keep]
        ng=processor.transform(pd.Series([transform(u) for u in sample])).toarray()
        x=np.hstack([hc,ng]).astype(np.float32)
        calculated=bundle['model'].predict_proba(bundle['scaler'].transform(x[:,chosen]),num_threads=2)[:,1]
        err=float(np.max(np.abs(calculated-prob[:40]))); maxerr=max(maxerr,err)
        assert err<1e-7
        # Recompute source-only scaling over every training row in bounded batches.
        total=np.zeros(len(chosen),dtype=np.float64)
        for start in range(0,len(train_urls),500):
            urls=train_urls.iloc[start:start+500]
            hc=np.asarray([extractor.generate_features(u) for u in urls],dtype=np.float32)[:,keep]
            ng=processor.transform(urls.map(transform)).toarray()
            total+=np.hstack([hc,ng]).astype(np.float32)[:,chosen].sum(axis=0,dtype=np.float64)
        assert np.allclose(total/len(train_urls),bundle['scaler'].mean_,rtol=1e-7,atol=1e-7), 'Scaler not source training fitted'
        selected_names[condition]=bundle['selected_names']
        artifacts[condition]=sha(out/(condition+'_pipeline.joblib'))
    assert selected_names['original']==selected_names['source_nuisance_balanced']
    pd.DataFrame({'draw_position':np.arange(len(draws)),'source_row_id':train[draws],'label':ytrain[draws]}).to_csv(out/'verified_balancing_draws.csv',index=False)
    report=dict(status='SOURCE_FIT_VERIFIED',conditions=conditions,validation_n=len(validation),
        training_n=len(train),no_test_or_external_fitting=True,train_vocabulary_and_scaler_verified=True,
        common_original_balanced_selection_verified=True,resampling_reproduced=True,
        max_recomputed_probability_error=maxerr,pipeline_sha256=artifacts,
        qualification_boundary='This verifies implementation, not unseen-domain effectiveness; preserve every condition regardless of validation score.')
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'}
    (out/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
