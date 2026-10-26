"""Fit source-only URL representations and nuisance-balanced comparisons."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import shutil
import time
from urllib.parse import urlsplit
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score,confusion_matrix

IDENTITY_DESCRIPTORS=('Suspicious Keyword','URL Shortener','TLD Risk Score',
    'Suspicious Hosting','TLD-Domain Mismatch','TLD is .com','Redirect TLD Disparity')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def balance_indices(urls,y,seed=42):
    """Source-only nuisance matching by resampling existing training rows."""
    q=np.unique(np.quantile([len(u) for u in urls],[.25,.5,.75]))
    groups=[]
    for url in urls:
        p=urlsplit(url)
        groups.append((p.scheme,int(np.searchsorted(q,len(url),side='right')),
                       min(len([s for s in p.path.split('/') if s]),2),bool(p.query)))
    counts={}
    for label,group in zip(y,groups): counts[(int(label),group)]=counts.get((int(label),group),0)+1
    sizes={label:int((y==label).sum()) for label in (0,1)}
    weights=[]
    unsupported=0
    for label,group in zip(y,groups):
        if not all((c,group) in counts for c in (0,1)):
            weights.append(1.); unsupported+=1; continue
        target=.5*sum(counts[(c,group)]/sizes[c] for c in (0,1))
        weights.append(float(np.clip(target/(counts[(int(label),group)]/sizes[int(label)]),.2,5)))
    weights=np.asarray(weights)
    rng=np.random.RandomState(seed)
    selected=[]
    for label in (0,1):
        idx=np.flatnonzero(y==label)
        selected.extend(rng.choice(idx,len(idx),replace=True,p=weights[idx]/weights[idx].sum()))
    selected=np.asarray(selected,dtype=int)
    rng.shuffle(selected)
    assert np.array_equal(np.bincount(y[selected],minlength=2),np.bincount(y,minlength=2))
    return selected,dict(length_quartile_cutpoints=q.tolist(),unsupported_source_rows=unsupported,
        explanation='Resample only existing source-training rows; preserve class counts; no target statistics. Missing class-by-stratum support receives weight1, not fabricated samples.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    p.add_argument('--pipeline-root',type=Path,default=Path('.'))
    p.add_argument('--dataset',type=Path,default=Path('your_path/PhishFusion.csv'))
    p.add_argument('--full',action='store_true',help='Use the complete source train/validation partitions for the reported comparison')
    args=p.parse_args()
    assert not args.output.exists(), 'Output directory already exists'
    if hasattr(os,'nice'): os.nice(10)
    if hasattr(os,'sched_getaffinity'):
        allowed=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,set(allowed[:8]))
    os.environ['CUDA_VISIBLE_DEVICES']=''
    root=args.pipeline_root.resolve()
    sys.path.insert(0,str(root))
    from feature_extraction import FeatureExtractor
    from feature_selection import FeatureSelector
    from ngram_processing import NgramProcessor
    from model import FocalLossLGBM
    from main import PhishingDetector,extract_tld_parts
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    source=args.dataset
    df=pd.read_csv(source)
    train_idx,val_idx,test_idx=PhishingDetector().split_data_indices(df.url,df.phishing)
    assert train_idx is not None
    historical_test_hash=hashlib.sha256(np.ascontiguousarray(test_idx).view(np.uint8)).hexdigest()
    assert historical_test_hash=='b37ff11d8baf705026dbe833e65a9425c3d2822cabcebc5ed9ed300504082921', 'Historical split reproduction failed'
    partitions=np.empty(len(df),dtype=object)
    partitions[train_idx]='train'; partitions[val_idx]='validation'; partitions[test_idx]='test'
    groups=[]
    for url in df.url:
        ext=extract_tld_parts(str(url)); groups.append(f'{ext.domain}.{ext.suffix}')
    manifest=pd.DataFrame({'row_id':np.arange(len(df)),'label':df.phishing,'split':partitions,'esld':groups})
    assert manifest.groupby('esld').split.nunique().max()==1
    rng=np.random.RandomState(42)
    indices={}
    for part,cap in [('train',None if args.full else 12000),('validation',None if args.full else 2000)]:
        candidate=np.flatnonzero(manifest.split.eq(part).to_numpy())
        if cap is None:
            indices[part]=candidate
            continue
        # Proportional stratification, entirely within fixed source partitions.
        per_label=[]
        for label in (0,1):
            choices=candidate[df.phishing.iloc[candidate].to_numpy()==label]
            count=min(len(choices),max(1,round(cap*len(choices)/len(candidate))))
            per_label.extend(rng.choice(choices,count,replace=False))
        indices[part]=np.sort(per_label)
    selected_rows=np.concatenate([indices['train'],indices['validation']])
    urls=df.url.iloc[selected_rows].reset_index(drop=True).astype(str)
    y=df.phishing.iloc[selected_rows].to_numpy(dtype=int)
    ntrain=len(indices['train']); ytr=y[:ntrain]; yval=y[ntrain:]
    assert set(indices['train']).isdisjoint(indices['validation'])
    args.output.mkdir(parents=True)
    shutil.copyfile(Path(__file__),args.output/'runner_snapshot.py')
    split=args.output/'reconstructed_source_split.csv'
    manifest.to_csv(split,index=False)
    pd.DataFrame({'source_row_id':selected_rows,'partition':['train']*ntrain+['validation']*len(yval)}).to_csv(args.output/'source_pilot_rows.csv',index=False)
    config=dict(status='FULL_SOURCE_TRAINING' if args.full else 'SOURCE_ONLY_QUALIFICATION',source_sha256=sha(source),split_sha256=sha(split),
        seed=42,train_rows=ntrain,validation_rows=len(yval),source_test_opened=False,external_targets_opened=False,
        threshold=.5,excluded_identity_descriptors=IDENTITY_DESCRIPTORS,source_code_sha256=sha(Path(__file__)),
        training_pipeline_sources={name:sha(root/name) for name in ('feature_extraction.py','feature_selection.py','ngram_processing.py','model.py')},
        caution='Training/validation only; no external effectiveness claim. Final external evaluation remains separate.' if args.full else 'Small source-only pilot establishes implementation feasibility, not external effectiveness or final statistical evidence.')
    (args.output/'configuration.json').write_text(json.dumps(config,indent=2))
    extract=FeatureExtractor()
    hc=np.asarray([extract.generate_features(url) for url in urls],dtype=np.float32)
    assert hc.shape[1]==len(extract.FEATURE_NAMES)
    metrics=[]
    balance,balancing=balance_indices(urls.iloc[:ntrain].tolist(),ytr)
    (args.output/'balancing_rule.json').write_text(json.dumps(balancing,indent=2))
    pd.DataFrame({'draw_position':np.arange(len(balance)),'source_row_id':indices['train'][balance],
        'label':ytr[balance]}).to_csv(args.output/'resampling_draws.csv',index=False)
    for representation in ('original','hostname_independent'):
        logging.info('Starting source-only representation: %s',representation)
        text=urls if representation=='original' else urls.map(lambda u:(urlsplit(u).path or '/')+('?' + urlsplit(u).query if urlsplit(u).query else ''))
        processor=NgramProcessor(); processor.fit_vectorizer(text.iloc[:ntrain])
        processor.select_features(text.iloc[:ntrain],pd.Series(ytr))
        keep=np.array([i for i,name in enumerate(extract.FEATURE_NAMES) if representation=='original' or name not in IDENTITY_DESCRIPTORS])
        names=[extract.FEATURE_NAMES[i] for i in keep]+list(processor.get_feature_names())
        x=np.hstack([hc[:,keep],processor.transform(text).toarray()]).astype(np.float32)
        selector=FeatureSelector()
        chosen=selector.perform_merged_selection(pd.DataFrame(x[:ntrain],columns=names),pd.Series(ytr),
            stability_runs=20,sample_fraction=.905,frequency_threshold=.705,regularization_strength=.191,
            mi_weight=.318,alpha=.767,num_features=1000,correlation_threshold=.956,
            sample_size=20000,random_seed=42,n_estimators=200)
        assert len(chosen)>0 and len(set(chosen))==len(chosen)
        scaler=StandardScaler().fit(x[:ntrain,chosen])
        xtr=scaler.transform(x[:ntrain,chosen]); xval=scaler.transform(x[ntrain:,chosen])
        for condition in ([representation, 'source_nuisance_balanced'] if representation=='original' else [representation]):
            train_rows=balance if condition=='source_nuisance_balanced' else np.arange(ntrain)
            model=FocalLossLGBM(gamma=.55,alpha=.95,n_estimators=500,num_leaves=50,learning_rate=.15,
                min_child_samples=40,reg_alpha=.1,reg_lambda=.3,random_state=42,n_jobs=8,
                deterministic=True,force_col_wise=True,verbosity=-1,device='cpu')
            start=time.perf_counter(); model.fit(xtr[train_rows],ytr[train_rows])
            prob=model.predict_proba(xval,num_threads=2)[:,1]
            assert np.isfinite(prob).all() and np.ptp(prob)>0
            pred=prob>=.5
            tn,fp,fn,tp=confusion_matrix(yval,pred,labels=[0,1]).ravel()
            metrics.append(dict(condition=condition,selected_features=len(chosen),fit_s=time.perf_counter()-start,
                accuracy=float(accuracy_score(yval,pred)),precision=float(precision_score(yval,pred,zero_division=0)),
                recall=float(recall_score(yval,pred)),f1=float(f1_score(yval,pred)),auc=float(roc_auc_score(yval,prob)),
                fpr=float(fp/(fp+tn)),fnr=float(fn/(fn+tp))))
            pd.DataFrame({'source_row_id':indices['validation'],'label':yval,'probability':prob,'prediction':pred.astype(int)}).to_csv(args.output/(condition+'_validation_predictions.csv'),index=False)
            joblib.dump(dict(model=model,processor=processor,keep_handcrafted=keep,selected=chosen,
                selected_names=[names[i] for i in chosen],scaler=scaler,representation=representation),args.output/(condition+'_pipeline.joblib'))
            pd.DataFrame(metrics).to_csv(args.output/'validation_summary.csv',index=False)
        del x,xtr,xval
    (args.output/'complete.json').write_text(json.dumps(dict(status='FULL_SOURCE_TRAINING_COMPLETE' if args.full else 'SOURCE_ONLY_PILOT_COMPLETE',metrics=metrics),indent=2))


if __name__=='__main__': main()
