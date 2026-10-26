"""Fixed held-out comparison of all three verified source-fitted conditions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlsplit
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix

CONDITIONS=('original','source_nuisance_balanced','hostname_independent')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def metrics(frame,condition):
    y=frame.label.to_numpy(dtype=int)
    score=frame[condition+'_probability'].to_numpy(dtype=float)
    decision=score>=.5
    tn,fp,fn,tp=confusion_matrix(y,decision,labels=[0,1]).ravel()
    npos=int((y==1).sum()); nneg=int((y==0).sum())
    mixed=npos>0 and nneg>0
    # Single-class diagnostic cohorts do not establish comparative precision/F1/AUC.
    return dict(condition=condition,n=len(frame),benign=nneg,phishing=npos,
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
        accuracy=float(accuracy_score(y,decision)) if mixed else None,
        precision=float(precision_score(y,decision,zero_division=0)) if mixed else None,
        recall=float(tp/npos) if npos else None,fnr=float(fn/npos) if npos else None,
        fpr_or_proxy_flag_rate=float(fp/nneg) if nneg else None,
        f1=float(f1_score(y,decision,zero_division=0)) if mixed else None,
        roc_auc=float(roc_auc_score(y,score)) if mixed else None,
        average_precision=float(average_precision_score(y,score)) if mixed else None)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('base',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--pipeline-root',type=Path,default=Path('.'))
    p.add_argument('--dataset',type=Path,default=Path('your_path/PhishFusion.csv'))
    p.add_argument('--reference-root',type=Path,default=Path('your_path/reference_experiments'))
    p.add_argument('--qualification',action='store_true')
    p.add_argument('--batch-size',type=int,default=500)
    args=p.parse_args()
    assert not args.output.exists(), 'Preserve prior evidence and never duplicate evaluation'
    assert shutil.disk_usage(args.base).free>2*1024**3, 'Less than 2GiB free; do not delete user data'
    if hasattr(os,'nice'): os.nice(10)
    if hasattr(os,'sched_getaffinity'): os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:8]))
    root=args.pipeline_root.resolve(); sys.path.insert(0,str(root)); sys.path.insert(0,str(root/'experiments'))
    from feature_extraction import FeatureExtractor
    from model import FocalLossLGBM
    from audit_shift_datasets import parse
    from run_litephish_failure_adversarial import natural_flags
    fit=args.base/'full_source'; auditdir=args.base/'audit_v2'
    verification=json.loads((fit/'verification.json').read_text())
    assert verification['status']=='SOURCE_FIT_VERIFIED' and set(verification['conditions'])==set(CONDITIONS)
    config=json.loads((fit/'configuration.json').read_text())
    assert config['status']=='FULL_SOURCE_TRAINING'
    for name,digest in config['training_pipeline_sources'].items(): assert sha(root/name)==digest
    assert sha(args.dataset)==config['source_sha256']
    bundles={condition:joblib.load(fit/(condition+'_pipeline.joblib')) for condition in CONDITIONS}
    for condition in CONDITIONS:
        assert sha(fit/(condition+'_pipeline.joblib'))==verification['pipeline_sha256'][condition]
    a,b=bundles['original'],bundles['source_nuisance_balanced']
    assert a['selected_names']==b['selected_names']
    assert np.array_equal(a['selected'],b['selected'])
    assert np.array_equal(a['keep_handcrafted'],b['keep_handcrafted'])
    assert np.array_equal(a['scaler'].mean_,b['scaler'].mean_)
    assert np.array_equal(a['scaler'].scale_,b['scaler'].scale_)
    assert a['processor'].vectorizer.vocabulary_==b['processor'].vectorizer.vocabulary_
    assert np.array_equal(a['processor'].selected_ngram_indices,b['processor'].selected_ngram_indices)
    source=pd.read_csv(args.dataset)
    source_keys=[parse(url) for url in source.url.astype(str)]
    source_domains={r[1] for r in source_keys}
    source_normalized={r[0] for r in source_keys}
    split=pd.read_csv(fit/'reconstructed_source_split.csv')
    assert sha(fit/'reconstructed_source_split.csv')==config['split_sha256']
    train_domains={source_keys[i][1] for i in split.loc[split.split.eq('train'),'row_id']}
    testids=split.loc[split.split.eq('test'),'row_id'].to_numpy(dtype=int)
    assert train_domains.isdisjoint({source_keys[i][1] for i in testids})
    heldout=source.iloc[testids].copy(); heldout.insert(0,'source_row_id',testids)
    heldout=heldout.rename(columns={'phishing':'label'})[['source_row_id','url','label']]
    cohorts={'source_heldout':heldout}
    provenance={'source_heldout':'Original fixed domain-disjoint source heldout; never fitted or used for condition selection.'}
    inputs={}
    audit=json.loads((auditdir/'dataset_audit.json').read_text())
    for name in ('ExpertFusion_TE6','LitePhish_PhishTank'):
        path=Path(audit[name]['path']); assert sha(path)==audit[name]['sha256']
        d=pd.read_csv(path); elig=pd.read_csv(auditdir/(name+'_eligibility.csv.gz'))
        ids=elig.loc[elig.eligible_full_domain_disjoint,'row_id'].to_numpy(dtype=int)
        frame=d.iloc[ids].copy(); frame.insert(0,'source_row_id',ids)
        frame=frame.rename(columns={'phishing':'label'})[['source_row_id','url','label']]
        assert np.array_equal(frame.label.to_numpy(),elig.iloc[ids].label.to_numpy())
        keys=[parse(u) for u in frame.url.astype(str)]
        assert all(r[3] and r[1] not in source_domains and r[0] not in source_normalized for r in keys)
        cohorts[name]=frame
        provenance[name]='External existing annotated full URLs; complete-source URL/domain overlap removed; not newly verified benign browser traffic or strict temporal independence.'
        inputs[name]={'data_sha256':sha(path),'eligibility_sha256':sha(auditdir/(name+'_eligibility.csv.gz'))}
    historical=args.reference_root
    temporal_path=historical/'experiment_25_temporal_generalization/cohort/openphish_post_freeze_cohort.csv'
    temporal=pd.read_csv(temporal_path).rename(columns={'row_id':'source_row_id','phishing':'label'})
    assert len(temporal)==290 and temporal.label.eq(1).all()
    assert all(parse(u)[1] not in source_domains for u in temporal.url)
    cohorts['historical_positive_temporal']=temporal[['source_row_id','url','label']].copy()
    provenance['historical_positive_temporal']='Historically exposed all-positive OpenPhish snapshot; detection recall only, not a new prospective validation of freshly trained conditions.'
    inputs['historical_positive_temporal']={'data_sha256':sha(temporal_path)}
    dns_path=historical/'experiment_29_consumer_benign/frozen_released/LitePhish_consumer_benign_predictions.csv.gz'
    dns=pd.read_csv(dns_path)
    assert len(dns)==89937 and dns.phishing.eq(0).all()
    dns=dns.rename(columns={'rank':'source_row_id','phishing':'label'})
    cohorts['historical_popular_dns_proxy']=dns[['source_row_id','url','label']].copy()
    provenance['historical_popular_dns_proxy']='Previously exposed DNS-derived https homepage proxy, not verified benign navigation. Retain full historic denominator; separately report source-domain-disjoint subset. Never relabel proxy membership as confirmed benignness.'
    inputs['historical_popular_dns_proxy']={'data_sha256':sha(dns_path)}
    if args.qualification:
        cohorts={name:frame.iloc[:40].copy() for name,frame in cohorts.items()}
    args.output.mkdir(parents=True)
    shutil.copyfile(Path(__file__),args.output/'evaluator_snapshot.py')
    (args.output/'protocol.json').write_text(json.dumps(dict(
        status='IMPLEMENTATION_QUALIFICATION' if args.qualification else 'FIXED_EVALUATION',
        conditions=CONDITIONS,threshold=.5,fit_verification_sha256=sha(fit/'verification.json'),
        pipeline_sha256=verification['pipeline_sha256'],source_config_sha256=sha(fit/'configuration.json'),
        input_sha256=inputs,cohort_provenance=provenance,
        no_fitting_or_threshold_selection=True,source_code_sha256=sha(Path(__file__))),indent=2))
    extractor=FeatureExtractor()
    summary=[]; strata=[]
    for name,frame in cohorts.items():
        frame=frame.reset_index(drop=True)
        assert not frame.source_row_id.duplicated().any()
        keys=[parse(u) for u in frame.url.astype(str)]
        frame['registered_domain']=[r[1] for r in keys]
        frame['source_domain_overlap']=[r[1] in source_domains for r in keys]
        frame['source_normalized_url_overlap']=[r[0] in source_normalized for r in keys]
        frame['path_or_query']=[r[4] for r in keys]
        frame['shared_or_reputable_host']=frame.url.map(lambda u:natural_flags(u)['natural_shared_or_reputable_host'])
        values={condition:[] for condition in CONDITIONS}
        for start in range(0,len(frame),args.batch_size):
            urls=frame.url.iloc[start:start+args.batch_size].astype(str)
            hc=np.asarray([extractor.generate_features(u) for u in urls],dtype=np.float32)
            prepared={}
            for condition in ('original','hostname_independent'):
                bundle=bundles[condition]
                text=urls if condition=='original' else urls.map(lambda u:(urlsplit(u).path or '/')+('?' + urlsplit(u).query if urlsplit(u).query else ''))
                ng=bundle['processor'].transform(text).toarray()
                x=np.hstack([hc[:,bundle['keep_handcrafted']],ng]).astype(np.float32)
                prepared[condition]=bundle['scaler'].transform(x[:,bundle['selected']])
            for condition in CONDITIONS:
                representation=bundles[condition]['representation']
                prob=np.asarray(bundles[condition]['model'].predict_proba(prepared[representation],num_threads=2)[:,1])
                assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
                values[condition].extend(prob.tolist())
            if start%5000==0: print(f'{name}: {min(start+args.batch_size,len(frame))}/{len(frame)}',flush=True)
        for condition in CONDITIONS:
            frame[condition+'_probability']=values[condition]
            frame[condition+'_prediction']=(frame[condition+'_probability']>=.5).astype(int)
            row=metrics(frame,condition); row['cohort']=name; summary.append(row)
        for column in ('path_or_query','shared_or_reputable_host','source_domain_overlap'):
            for value,group in frame.groupby(column):
                for condition in CONDITIONS:
                    row=metrics(group,condition)
                    row.update(cohort=name,stratum=column,value=bool(value)); strata.append(row)
        path=args.output/(name+'_paired_predictions.csv.gz')
        frame.to_csv(path,index=False)
        (args.output/(name+'_evidence.json')).write_text(json.dumps(dict(n=len(frame),prediction_sha256=sha(path),provenance=provenance[name]),indent=2))
        pd.DataFrame(summary).to_csv(args.output/'summary.csv',index=False)
        pd.DataFrame(strata).to_csv(args.output/'stratified_metrics.csv',index=False)
    report=dict(status='QUALIFICATION_COMPLETE' if args.qualification else 'FIXED_EVALUATION_COMPLETE',
        assigned_cohorts=5,conditions=CONDITIONS,cohort_denominators={k:len(v) for k,v in cohorts.items()},
        no_target_based_selection=True,no_claim_of_verified_consumer_traffic=True,manuscript_edited=False)
    (args.output/'complete.json').write_text(json.dumps(report,indent=2))
    hashes={p.name:sha(p) for p in args.output.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'}
    (args.output/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
