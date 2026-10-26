"""Verify paired cohort denominators, source isolation, metrics and inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('base',type=Path)
    p.add_argument('evaluation',type=Path)
    p.add_argument('--pipeline-root',type=Path,default=Path('.'))
    p.add_argument('--dataset',type=Path,default=Path('your_path/PhishFusion.csv'))
    args=p.parse_args()
    out=args.evaluation
    assert not (out/'verification.json').exists(), 'Do not overwrite evidence'
    root=args.pipeline_root.resolve();sys.path.insert(0,str(root));sys.path.insert(0,str(root/'experiments'))
    from feature_extraction import FeatureExtractor
    from model import FocalLossLGBM
    complete=json.loads((out/'complete.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    assert complete['status'] in ('QUALIFICATION_COMPLETE','FIXED_EVALUATION_COMPLETE')
    qualification=complete['status']=='QUALIFICATION_COMPLETE'
    assert protocol['source_code_sha256']==sha(out/'evaluator_snapshot.py')
    conditions=tuple(complete['conditions'])
    assert set(conditions)=={'original','source_nuisance_balanced','hostname_independent'}
    expected={'source_heldout':11376,'ExpertFusion_TE6':94717,'LitePhish_PhishTank':41761,
              'historical_positive_temporal':290,'historical_popular_dns_proxy':89937}
    if qualification: expected={k:40 for k in expected}
    assert complete['cohort_denominators']==expected
    source=pd.read_csv(args.dataset)
    fit=args.base/'full_source'
    assert sha(fit/'verification.json')==protocol['fit_verification_sha256']
    split=pd.read_csv(fit/'reconstructed_source_split.csv')
    summary=pd.read_csv(out/'summary.csv')
    assert len(summary)==15
    extractor=FeatureExtractor()
    bundles={c:joblib.load(fit/(c+'_pipeline.joblib')) for c in conditions}
    max_error=0
    for name,n in expected.items():
        path=out/(name+'_paired_predictions.csv.gz')
        evidence=json.loads((out/(name+'_evidence.json')).read_text())
        assert sha(path)==evidence['prediction_sha256'] and evidence['n']==n
        frame=pd.read_csv(path)
        assert len(frame)==n and not frame.source_row_id.duplicated().any()
        if name=='source_heldout':
            ids=split.loc[split.split.eq('test'),'row_id'].to_numpy(dtype=int)
            if qualification: ids=ids[:40]
            assert np.array_equal(frame.source_row_id,ids)
            assert np.array_equal(frame.label,source.phishing.iloc[ids])
            assert np.array_equal(frame.url,source.url.iloc[ids])
        elif name in ('ExpertFusion_TE6','LitePhish_PhishTank'):
            eligibility=pd.read_csv(args.base/'audit_v2'/(name+'_eligibility.csv.gz'))
            ids=eligibility.loc[eligibility.eligible_full_domain_disjoint,'row_id'].to_numpy(dtype=int)
            if qualification: ids=ids[:40]
            assert np.array_equal(frame.source_row_id,ids)
            assert np.array_equal(frame.label,eligibility.iloc[ids].label)
            assert not frame.source_domain_overlap.any() and not frame.source_normalized_url_overlap.any()
        elif name=='historical_positive_temporal':
            assert frame.label.eq(1).all() and not frame.source_domain_overlap.any()
        else:
            assert frame.label.eq(0).all()
            assert 'not verified benign' in evidence['provenance']
        y=frame.label.to_numpy(dtype=int)
        mixed=len(np.unique(y))==2
        for c in conditions:
            score=frame[c+'_probability'].to_numpy(dtype=float)
            decision=score>=.5
            assert np.isfinite(score).all() and ((score>=0)&(score<=1)).all()
            assert np.array_equal(decision.astype(int),frame[c+'_prediction'])
            tn,fp,fn,tp=confusion_matrix(y,decision,labels=[0,1]).ravel()
            row=summary.loc[summary.cohort.eq(name)&summary.condition.eq(c)]
            assert len(row)==1; row=row.iloc[0]
            for key,val in [('n',n),('tn',tn),('fp',fp),('fn',fn),('tp',tp)]: assert row[key]==val
            recomputed={}
            if mixed:
                recomputed.update(accuracy=accuracy_score(y,decision),precision=precision_score(y,decision,zero_division=0),
                    f1=f1_score(y,decision,zero_division=0),roc_auc=roc_auc_score(y,score),average_precision=average_precision_score(y,score))
            else:
                assert all(pd.isna(row[k]) for k in ('accuracy','precision','f1','roc_auc','average_precision'))
            if (y==1).any(): recomputed.update(recall=decision[y==1].mean(),fnr=1-decision[y==1].mean())
            if (y==0).any(): recomputed['fpr_or_proxy_flag_rate']=decision[y==0].mean()
            assert all(abs(float(v)-row[k])<1e-12 for k,v in recomputed.items())
            bundle=bundles[c]
            assert sha(fit/(c+'_pipeline.joblib'))==protocol['pipeline_sha256'][c]
            urls=frame.url.iloc[:40].astype(str)
            text=urls if bundle['representation']=='original' else urls.map(lambda u:(urlsplit(u).path or '/')+('?' + urlsplit(u).query if urlsplit(u).query else ''))
            hc=np.asarray([extractor.generate_features(u) for u in urls],dtype=np.float32)[:,bundle['keep_handcrafted']]
            ng=bundle['processor'].transform(text).toarray()
            x=np.hstack([hc,ng]).astype(np.float32)
            pred=bundle['model'].predict_proba(bundle['scaler'].transform(x[:,bundle['selected']]),num_threads=2)[:,1]
            error=float(np.max(np.abs(pred-score[:40])));max_error=max(max_error,error)
            assert error<1e-7
    report=dict(status='EVALUATION_QUALIFICATION_VERIFIED' if qualification else 'EVALUATION_VERIFIED',
        verified_cohorts=5,conditions=conditions,denominators=expected,
        max_independently_recomputed_probability_error=max_error,all_metrics_recomputed=True,
        no_condition_selected_using_external_scores=True,
        claim_boundary='Domain/snapshot robustness analysis on existing annotated corpora and explicitly limited stress proxies; not newly verified consumer traffic or new prospective temporal validation.')
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'}
    (out/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
