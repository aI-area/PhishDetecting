"""Exploratory paired domain-cluster uncertainty for fixed trained conditions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

CONDITIONS=('original','source_nuisance_balanced','hostname_independent')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(counts):
    # Last dimension: TP, FP, FN, TN. Preserve undefined denominators.
    tp,fp,fn,tn=np.moveaxis(counts,-1,0)
    with np.errstate(divide='ignore',invalid='ignore'):
        return {'f1':2*tp/(2*tp+fp+fn),'recall':tp/(tp+fn),'fpr_or_proxy_flag_rate':fp/(fp+tn)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('evaluation',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--replicates',type=int,default=2000)
    args=p.parse_args()
    assert not args.output.exists()
    verification=json.loads((args.evaluation/'verification.json').read_text())
    assert verification['status']=='EVALUATION_VERIFIED'
    os.nice(10); os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    args.output.mkdir(parents=True)
    shutil.copyfile(Path(__file__),args.output/'analysis_snapshot.py')
    protocol=dict(analysis='Exploratory paired nonparametric registered-domain cluster bootstrap',
        seed=20261020,replicates=args.replicates,confidence=.95,
        resampling='Uniformly sample G registered-domain clusters with replacement; retain all URLs and all three predictions from each selected cluster. Use common draws for every condition. Estimate URL-weighted ratios from sampled counts, not equal-weighted mean domain scores.',
        effects='Alternative minus matched original, in percentage points: positive F1/recall is higher; negative FPR/proxy flag rate is lower.',
        limitations='Pointwise percentile intervals, not simultaneous intervals or confirmatory significance tests. Descriptive sensitivity to sampled domains, conditional on these fixed fitted models and convenience cohorts. No training-seed variability, no real-world traffic sampling guarantee, no correction for residual campaign/feed correlations across domains. Source test and external results were already computed; analysis is exploratory, not preregistered.',
        metrics='F1, recall and FPR in mixed cohorts; recall only in positive temporal cohort; proxy flag rate only in DNS proxy. Retain all cohorts and both alternatives.',
        source_verification_sha256=sha(args.evaluation/'verification.json'))
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    effects=[]; diagnostics=[]; inputs={}
    for number,name in enumerate(verification['denominators']):
        path=args.evaluation/(name+'_paired_predictions.csv.gz')
        frame=pd.read_csv(path)
        assert len(frame)==verification['denominators'][name]
        assert frame.registered_domain.notna().all() and frame.registered_domain.astype(str).str.len().gt(0).all()
        codes,domains=pd.factorize(frame.registered_domain,sort=True)
        g=len(domains); assert g>=20
        sizes=np.bincount(codes,minlength=g)
        y=frame.label.to_numpy(dtype=int)
        counts=np.zeros((g,3,4),dtype=np.int64)
        for c,condition in enumerate(CONDITIONS):
            decision=frame[condition+'_prediction'].to_numpy(dtype=int)
            assert np.array_equal(decision,(frame[condition+'_probability'].to_numpy()>=.5).astype(int))
            for k,mask in enumerate(((y==1)&(decision==1),(y==0)&(decision==1),(y==1)&(decision==0),(y==0)&(decision==0))):
                counts[:,c,k]=np.bincount(codes[mask],minlength=g)
        assert np.array_equal(counts.sum(axis=(0,2)),np.full(3,len(frame)))
        rng=np.random.default_rng(20261020+number)
        distributions=[]
        probability=np.full(g,1/g)
        for start in range(0,args.replicates,25):
            n=min(25,args.replicates-start)
            multiplicities=rng.multinomial(g,probability,size=n)
            sample=(multiplicities@counts.reshape(g,-1)).reshape(n,3,4)
            distributions.append(sample)
        draws=np.concatenate(distributions)
        point=summarize(counts.sum(axis=0))
        bootstrap=summarize(draws)
        mixed=len(np.unique(y))==2
        selected_metrics=('f1','recall','fpr_or_proxy_flag_rate') if mixed else ('recall',) if y[0]==1 else ('fpr_or_proxy_flag_rate',)
        saved={}
        for metric in selected_metrics:
            for c,condition in enumerate(CONDITIONS[1:],start=1):
                difference=100*(bootstrap[metric][:,c]-bootstrap[metric][:,0])
                valid=np.isfinite(difference)
                assert valid.sum()>=.95*args.replicates
                low,high=np.quantile(difference[valid],[.025,.975])
                saved[condition+'__'+metric]=difference
                effects.append(dict(cohort=name,alternative=condition,metric=metric,
                    original_score=float(point[metric][0]),alternative_score=float(point[metric][c]),
                    difference_pp=float(100*(point[metric][c]-point[metric][0])),
                    pointwise_95_ci_low_pp=float(low),pointwise_95_ci_high_pp=float(high),
                    valid_replicates=int(valid.sum()),total_replicates=args.replicates,
                    clusters=g,rows=len(frame),interval_scope='Exploratory fixed-model paired domain resampling'))
        np.savez_compressed(args.output/(name+'_bootstrap_differences.npz'),**saved)
        diagnostics.append(dict(cohort=name,rows=len(frame),clusters=g,benign=int((y==0).sum()),phishing=int((y==1).sum()),
            cluster_size_min=int(sizes.min()),cluster_size_median=float(np.median(sizes)),
            cluster_size_p95=float(np.quantile(sizes,.95)),cluster_size_max=int(sizes.max()),
            largest_cluster_row_share=float(sizes.max()/len(frame)),
            missing_labels=int(frame.label.isna().sum()),missing_probabilities=int(frame[[c+'_probability' for c in CONDITIONS]].isna().sum().sum()),
            independent_url_assumption=False,domain_cluster_exchangeability_assumed=True))
        inputs[name]=sha(path)
        print(f'{name}: {g} clusters, {len(frame)} rows, {args.replicates} paired draws',flush=True)
    pd.DataFrame(effects).to_csv(args.output/'paired_domain_effects.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(args.output/'cluster_diagnostics.csv',index=False)
    (args.output/'complete.json').write_text(json.dumps(dict(status='EXPLORATORY_DOMAIN_UNCERTAINTY_COMPLETE',
        effects=len(effects),cohorts=len(diagnostics),replicates=args.replicates,input_sha256=inputs,
        no_model_fit=True,no_p_value_or_superiority_claim=True,manuscript_edited=False),indent=2))
    hashes={p.name:sha(p) for p in args.output.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'}
    (args.output/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2))


if __name__=='__main__': main()
