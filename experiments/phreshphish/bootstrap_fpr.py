"""Descriptive fixed-model domain-cluster bootstrap for external benign FPR."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.results;out=a.output;out.mkdir(exist_ok=False,parents=True)
    frame=pd.read_csv(r/'predictions.csv.gz',float_precision='round_trip');thresholds=pd.read_csv(r/'thresholds.csv',float_precision='round_trip');rows=[];differences=[];rng=np.random.RandomState(20261026)
    for population in ['url_disjoint','domain_disjoint']:
        d=frame.loc[frame.label.eq(0)].copy()
        if population=='domain_disjoint':d=d.loc[d.eligible_domain_disjoint]
        codes,names=pd.factorize(d.registered_domain,sort=True);k=len(names);sizes=np.bincount(codes,minlength=k);flags=[]
        for t in thresholds.itertuples():flags.append(np.bincount(codes,weights=(d[t.condition+'_probability'].to_numpy()>=t.threshold).astype(int),minlength=k))
        flags=np.column_stack(flags);boot=[]
        for start in range(0,2000,100):
            draws=rng.multinomial(k,np.full(k,1/k),size=min(100,2000-start));den=draws@sizes;assert np.all(den>0);boot.append((draws@flags)/den[:,None])
        boot=np.vstack(boot);assert boot.shape==(2000,16)
        np.savez_compressed(out/(population+'_bootstrap_fpr.npz'),fpr=boot,domain_count=k,benign_urls=len(d))
        for i,t in enumerate(thresholds.itertuples()):
            values=boot[:,i];lo,hi=np.quantile(values,[.025,.975]);rows.append(dict(population=population,condition=t.condition,operating_point=t.operating_point,benign_urls=len(d),registered_domains=k,fpr=float(flags[:,i].sum()/len(d)),domain_bootstrap_ci_low=float(lo),domain_bootstrap_ci_high=float(hi),replicates=2000))
            if t.condition!='frozen_568':
                ref=next(j for j,v in enumerate(thresholds.itertuples()) if v.condition=='frozen_568' and v.operating_point==t.operating_point);delta=values-boot[:,ref];lo,hi=np.quantile(delta,[.025,.975]);differences.append(dict(population=population,comparison=t.condition+'_minus_frozen_568',operating_point=t.operating_point,difference=float((flags[:,i].sum()-flags[:,ref].sum())/len(d)),domain_bootstrap_ci_low=float(lo),domain_bootstrap_ci_high=float(hi),replicates=2000))
    pd.DataFrame(rows).to_csv(out/'absolute_fpr_intervals.csv',index=False);pd.DataFrame(differences).to_csv(out/'paired_fpr_differences.csv',index=False)
    (out/'complete.json').write_text(json.dumps(dict(status='DESCRIPTIVE_DOMAIN_BOOTSTRAP_COMPLETE',replicates=2000,seed=20261026,absolute_intervals=len(rows),paired_intervals=len(differences),scope='Fixed models and curated source; not population representativeness or training variance'),indent=2)+'\n');print((out/'complete.json').read_text(),flush=True)
if __name__=='__main__':main()
