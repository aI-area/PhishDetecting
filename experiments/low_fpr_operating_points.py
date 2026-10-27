"""Evaluate frozen models at prespecified validation-benign FPR budgets."""
import argparse,csv,gzip,json,hashlib,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def wilson(k,n):
    if not n:return (None,None)
    z=stats.norm.ppf(.975);a=k/n;den=1+z*z/n
    mid=(a+z*z/(2*n))/den;half=z*math.sqrt(a*(1-a)/n+z*z/(4*n*n))/den
    return mid-half,mid+half
def measures(y,p,t):
    pred=p>=t;pos=y==1;neg=y==0
    tp=int(np.sum(pred&pos));fp=int(np.sum(pred&neg));fn=int(np.sum(~pred&pos));tn=int(np.sum(~pred&neg))
    rec=tp/(tp+fn) if tp+fn else None;fpr=fp/(fp+tn) if fp+tn else None
    mixed=pos.any() and neg.any()
    return dict(n=len(y),phishing=int(pos.sum()),benign=int(neg.sum()),tp=tp,fp=fp,fn=fn,tn=tn,
        recall=rec,recall_ci_low=wilson(tp,tp+fn)[0],recall_ci_high=wilson(tp,tp+fn)[1],
        fpr_or_proxy_flag_rate=fpr,fpr_ci_low=wilson(fp,fp+tn)[0],fpr_ci_high=wilson(fp,fp+tn)[1],
        precision=tp/(tp+fp) if mixed and tp+fp else None,
        f1=2*tp/(2*tp+fp+fn) if mixed and 2*tp+fp+fn else None)
def thresholds(y,p):
    benign=np.sort(p[y==0])[::-1];assert len(benign)>0
    values=[('fixed_0.5',None,.5)]
    for budget in [.01,.005,.001]:
        k=int(math.floor(budget*len(benign)))
        t=float(np.nextafter(benign[k],np.inf));assert int(np.sum(benign>=t))<=k
        values.append(('validation_fpr_'+str(budget),budget,t))
    return values
def holm(ps):
    order=np.argsort(ps);out=np.empty(len(ps));v=0.
    for rank,i in enumerate(order):v=max(v,min(1.,(len(ps)-rank)*ps[i]));out[i]=v
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--paper-data',type=Path,required=True);ap.add_argument('--shift',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(exist_ok=False,parents=True);inputs={};objective=[];shift=[];threshold_records=[]
    partition=args.paper_data/'objective_partitions.npz';inputs[str(partition)]=sha(partition)
    with np.load(partition) as z:yv=z['y_validation'];yt=z['y_test']
    for folder in sorted((args.paper_data/'objective_predictions').iterdir()):
        ratio,seed=folder.name.rsplit('_seed',1)
        for path in sorted(folder.glob('*_probabilities.npz')):
            method=path.name.removesuffix('_probabilities.npz');inputs[str(path)]=sha(path)
            with np.load(path) as z:pv=z['validation'];pt=z['test']
            assert np.isfinite(pv).all() and np.isfinite(pt).all() and len(pv)==len(yv) and len(pt)==len(yt)
            for operating_point,budget,t in thresholds(yv,pv):
                val=measures(yv,pv,t);key=dict(ratio=ratio,seed=int(seed),method=method,operating_point=operating_point,target_validation_fpr=budget,threshold=t,validation_benign=val['benign'],validation_fp=val['fp'],validation_fpr=val['fpr_or_proxy_flag_rate'])
                objective.append(dict(**key,**measures(yt,pt,t)));threshold_records.append(dict(study='objective',**key))
    assert len(objective)==480
    conditions=['original','source_nuisance_balanced','hostname_independent'];selected={}
    for condition in conditions:
        p=args.shift/'full_source'/(condition+'_validation_predictions.csv');inputs[str(p)]=sha(p);r=read(p)
        y=np.array([int(x['label']) for x in r]);pval=np.array([float(x['probability']) for x in r]);selected[condition]=[]
        for point,budget,t in thresholds(y,pval):
            val=measures(y,pval,t);key=dict(condition=condition,operating_point=point,target_validation_fpr=budget,threshold=t,validation_benign=val['benign'],validation_fp=val['fp'],validation_fpr=val['fpr_or_proxy_flag_rate']);selected[condition].append(key)
    for path in sorted((args.shift/'evaluation_full').glob('*_paired_predictions.csv.gz')):
        inputs[str(path)]=sha(path);r=read(path);cohort=path.name.removesuffix('_paired_predictions.csv.gz');y=np.array([int(x['label']) for x in r])
        for condition in conditions:
            pt=np.array([float(x[condition+'_probability']) for x in r])
            for key in selected[condition]:shift.append(dict(cohort=cohort,annotation='DNS proxy' if 'dns_proxy' in cohort else ('positive only' if 'positive_temporal' in cohort else 'inherited mixed labels'),**key,**measures(y,pt,key['threshold'])))
    assert len(shift)==60
    write(args.output/'objective_operating_points.csv',objective);write(args.output/'shift_operating_points.csv',shift);write(args.output/'objective_thresholds.csv',threshold_records)
    grouped=defaultdict(list)
    for r in objective:grouped[(r['ratio'],r['method'],r['operating_point'])].append(r)
    summary=[]
    for (ratio,method,point),r in sorted(grouped.items()):
        row=dict(ratio=ratio,method=method,operating_point=point,seeds=len(r))
        for metric in ['threshold','validation_fpr','recall','fpr_or_proxy_flag_rate','precision','f1']:
            v=np.array([x[metric] if x[metric] is not None else np.nan for x in r]);row[metric+'_mean']=float(v.mean());row[metric+'_sd']=float(v.std(ddof=1))
        summary.append(row)
    write(args.output/'objective_summary.csv',summary)
    comparisons=[]
    for ratio in ['1_to_3','1_to_5','1_to_10']:
        for point in ['validation_fpr_0.01','validation_fpr_0.005','validation_fpr_0.001']:
            for other in ['standard','class_weight_balanced','alpha_only']:
                for metric in ['recall','fpr_or_proxy_flag_rate']:
                    a={r['seed']:r[metric] for r in objective if r['ratio']==ratio and r['operating_point']==point and r['method']=='focal'}
                    b={r['seed']:r[metric] for r in objective if r['ratio']==ratio and r['operating_point']==point and r['method']==other}
                    delta=np.array([a[s]-b[s] for s in sorted(a)]);assert len(delta)==10
                    rounded=np.round(delta,12);p=float(stats.wilcoxon(rounded,zero_method='pratt',method='auto').pvalue) if np.any(rounded) else 1.;half=stats.t.ppf(.975,9)*delta.std(ddof=1)/math.sqrt(10)
                    comparisons.append(dict(ratio=ratio,operating_point=point,comparison='focal_minus_'+other,metric=metric,n=10,mean_difference=float(delta.mean()),ci95_low=float(delta.mean()-half),ci95_high=float(delta.mean()+half),p_raw=p,differences=json.dumps(delta.tolist())))
    assert len(comparisons)==54
    for row,p in zip(comparisons,holm([r['p_raw'] for r in comparisons])):row['p_holm_54']=float(p)
    write(args.output/'paired_comparisons.csv',comparisons)
    (args.output/'input_hashes.json').write_text(json.dumps(inputs,indent=2)+'\n')
    (args.output/'complete.json').write_text(json.dumps(dict(status='COMPLETE',objective_rows=len(objective),shift_rows=len(shift),comparisons=len(comparisons),test_labels_used_for_thresholds=False),indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETE',objective_rows=len(objective),shift_rows=len(shift),output=str(args.output))))
if __name__=='__main__':main()
