"""Summarize all ten paired ablation seeds without excluding outcomes."""
import argparse,json,math,hashlib
from pathlib import Path
from itertools import combinations
import numpy as np,pandas as pd
from scipy import stats
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(exist_ok=False,parents=True)
    rows=pd.read_csv(a.base/'fits/per_seed_metrics.csv');assert len(rows)==40 and not rows[['seed','method']].duplicated().any();assert set(rows.seed)==set(range(42,52));rows.to_csv(out/'all_per_seed_metrics.csv',index=False)
    summary=[]
    for method,group in rows.groupby('method'):
        row=dict(method=method,seeds=len(group));assert len(group)==10;sets=[]
        for seed in range(42,52):
            parent=a.base/'fits';features=pd.read_csv(parent/('seed_'+str(seed))/method/'selected_features.csv',keep_default_na=False);sets.append(set(features.feature_index))
        jaccard=[len(x&y)/len(x|y) for x,y in combinations(sets,2)]
        row.update(pairwise_jaccard_mean=float(np.mean(jaccard)),pairwise_jaccard_sd=float(np.std(jaccard,ddof=1)))
        for metric in ['accuracy','precision','recall','f1','roc_auc','feature_count','selection_seconds','model_fit_seconds','mean_abs_correlation','max_abs_correlation']:
            values=group[metric].to_numpy();row[metric+'_mean']=float(values.mean());row[metric+'_sd']=float(values.std(ddof=1))
        summary.append(row)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    comparisons=[]
    for other in ['relevance_only','relevance_stability','relevance_correlation']:
        for metric in ['f1','roc_auc','mean_abs_correlation']:
            full=rows.loc[rows.method.eq('full_sdcs')].set_index('seed')[metric];reference=rows.loc[rows.method.eq(other)].set_index('seed')[metric];delta=(full-reference).sort_index().to_numpy();half=stats.t.ppf(.975,9)*delta.std(ddof=1)/math.sqrt(10);rounded=np.round(delta,12)
            prob=float(stats.wilcoxon(rounded,zero_method='pratt',method='auto').pvalue) if np.any(rounded) else 1.
            comparisons.append(dict(comparison='full_minus_'+other,metric=metric,n=10,mean_difference=float(delta.mean()),ci95_low=float(delta.mean()-half),ci95_high=float(delta.mean()+half),p_raw=prob,differences=json.dumps(delta.tolist())))
    order=np.argsort([r['p_raw'] for r in comparisons]);largest=0.
    for rank,i in enumerate(order):largest=max(largest,min(1.,(len(order)-rank)*comparisons[i]['p_raw']));comparisons[i]['p_holm_9']=largest
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv',index=False)
    (out/'complete.json').write_text(json.dumps(dict(status='TEN_SEED_COMPONENT_ABLATION_COMPLETE',seeds=10,conditions=4,models=40,all_conditions_retained=True),indent=2)+'\n')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
