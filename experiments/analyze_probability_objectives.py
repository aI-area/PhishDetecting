#!/usr/bin/env python3
"""Seed-paired probability analysis and descriptive research figures."""
import argparse
import hashlib
import itertools
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy import stats

METHODS = ['standard','class_weight_balanced','alpha_only','focal']
NAMES = ['Standard BCE','Balanced BCE','Alpha-only','Focal weighting']
COLORS = ['#333333','#0072B2','#A65628','#007F5F']
STYLES = ['-', '--', '-.', ':']
RATIOS = ['1_to_3','1_to_5','1_to_10']
METRICS = ['f1','average_precision','roc_auc','brier','log_loss','ece']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def holm(values):
    order = np.argsort(values)
    result = np.empty(len(values))
    maximum = 0.
    for rank,i in enumerate(order):
        maximum = max(maximum, min(1., (len(values)-rank)*values[i]))
        result[i] = maximum
    return result


def paired(left, right, shared):
    differences = np.asarray(left)-np.asarray(right)
    assert len(differences)==10 and np.isfinite(differences).all()
    sd = float(differences.std(ddof=1))
    mean = float(differences.mean())
    half = stats.t.ppf(.975,9)*sd/math.sqrt(10)
    # Round only for rank-test tie handling; original differences remain retained.
    ranks = np.round(differences,12)
    rank_test = stats.wilcoxon(ranks,zero_method='pratt',alternative='two-sided',method='auto') if np.any(ranks) else None
    t_test = stats.ttest_rel(left,right) if sd>0 else None
    normality = stats.shapiro(differences) if sd>0 else None
    record = dict(**shared,n=10,left_mean=float(np.mean(left)),right_mean=float(np.mean(right)),
        mean_difference=mean,median_difference=float(np.median(differences)),sd_difference=sd,
        ci95_low=mean-half,ci95_high=mean+half,df=9,
        paired_dz=mean/sd if sd else None,
        wilcoxon_statistic=float(rank_test.statistic) if rank_test else 0.,
        wilcoxon_p=float(rank_test.pvalue) if rank_test else 1.,
        t_statistic=float(t_test.statistic) if t_test else None,
        t_p=float(t_test.pvalue) if t_test else (1. if mean==0 else 0.),
        shapiro_w=float(normality.statistic) if normality else None,
        shapiro_p=float(normality.pvalue) if normality else None)
    return record,differences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base',type=Path)
    args=ap.parse_args()
    base=args.base
    full=base/'full'
    verification=json.loads((full/'verification.json').read_text())
    assert verification['status']=='PROBABILITY_EVIDENCE_VERIFIED' and verification['models']==120
    assert (base/'ANALYSIS_PLAN.md').is_file()
    output=base/'analysis'
    output.mkdir(exist_ok=False)
    shutil.copy2(__file__,output/'analysis_runner_snapshot.py')
    shutil.copy2(base/'ANALYSIS_PLAN.md',output/'ANALYSIS_PLAN.md')
    df=pd.read_csv(full/'metrics.csv')
    assert len(df)==360 and not df[METRICS].isna().any().any()
    assert not df.duplicated(['imbalance_ratio','seed','method','variant']).any()
    summary=[]
    for (ratio,method,variant),g in df.groupby(['imbalance_ratio','method','variant']):
        assert len(g)==10 and set(g.seed)==set(range(42,52))
        for metric in METRICS+['precision','recall','accuracy']:
            values=g[metric].to_numpy()
            sd=values.std(ddof=1)
            half=stats.t.ppf(.975,9)*sd/np.sqrt(10)
            summary.append(dict(imbalance_ratio=ratio,method=method,variant=variant,metric=metric,
                n=10,mean=values.mean(),sd=sd,median=np.median(values),min=values.min(),max=values.max(),
                ci95_low=values.mean()-half,ci95_high=values.mean()+half))
    pd.DataFrame(summary).to_csv(output/'metric_summary.csv',index=False)
    comparisons=[]
    differences=[]
    for ratio,variant in itertools.product(RATIOS,['raw_fixed','sigmoid_fixed','raw_validation_f1']):
        for comparator in METHODS[:-1]:
            for metric in (['f1'] if variant=='raw_validation_f1' else METRICS):
                group=df[(df.imbalance_ratio==ratio)&(df.variant==variant)]
                left=group[group.method=='focal'].set_index('seed').sort_index()[metric].to_numpy()
                right=group[group.method==comparator].set_index('seed').sort_index()[metric].to_numpy()
                record,delta=paired(left,right,dict(family=variant,imbalance_ratio=ratio,
                    metric=metric,left='focal',right=comparator))
                comparisons.append(record)
                differences.extend([dict(**{k:record[k] for k in ['family','imbalance_ratio','metric','left','right']},
                    seed=seed,difference=float(d)) for seed,d in zip(range(42,52),delta)])
    for ratio,method,metric in itertools.product(RATIOS,METHODS,['f1','brier','log_loss','ece']):
        group=df[(df.imbalance_ratio==ratio)&(df.method==method)]
        left=group[group.variant=='sigmoid_fixed'].set_index('seed').sort_index()[metric].to_numpy()
        right=group[group.variant=='raw_fixed'].set_index('seed').sort_index()[metric].to_numpy()
        record,delta=paired(left,right,dict(family='within_method_calibration',imbalance_ratio=ratio,
            metric=metric,left=method+'_sigmoid',right=method+'_raw'))
        comparisons.append(record)
        differences.extend([dict(**{k:record[k] for k in ['family','imbalance_ratio','metric','left','right']},
            seed=seed,difference=float(d)) for seed,d in zip(range(42,52),delta)])
    paired_df=pd.DataFrame(comparisons)
    for family,index in paired_df.groupby('family').groups.items():
        expected={'raw_fixed':54,'sigmoid_fixed':54,'raw_validation_f1':9,'within_method_calibration':48}[family]
        assert len(index)==expected
        for test in ['wilcoxon','t']:
            paired_df.loc[index,test+'_p_holm']=holm(paired_df.loc[index,test+'_p'].to_numpy())
    paired_df.to_csv(output/'paired_comparisons.csv',index=False)
    difference_df=pd.DataFrame(differences)
    difference_df.to_csv(output/'paired_seed_differences.csv',index=False)
    figure_data=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    # Every assigned raw PR curve is shown; no interpolation or best-seed selection.
    for detail in [False,True]:
        fig,axes=plt.subplots(1,3,figsize=(10.6,3.5),layout='constrained')
        for ax,ratio in zip(axes,RATIOS):
            for method,name,color,style in zip(METHODS,NAMES,COLORS,STYLES):
                ap_mean=df[(df.imbalance_ratio==ratio)&(df.method==method)&(df.variant=='raw_fixed')].average_precision.mean()
                for seed in range(42,52):
                    curves=np.load(full/(ratio+'_seed'+str(seed))/(method+'_raw_fixed_curves.npz'))
                    ax.step(curves['recall'],curves['precision'],where='post',color=color,ls=style,
                        alpha=.65,lw=.75,label=name+f' (AP {ap_mean:.5f})' if seed==42 else None)
            ax.axhline(json.loads((full/'configuration.json').read_text())['test_prevalence'],
                color='#777777',ls='--',lw=.6)
            ax.set(xlim=(.8,1) if detail else (0,1),ylim=(.95,1) if detail else (0,1.01),
                xlabel='Recall',ylabel='Precision',title=ratio.replace('_to_',':')+' training ratio')
            ax.grid(alpha=.2)
            ax.legend(loc='lower left',fontsize=6.5)
        title='Precision–recall: high-performance view' if detail else 'Precision–recall: all ten seed curves'
        fig.suptitle(title+'; AP labels are per-seed means',fontsize=10)
        stem='precision_recall_detail' if detail else 'precision_recall_full'
        fig.savefig(output/(stem+'.pdf'),facecolor='white')
        fig.savefig(output/(stem+'.png'),dpi=220,facecolor='white')
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10.6,6.2),layout='constrained')
    for row,variant in enumerate(['raw_fixed','sigmoid_fixed']):
        for col,ratio in enumerate(RATIOS):
            ax=axes[row,col]
            ax.plot([0,1],[0,1],color='#777777',ls='--',lw=.8)
            for method,name,color,style,marker in zip(METHODS,NAMES,COLORS,STYLES,['o','s','^','D']):
                bins=[pd.read_csv(full/(ratio+'_seed'+str(seed))/(method+'_'+variant+'_reliability.csv')) for seed in range(42,52)]
                counts=sum(b['count'].to_numpy() for b in bins)
                probability_sum=sum(b['mean_probability'].fillna(0).to_numpy()*b['count'].to_numpy() for b in bins)
                positive_sum=sum(b['observed_positive_rate'].fillna(0).to_numpy()*b['count'].to_numpy() for b in bins)
                nonzero=counts>0
                x=probability_sum[nonzero]/counts[nonzero]
                y=positive_sum[nonzero]/counts[nonzero]
                ax.plot(x,y,marker=marker,ms=3,lw=.9,color=color,ls=style,label=name)
                for b in np.flatnonzero(nonzero):
                    figure_data.append(dict(imbalance_ratio=ratio,variant=variant,method=method,bin=int(b),
                        repeated_prediction_count=int(counts[b]),mean_probability=probability_sum[b]/counts[b],
                        observed_positive_rate=positive_sum[b]/counts[b]))
            ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted phishing probability',
                ylabel='Observed phishing fraction',title=ratio.replace('_to_',':')+' — '+('Raw' if row==0 else 'Sigmoid calibrated'))
            ax.grid(alpha=.2)
            if row==0 and col==0: ax.legend(fontsize=7)
    fig.suptitle('Reliability diagrams: pooled seed-run predictions, descriptive only',fontsize=10)
    fig.savefig(output/'calibration.pdf',facecolor='white')
    fig.savefig(output/'calibration.png',dpi=220,facecolor='white')
    plt.close(fig)
    pd.DataFrame(figure_data).to_csv(output/'calibration_figure_data.csv',index=False)
    fig,axes=plt.subplots(3,3,figsize=(10.6,8),layout='constrained')
    for row,metric in enumerate(['f1','average_precision','brier']):
        for col,ratio in enumerate(RATIOS):
            ax=axes[row,col]
            for pos,(method,name,color,marker) in enumerate(zip(METHODS,NAMES,COLORS,['o','s','^','D'])):
                values=df[(df.imbalance_ratio==ratio)&(df.method==method)&(df.variant=='raw_fixed')].sort_values('seed')[metric].to_numpy()
                ax.scatter(pos+np.linspace(-.12,.12,10),values,color=color,marker=marker,s=13,alpha=.8)
                ax.plot([pos-.18,pos+.18],[values.mean()]*2,color=color,lw=1.5)
            ax.set_xticks(range(4),['Std.','Balanced','Alpha','Focal'])
            ax.set_ylabel({'f1':'F1','average_precision':'Average precision','brier':'Brier score'}[metric])
            ax.set_title(ratio.replace('_to_',':')+' training ratio')
            ax.grid(axis='y',alpha=.2)
    fig.suptitle('Raw-probability outcomes: all ten paired seeds; bars denote means',fontsize=10)
    fig.savefig(output/'seed_outcomes.pdf',facecolor='white')
    fig.savefig(output/'seed_outcomes.png',dpi=220,facecolor='white')
    plt.close(fig)
    with PdfPages(output/'paired_difference_diagnostics.pdf') as pdf:
        for variant,ratio,comparator in itertools.product(['raw_fixed','sigmoid_fixed'],RATIOS,METHODS[:-1]):
            fig,axes=plt.subplots(2,3,figsize=(10.6,6.5),layout='constrained')
            for ax,metric in zip(axes.flat,METRICS):
                values=difference_df[(difference_df.family==variant)&(difference_df.imbalance_ratio==ratio)&
                    (difference_df.right==comparator)&(difference_df.metric==metric)].sort_values('seed').difference.to_numpy()
                stats.probplot(values,dist='norm',plot=ax)
                ax.set_title(metric)
            fig.suptitle(variant+' '+ratio+' focal minus '+comparator,fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
    captions={
        'precision_recall':'All ten raw per-seed precision–recall curves for four fixed objectives at each training imbalance ratio. Full panels show [0,1] context; detail panels explicitly restrict recall to [.8,1] and precision to [.95,1]. AP labels are means of individual APs, not the area under an averaged curve. Dashed horizontal line denotes original test prevalence, unchanged across training ratios.',
        'calibration':'Fifteen equal-width bins. Counts and probability/positive totals are pooled across seed-run predictions descriptively; the same test URLs recur in ten runs and are not independent observations. Mean per-seed ECE and its seed variability are in metric_summary.csv, not inferred from pooled curves. Sigmoid calibrators fit validation only.',
        'seed_outcomes':'Every assigned paired-seed value is shown; deterministic horizontal displacement aids visibility, and short horizontal bars indicate means. Axis scales vary by metric and ratio and are labeled. No significance stars.',
        'destination':'Provisional standalone research figures. No manuscript insertion or publisher-compliance claim. PDF fonts embedded; PNG220dpi previews.'}
    (output/'FIGURE_DESCRIPTIONS.json').write_text(json.dumps(captions,indent=2))
    audit=dict(status='ANALYSIS_COMPLETE',model_denominator=120,seed_count=10,paired_comparisons=len(paired_df),
        family_sizes=paired_df.groupby('family').size().to_dict(),normality_flags=int((paired_df.shapiro_p<.05).sum()),
        primary_test='two-sided Wilcoxon Pratt, Holm within declared family',
        pointwise_ci='paired mean difference Student t df9, not multiplicity adjusted',
        analysis_plan_sha256=sha(base/'ANALYSIS_PLAN.md'),metrics_sha256=sha(full/'metrics.csv'),
        software=dict(matplotlib=matplotlib.__version__,numpy=np.__version__,pandas=pd.__version__))
    (output/'analysis_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
    print(df[df.variant=='raw_fixed'].groupby(['imbalance_ratio','method'])[METRICS].mean().to_string())


if __name__=='__main__':
    main()
