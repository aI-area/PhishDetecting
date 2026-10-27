"""Paired fixed-budget ablation of relevance, stability and correlation control."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(k,'1')
import argparse,hashlib,json,logging,shutil,sys,time,traceback
from pathlib import Path
import joblib,numpy as np,pandas as pd
from joblib import Parallel,delayed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,matthews_corrcoef

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def reserve(path):
    assert shutil.disk_usage(path).free>2*1024**3,'Disk reserve exhausted'
class FailOnError(logging.Handler):
    def emit(self,record):
        if record.levelno>=logging.ERROR:raise RuntimeError(record.getMessage())
def metrics(y,p):
    pred=(p>=.5).astype(int)
    return dict(accuracy=float(accuracy_score(y,pred)),precision=float(precision_score(y,pred)),recall=float(recall_score(y,pred)),f1=float(f1_score(y,pred)),roc_auc=float(roc_auc_score(y,p)),mcc=float(matthews_corrcoef(y,pred)))
def prune(x,ranking,k=None):
    start=time.perf_counter();ranking=ranking[:5000]
    c=np.abs(np.corrcoef(x[:,ranking],rowvar=False));np.nan_to_num(c,copy=False)
    rejected=np.zeros(len(ranking),dtype=bool);selected=[];positions=[]
    for i in range(len(ranking)):
        if rejected[i]:continue
        selected.append(int(ranking[i]));positions.append(i)
        if len(selected)==(k if k is not None else 1000):break
        mask=c[i]>.956;mask[:i+1]=False;rejected|=mask
    if k is not None:assert len(selected)==k,'Cannot satisfy matched feature budget'
    a=c[np.ix_(positions,positions)];upper=a[np.triu_indices_from(a,k=1)]
    maximum=float(upper.max()) if len(upper) else 0.
    assert maximum<=.956+1e-12,'Pruning constraint failed on selection population'
    return np.array(selected),time.perf_counter()-start,maximum
def redundancy(x,indices,seed):
    rng=np.random.RandomState(seed);rows=rng.choice(len(x),min(20000,len(x)),replace=False)
    c=np.abs(np.corrcoef(x[rows][:,indices],rowvar=False));upper=c[np.triu_indices_from(c,k=1)];upper=upper[np.isfinite(upper)]
    return dict(mean_abs_correlation=float(upper.mean()),max_abs_correlation=float(upper.max()),p95_abs_correlation=float(np.quantile(upper,.95)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--anchor',type=Path,required=True);ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seeds',type=int,nargs='+',required=True);ap.add_argument('--jobs',type=int,default=8);ap.add_argument('--pipeline-root',type=Path,default=Path(__file__).resolve().parents[1]/'artifacts/phreshphish/source_definitions');ap.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'manifests/PhishFusion.csv');args=ap.parse_args()
    args.output.mkdir(exist_ok=False,parents=True);shutil.copy2(__file__,args.output/'runner_snapshot.py')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s');logging.getLogger().addHandler(FailOnError())
    sys.path.insert(0,str(args.pipeline_root))
    from feature_extraction import FeatureExtractor
    from feature_selection import FeatureSelector
    from ngram_processing import NgramProcessor
    from model import FocalLossLGBM
    anchor=args.anchor;verified=json.loads((anchor/'selectors_full/verification.json').read_text());assert verified['status']=='MATCHED_SELECTORS_VERIFIED' and verified['models']==60
    assert sha(args.dataset)==verified['dataset_sha256']=='c8af8da76bb6a521fcb7515eab802fec450dd10eafdfb49593096dcc07f1de8c'
    manifest_path=args.manifest;assert sha(manifest_path)==verified['manifest_sha256']
    manifest=pd.read_csv(manifest_path);data=pd.read_csv(args.dataset,encoding='latin1');assert len(data)==len(manifest)==105752
    label='phishing' if 'phishing' in data else 'label';labels=data[label].to_numpy(dtype=int);urls=data.url.astype(str)
    assert np.array_equal(labels,manifest.label.to_numpy(dtype=int))
    assert all(hashlib.sha256(u.encode()).hexdigest()==h for u,h in zip(urls,manifest.url_sha256))
    ix={s:np.flatnonzero(manifest.split.eq(s)) for s in ['train','validation','test']}
    assert [len(ix[s]) for s in ix]==[86098,8278,11376]
    cache=args.base/'candidate_cache';reserve(args.base)
    if not (cache/'complete.json').exists():
        cache.mkdir(exist_ok=False)
        processor=joblib.load(anchor/'selectors_full/ngram_processor.joblib');extractor=FeatureExtractor()
        times={};start=time.perf_counter()
        hc=np.asarray(Parallel(n_jobs=args.jobs,prefer='threads')(delayed(extractor.generate_features)(u) for u in urls),dtype=np.float32);times['handcrafted_all_rows_s']=time.perf_counter()-start
        start=time.perf_counter();ng=processor.transform(urls).toarray().astype(np.float32,copy=False);times['selected_ngram_transform_all_rows_s']=time.perf_counter()-start
        start=time.perf_counter();x=np.hstack([hc,ng]).astype(np.float32,copy=False);del hc,ng
        assert x.shape==(105752,5086) and np.isfinite(x).all();names=list(extractor.FEATURE_NAMES)+list(processor.get_feature_names())
        assert len(names)==5086
        for s in ['train','test']:np.save(cache/('x_'+s+'.npy'),x[ix[s]])
        times['assembly_and_cache_write_s']=time.perf_counter()-start
        np.savez(cache/'partitions.npz',**{s+'_indices':ix[s] for s in ix},**{'y_'+s:labels[ix[s]] for s in ix})
        dump(cache/'feature_names.json',names);dump(cache/'timing.json',times)
        dump(cache/'complete.json',dict(dataset_sha256=sha(args.dataset),manifest_sha256=sha(manifest_path),processor_sha256=sha(anchor/'selectors_full/ngram_processor.joblib'),files={p.name:sha(p) for p in cache.iterdir() if p.is_file()}));del x
    metadata=json.loads((cache/'complete.json').read_text());assert metadata['dataset_sha256']==sha(args.dataset) and metadata['manifest_sha256']==sha(manifest_path)
    for name,h in metadata['files'].items():assert sha(cache/name)==h
    x_train=np.load(cache/'x_train.npy',mmap_mode='r');x_test=np.load(cache/'x_test.npy',mmap_mode='r');y_train=labels[ix['train']];y_test=labels[ix['test']]
    names=json.loads((cache/'feature_names.json').read_text());selector=FeatureSelector();all_rows=[]
    for seed in args.seeds:
        reserve(args.base);folder=args.output/('seed_'+str(seed));folder.mkdir(exist_ok=False)
        logging.info('Starting component ablation seed %s',seed)
        rng=np.random.RandomState(seed);subset=np.sort(rng.choice(len(y_train),int(.905*len(y_train)),replace=False));xs=np.array(x_train[subset],dtype=np.float32);ys=y_train[subset]
        np.save(folder/'selector_training_positions.npy',subset)
        start=time.perf_counter();frequency=selector.perform_stability_selection(xs,ys,.905,20,.191,np.random.RandomState(seed));stability_s=time.perf_counter()-start
        start=time.perf_counter();hybrid=selector.perform_hybrid_selection(xs,ys,.318,200);relevance_s=time.perf_counter()-start
        assert np.isfinite(frequency).all() and np.isfinite(hybrid).all()
        composite=.767*frequency+(1-.767)*hybrid;eligible=np.where(frequency>=.705)[0];assert len(eligible)>0,'No stability-admissible variables'
        rank_stable=eligible[np.argsort(composite[eligible])[::-1]];rank_relevance=np.argsort(hybrid)[::-1]
        full,full_prune_s,full_max=prune(xs,rank_stable)
        reference=pd.read_csv(anchor/'selectors_full/selected_features'/('seed_'+str(seed))/'sdcs.csv',keep_default_na=False)
        assert np.array_equal(full,reference.feature_index.to_numpy(dtype=int)),'Full SDCS indices differ from frozen reference'
        k=len(full);assert 1<=k<=1000
        rc,rc_prune_s,rc_max=prune(xs,rank_relevance,k)
        variants=[('relevance_only',rank_relevance[:k],relevance_s,None),('relevance_stability',rank_stable[:k],relevance_s+stability_s,None),('relevance_correlation',rc,relevance_s+rc_prune_s,rc_max),('full_sdcs',full,relevance_s+stability_s+full_prune_s,full_max)]
        # Fixed seeded run order avoids a systematic warm-up advantage.
        order=np.random.RandomState(20261026+seed).permutation(len(variants));seed_rows=[]
        for position in order:
            method,chosen,selection_s,selection_max=variants[position];reserve(args.base);assert len(set(chosen))==k
            target=folder/method;target.mkdir(exist_ok=False);pd.DataFrame({'feature_index':chosen,'feature_name':[names[i] for i in chosen]}).to_csv(target/'selected_features.csv',index=False)
            scaler=StandardScaler();start=time.perf_counter();xtr=scaler.fit_transform(x_train[:,chosen]);xte=scaler.transform(x_test[:,chosen]);scaling_s=time.perf_counter()-start
            model=FocalLossLGBM(gamma=.30,alpha=.95,boosting_type='gbdt',num_leaves=50,max_depth=-1,learning_rate=.15,n_estimators=500,min_child_samples=40,reg_alpha=.1,reg_lambda=.3,verbosity=-1,random_state=42,device='cpu',n_jobs=args.jobs)
            start=time.perf_counter();model.fit(xtr,y_train);fit_s=time.perf_counter()-start;p=np.asarray(model.predict_proba(xte)[:,1]);assert np.isfinite(p).all()
            reference_error=None
            if method=='full_sdcs':
                old=pd.read_csv(anchor/'selectors_full'/('seed_'+str(seed)+'_predictions.csv.gz'));old=old.loc[old.method.eq('sdcs')].sort_values('row_id')
                assert np.array_equal(old.row_id,ix['test']) and np.array_equal(old.label,y_test)
                reference_error=float(np.max(np.abs(old.probability.to_numpy()-p)));assert reference_error<1e-10,'Full SDCS probabilities differ from frozen reference'
            pd.DataFrame({'row_id':ix['test'],'label':y_test,'probability':p,'prediction':(p>=.5).astype(int)}).to_csv(target/'test_predictions.csv.gz',index=False,compression='gzip')
            model.booster_.save_model(str(target/'booster.txt'));joblib.dump(scaler,target/'scaler.joblib')
            row=dict(seed=seed,method=method,feature_count=k,stability_seconds=stability_s if 'stability' in method or method=='full_sdcs' else 0.,relevance_seconds=relevance_s,selection_seconds=selection_s,scaling_seconds=scaling_s,model_fit_seconds=fit_s,selection_population_max_abs_correlation=selection_max,reference_max_probability_error=reference_error,**metrics(y_test,p),**redundancy(x_train,chosen,seed))
            dump(target/'metrics.json',row);all_rows.append(row);seed_rows.append(row);pd.DataFrame(all_rows).to_csv(args.output/'per_seed_metrics.csv',index=False)
            logging.info('Completed seed %s %s F1 %.6f',seed,method,row['f1']);del xtr,xte,model,scaler
        np.savez(folder/'selection_scores.npz',stability=frequency,relevance=hybrid,composite=composite)
        dump(folder/'complete.json',dict(seed=seed,budget=k,conditions=4,full_sdcs_reproduced=True));del xs
    dump(args.output/'complete.json',dict(status='COMPONENT_FITS_COMPLETE',seeds=args.seeds,models=len(all_rows),source_dataset_sha256=sha(args.dataset),manifest_sha256=sha(manifest_path),source_code_sha256=sha(Path(__file__))))
if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise
