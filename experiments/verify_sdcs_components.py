"""Check component ablation bindings, budgets, saved scalers and predictions."""
import argparse,hashlib,json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd,lightgbm as lgb
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,matthews_corrcoef
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--anchor',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--pipeline-root',type=Path,default=Path(__file__).resolve().parents[1]/'artifacts/phreshphish/source_definitions');a=ap.parse_args()
    assert not (a.output/'verification.json').exists();complete=json.loads((a.output/'complete.json').read_text());assert complete['status']=='COMPONENT_FITS_COMPLETE'
    assert sha(a.dataset)==complete['source_dataset_sha256'];assert sha(a.output/'runner_snapshot.py')==complete['source_code_sha256']
    sys.path.insert(0,str(a.pipeline_root))
    from feature_extraction import FeatureExtractor
    from ngram_processing import NgramProcessor
    cache=a.base/'candidate_cache';metadata=json.loads((cache/'complete.json').read_text())
    for name,h in metadata['files'].items():assert sha(cache/name)==h
    arrays=np.load(cache/'partitions.npz');train=np.load(cache/'x_train.npy',mmap_mode='r');test=np.load(cache/'x_test.npy',mmap_mode='r');names=json.loads((cache/'feature_names.json').read_text());labels=arrays['y_test'];test_ids=arrays['test_indices']
    assert len(labels)==11376 and len(arrays['train_indices'])==86098
    # Independently rebuild a fixed, score-independent feature-cache sample.
    dataset=pd.read_csv(a.dataset,encoding='latin1');extractor=FeatureExtractor();processor=joblib.load(a.anchor/'selectors_full/ngram_processor.joblib')
    for matrix,part in [(train,'train'),(test,'test')]:
        pos=np.random.RandomState(20261026).choice(len(matrix),20,replace=False);urls=dataset.url.iloc[arrays[part+'_indices'][pos]].astype(str)
        hc=np.asarray([extractor.generate_features(u) for u in urls],dtype=np.float32);ng=processor.transform(urls).toarray().astype(np.float32,copy=False);fresh=np.hstack([hc,ng]);np.testing.assert_allclose(fresh,matrix[pos],rtol=0,atol=0)
    rows=pd.read_csv(a.output/'per_seed_metrics.csv');assert len(rows)==complete['models']==4*len(complete['seeds']);maximum=0.
    for seed in complete['seeds']:
        folder=a.output/('seed_'+str(seed));state=json.loads((folder/'complete.json').read_text());k=state['budget'];subset=np.load(folder/'selector_training_positions.npy')
        expected=np.sort(np.random.RandomState(seed).choice(len(train),int(.905*len(train)),replace=False));assert np.array_equal(subset,expected)
        scores=np.load(folder/'selection_scores.npz');np.testing.assert_allclose(scores['composite'],.767*scores['stability']+.233*scores['relevance'],rtol=0,atol=1e-15)
        for method in ['relevance_only','relevance_stability','relevance_correlation','full_sdcs']:
            target=folder/method;chosen=pd.read_csv(target/'selected_features.csv',keep_default_na=False);ids=chosen.feature_index.to_numpy(dtype=int)
            assert len(ids)==len(set(ids))==k and chosen.feature_name.tolist()==[names[i] for i in ids]
            if method=='relevance_only':assert np.array_equal(ids,np.argsort(scores['relevance'])[::-1][:k])
            if method=='relevance_stability':
                eligible=np.where(scores['stability']>=.705)[0];expected=eligible[np.argsort(scores['composite'][eligible])[::-1]][:k];assert np.array_equal(ids,expected)
            if method in ['relevance_correlation','full_sdcs']:
                c=np.abs(np.corrcoef(train[subset][:,ids],rowvar=False));upper=c[np.triu_indices_from(c,k=1)];assert upper.max()<=.956+1e-12
            if method=='full_sdcs':
                frozen=pd.read_csv(a.anchor/'selectors_full/selected_features'/('seed_'+str(seed))/'sdcs.csv',keep_default_na=False);assert np.array_equal(ids,frozen.feature_index)
            scaler=joblib.load(target/'scaler.joblib');x=np.array(train[:,ids]);np.testing.assert_allclose(scaler.mean_,x.mean(axis=0,dtype=np.float64),rtol=1e-10,atol=1e-10);del x
            frame=pd.read_csv(target/'test_predictions.csv.gz');assert np.array_equal(frame.row_id,test_ids) and np.array_equal(frame.label,labels)
            model=lgb.Booster(model_file=str(target/'booster.txt'));raw=model.predict(scaler.transform(test[:,ids]),num_threads=8);prob=1/(1+np.exp(-raw));error=float(np.max(np.abs(prob-frame.probability)));maximum=max(maximum,error);assert error<1e-10
            pred=(frame.probability.to_numpy()>=.5).astype(int);assert np.array_equal(pred,frame.prediction)
            metrics=dict(accuracy=accuracy_score(labels,pred),precision=precision_score(labels,pred),recall=recall_score(labels,pred),f1=f1_score(labels,pred),roc_auc=roc_auc_score(labels,frame.probability),mcc=matthews_corrcoef(labels,pred));record=rows.loc[(rows.seed==seed)&(rows.method==method)].iloc[0]
            for key,value in metrics.items():assert abs(value-record[key])<1e-10
    report=dict(status='COMPONENT_ABLATION_VERIFIED',seeds=complete['seeds'],models=len(rows),prediction_rows=len(rows)*len(labels),raw_feature_spot_checks=40,maximum_probability_error=maximum,full_sdcs_reference_reproduced=True)
    (a.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
