#!/usr/bin/env python3
"""Evaluate a validated LitePhish source model on its held-out source test split."""
from __future__ import annotations
import argparse, hashlib, json, pickle, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('dataset',type=Path); p.add_argument('manifest',type=Path)
    p.add_argument('artifact_dir',type=Path); p.add_argument('pipeline_root',type=Path); p.add_argument('output_dir',type=Path)
    p.add_argument('--batch-size',type=int,default=500); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(a.pipeline_root),str(a.pipeline_root/'experiments')]
    from feature_extraction import FeatureExtractor
    from experiments.run_litephish_experiments import predict_probabilities
    audit=json.load((a.artifact_dir/'LitePhish_audit.json').open())
    trained_manifest=pd.read_csv(audit['source_manifest']); current=pd.read_csv(a.manifest)
    for frame in (trained_manifest,current):
        frame['split']=frame['split'].replace({'val':'validation'})
    for split in ('train','validation'):
        old=trained_manifest.loc[trained_manifest.split.eq(split),['row_id','label']].sort_values('row_id').reset_index(drop=True)
        new=current.loc[current.split.eq(split),['row_id','label']].sort_values('row_id').reset_index(drop=True)
        if not old.equals(new): raise RuntimeError(f'{split} rows differ from saved model training provenance')
    data=pd.read_csv(a.dataset); url_col='url' if 'url' in data else 'URL'; label_col='phishing' if 'phishing' in data else 'label'
    test_manifest=current[current.split.eq('test')].copy(); ids=test_manifest.row_id.astype(int).to_numpy()
    test=pd.DataFrame({'row_id':ids,'url':data.iloc[ids][url_col].astype(str).to_numpy(),'phishing':data.iloc[ids][label_col].astype(int).to_numpy()})
    if not np.array_equal(test.phishing,test_manifest.label.astype(int)): raise RuntimeError('manifest labels differ from dataset')
    model=joblib.load(a.artifact_dir/'LitePhish_model.joblib'); scaler=joblib.load(a.artifact_dir/'LitePhish_scaler.joblib')
    with (a.artifact_dir/'LitePhish_preprocessing.pkl').open('rb') as f: prep=pickle.load(f)
    processor=prep['ngram_processor']; selected=np.asarray(prep['selected_indices'],int); extractor=FeatureExtractor(); probs=[]
    for start in range(0,len(test),a.batch_size):
        urls=test.url.iloc[start:start+a.batch_size]
        hand=np.asarray([extractor.generate_features(u) for u in urls],np.float32)
        grams=processor.transform(urls).toarray().astype(np.float32,copy=False)
        x=np.hstack((hand,grams))[:,selected]
        probs.append(np.asarray(predict_probabilities(model,scaler.transform(x)),float))
    prob=np.concatenate(probs); test['prediction']=(prob>=.5).astype(int); test['probability']=prob
    out=a.output_dir/'LitePhish_predictions.csv'; test.to_csv(out,index=False)
    record={'status':'PASS','retraining_performed':False,'source_artifact_dir':str(a.artifact_dir.resolve()),
            'saved_model_training_rows_verified_against_manifest':True,'dataset_sha256':sha256(a.dataset),'manifest_sha256':sha256(a.manifest),
            'n':len(test),'selected_features':len(selected),'predictions_sha256':sha256(out)}
    (a.output_dir/'evaluation_audit.json').write_text(json.dumps(record,indent=2)); print(json.dumps(record,indent=2))
if __name__=='__main__': main()
