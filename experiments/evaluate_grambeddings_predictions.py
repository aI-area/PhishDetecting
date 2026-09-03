#!/usr/bin/env python3
"""Inference-only evaluation of a saved GramBeddings shared-split artifact."""
from __future__ import annotations
import argparse, gzip, hashlib, json, os, pickle, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('input_csv',type=Path); p.add_argument('artifact_dir',type=Path)
    p.add_argument('baseline_dir',type=Path); p.add_argument('output_dir',type=Path); p.add_argument('--batch-size',type=int,default=512)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(Path(__file__).parent))
    import train_grambeddings_shared_split as runner
    NBeddingModel,*_=runner.load_baseline(a.baseline_dir)
    with gzip.open(a.artifact_dir/'PhishFusion_preprocessing.pkl.gz','rb') as f: transformers=pickle.load(f)
    frame=pd.read_csv(a.input_csv); urls=frame.url.astype(str)
    arrays=[np.asarray(t.Transform(urls),dtype=np.float32) for t in transformers]
    config=SimpleNamespace(max_seq_len=128,char_embedding_dim=95,embedding_dim=15,rnn_cell_size=128,attention_width=10)
    specs=[(97,95,None,'embed_char')]+[(160002,15,None,f'embed_ngram_{i}') for i in range(1,4)]
    model=runner.build_model(config,specs,NBeddingModel); weights=a.artifact_dir/'PhishFusion_best.weights.h5'; model.load_weights(str(weights))
    prob=model.predict(arrays,batch_size=a.batch_size,verbose=2).reshape(-1); pred=(prob>=.5).astype(int)
    out=frame[['url','phishing']].copy(); out['prediction']=pred; out['probability']=prob
    out_path=a.output_dir/'GramBeddings_results.csv'; out.to_csv(out_path,index=False)
    audit={'status':'PASS','retraining_performed':False,'n':len(out),'input_sha256':sha256(a.input_csv),
           'weights_sha256':sha256(weights),'preprocessing_sha256':sha256(a.artifact_dir/'PhishFusion_preprocessing.pkl.gz'),
           'predictions_sha256':sha256(out_path)}
    (a.output_dir/'evaluation_audit.json').write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2))
if __name__=='__main__':
    os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH','true'); main()
