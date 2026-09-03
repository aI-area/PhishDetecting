"""Train StealthPhisher from a row-aligned feature cache and shared manifest."""

from __future__ import annotations

import argparse, hashlib, json, random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Concatenate, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def build_model(n: int) -> Model:
    wide=Input((n,),name='wide_input'); wide_out=Dense(1,activation='sigmoid')(wide)
    deep=Input((n,),name='deep_input'); x=Dense(64,activation='relu')(deep); x=Dense(32,activation='relu')(x)
    deep_out=Dense(1,activation='sigmoid')(x)
    out=Dense(1,activation='sigmoid')(Concatenate()([wide_out,deep_out]))
    m=Model([wide,deep],out); m.compile(optimizer=Adam(0.001),loss='binary_crossentropy',metrics=['accuracy'])
    return m


def select_cspca(x: pd.DataFrame, y: np.ndarray, threshold: float=0.015):
    valid=[c for c in x.columns if x[c].std()>0]
    scaler=StandardScaler()
    def first_variance(z):
        if len(z)==0:return 0.0
        a=scaler.fit_transform(z[valid]); p=PCA(n_components=min(len(a),len(valid))).fit(a)
        return float(p.explained_variance_ratio_[0])
    score=(first_variance(x.loc[y==0])+first_variance(x.loc[y==1]))/2
    return valid if score>threshold else valid


def evaluate(y,pred,prob):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'n':len(y),'benign':int((y==0).sum()),'phishing':int((y==1).sum()),
      'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'accuracy':accuracy_score(y,pred),
      'precision':precision_score(y,pred,pos_label=1,zero_division=0),'recall':recall_score(y,pred,pos_label=1,zero_division=0),
      'f1':f1_score(y,pred,pos_label=1,zero_division=0),'roc_auc':roc_auc_score(y,prob),
      'average_precision':average_precision_score(y,prob),'mcc':matthews_corrcoef(y,pred)}


def main():
    p=argparse.ArgumentParser(); p.add_argument('dataset',type=Path); p.add_argument('manifest',type=Path)
    p.add_argument('feature_cache',type=Path); p.add_argument('output_dir',type=Path); p.add_argument('--name',required=True)
    a=p.parse_args(); random.seed(42); np.random.seed(42); tf.random.set_seed(42)
    for g in tf.config.list_physical_devices('GPU'):
        try: tf.config.experimental.set_memory_growth(g,True)
        except RuntimeError: pass
    d=pd.read_csv(a.dataset,encoding='ISO-8859-1',usecols=['url','phishing']).reset_index(drop=True)
    f=pd.read_csv(a.feature_cache); m=pd.read_csv(a.manifest)
    if len(f)!=len(d) or not d.url.astype(str).equals(f.url.astype(str)) or not d.phishing.astype(int).equals(f.phishing.astype(int)):
        raise ValueError('Feature cache is not row-aligned with source dataset')
    if m.row_id.tolist()!=list(range(len(d))) or not m.label.astype(int).equals(d.phishing.astype(int)):
        raise ValueError('Manifest mismatch')
    exclude={'url','phishing','probability','prediction'}; cols=[c for c in f.columns if c not in exclude and pd.api.types.is_numeric_dtype(f[c])]
    x=f[cols].fillna(0); y=d.phishing.astype(int).to_numpy(); masks={s:m.split.eq(s).to_numpy() for s in ('train','validation','test')}
    selected=select_cspca(x.loc[masks['train']],y[masks['train']]); scaler=StandardScaler().fit(x.loc[masks['train'],selected])
    xs={s:scaler.transform(x.loc[masks[s],selected]) for s in masks}; a.output_dir.mkdir(parents=True,exist_ok=True)
    model_path=a.output_dir/f'{a.name}_model.h5'; model=build_model(len(selected))
    model.fit([xs['train'],xs['train']],y[masks['train']],validation_data=([xs['validation'],xs['validation']],y[masks['validation']]),
      epochs=50,batch_size=64,callbacks=[EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True),ModelCheckpoint(model_path,save_best_only=True,verbose=0)],verbose=2)
    summaries={}
    for s in ('validation','test'):
        prob=model.predict([xs[s],xs[s]],batch_size=1024,verbose=0).reshape(-1); pred=(prob>=0.5).astype(int); summaries[s]=evaluate(y[masks[s]],pred,prob)
        rows=d.loc[masks[s],['url','phishing']].copy(); rows.insert(0,'row_id',m.loc[masks[s],'row_id'].to_numpy()); rows['prediction']=pred; rows['probability']=prob
        rows.to_csv(a.output_dir/f'{a.name}_{s}_predictions.csv',index=False)
    scaler_path=a.output_dir/f'{a.name}_scaler.joblib'; joblib.dump(scaler,scaler_path)
    rec={'baseline':'StealthPhisher','dataset':str(a.dataset.resolve()),'dataset_sha256':sha256(a.dataset),'manifest_sha256':sha256(a.manifest),
      'feature_cache_sha256':sha256(a.feature_cache),'model_sha256':sha256(model_path),'scaler_sha256':sha256(scaler_path),'selected_features':selected,
      'preprocessing_fit':'train only','feature_selection_fit':'train only','positive_class':{'value':1,'name':'phishing'},'decision_threshold':0.5,
      'validation_role':'early stopping only','metrics':summaries}
    (a.output_dir/f'{a.name}_metrics.json').write_text(json.dumps(rec,indent=2),encoding='utf-8'); print(json.dumps(rec,indent=2))


if __name__=='__main__': main()
