"""Train DEPHIDES on source train/validation rows and evaluate a clean target cohort."""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, random
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import accuracy_score,average_precision_score,confusion_matrix,f1_score,matthews_corrcoef,precision_score,recall_score,roc_auc_score
from tensorflow.keras.layers import Conv1D,Dense,Dropout,Embedding,Flatten,Input,MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def seed_all(s): os.environ['PYTHONHASHSEED']=str(s);random.seed(s);np.random.seed(s);tf.random.set_seed(s)
def data(p):
 d=pd.read_csv(p,encoding='ISO-8859-1',usecols=['url','phishing']).reset_index(drop=True);d.url=d.url.fillna('').astype(str);d.phishing=d.phishing.astype(int);return d
def source(data,p):
 m=pd.read_csv(p)
 if m.row_id.duplicated().any() or not m.row_id.between(0,len(data)-1).all():raise ValueError('Invalid source row_id')
 ids=m.row_id.astype(int).to_numpy()
 if not np.array_equal(data.iloc[ids].phishing.to_numpy(),m.label.astype(int).to_numpy()):raise ValueError('Source labels disagree')
 z={}
 for s in ('train','validation'):
  z[s]=np.zeros(len(data),dtype=bool);z[s][m.loc[m.split.eq(s),'row_id'].astype(int)]=True
 if any(not x.any() for x in z.values()):raise ValueError('Empty source split')
 return m,z
def target(data,p):
 m=pd.read_csv(p)
 if not {'row_id','label'}<=set(m):raise ValueError('Target manifest requires row_id,label')
 if m.row_id.duplicated().any() or not m.row_id.between(0,len(data)-1).all():raise ValueError('Invalid target row_id')
 if 'include' in m:m=m[m.include.astype(bool)]
 if 'split' in m:
  q=m.split.astype(str).str.lower().isin(['test','target_test'])
  if q.any():m=m[q]
 ids=m.row_id.astype(int).to_numpy()
 if not np.array_equal(data.iloc[ids].phishing.to_numpy(),m.label.astype(int).to_numpy()):raise ValueError('Target labels disagree')
 return m.reset_index(drop=True),ids
def metric(y,p,q):
 tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel();return {'n':int(len(y)),'benign':int((y==0).sum()),'phishing':int((y==1).sum()),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'accuracy':float(accuracy_score(y,p)),'precision':float(precision_score(y,p,pos_label=1,zero_division=0)),'recall':float(recall_score(y,p,pos_label=1,zero_division=0)),'f1':float(f1_score(y,p,pos_label=1,zero_division=0)),'roc_auc':float(roc_auc_score(y,q)),'average_precision':float(average_precision_score(y,q)),'mcc':float(matthews_corrcoef(y,p))}
def model(v,l):
 m=Sequential([Input((l,),dtype='int32'),Embedding(v+1,50),Conv1D(128,3,activation='tanh'),MaxPooling1D(3),Dropout(.2),Conv1D(128,7,activation='tanh',padding='same'),Dropout(.2),Conv1D(128,5,activation='tanh',padding='same'),Dropout(.2),Conv1D(128,3,activation='tanh',padding='same'),MaxPooling1D(3),Dropout(.2),Conv1D(128,5,activation='tanh',padding='same'),Dropout(.2),Conv1D(128,3,activation='tanh',padding='same'),MaxPooling1D(3),Dropout(.2),Conv1D(128,3,activation='tanh',padding='same'),MaxPooling1D(3),Dropout(.2),Flatten(),Dense(1,activation='sigmoid')]);m.compile(loss='binary_crossentropy',optimizer=Adam(.001),metrics=['accuracy']);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('source_dataset',type=Path);p.add_argument('source_manifest',type=Path);p.add_argument('target_dataset',type=Path);p.add_argument('target_manifest',type=Path);p.add_argument('output_dir',type=Path);p.add_argument('--name',required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--sequence-length',type=int,default=200);p.add_argument('--seed',type=int,default=42);a=p.parse_args();seed_all(a.seed)
 sd=data(a.source_dataset);td=data(a.target_dataset);sm,ms=source(sd,a.source_manifest);tm,ids=target(td,a.target_manifest);over=set(sd.loc[ms['train']|ms['validation'],'url'])&set(td.iloc[ids].url)
 if over:raise ValueError(f'Target overlaps source by {len(over)} exact URLs')
 tok=Tokenizer(lower=True,char_level=True,oov_token='-n-');tok.fit_on_texts(sd.loc[ms['train'],'url'].tolist());enc=lambda x:pad_sequences(tok.texts_to_sequences(x),maxlen=a.sequence_length,padding='post',truncating='post')
 X={'train':enc(sd.loc[ms['train'],'url'].tolist()),'validation':enc(sd.loc[ms['validation'],'url'].tolist()),'target_test':enc(td.iloc[ids].url.tolist())};y={'train':sd.loc[ms['train'],'phishing'].to_numpy(),'validation':sd.loc[ms['validation'],'phishing'].to_numpy(),'target_test':td.iloc[ids].phishing.to_numpy()};m=model(len(tok.word_index),a.sequence_length);h=m.fit(X['train'],y['train'],validation_data=(X['validation'],y['validation']),epochs=a.epochs,batch_size=a.batch_size,verbose=2)
 a.output_dir.mkdir(parents=True,exist_ok=True);mp=a.output_dir/f'{a.name}_model.h5';tp=a.output_dir/f'{a.name}_tokenizer.pickle';m.save(mp)
 with tp.open('wb') as f:pickle.dump(tok,f,pickle.HIGHEST_PROTOCOL)
 pd.DataFrame(h.history).to_csv(a.output_dir/f'{a.name}_history.csv',index=False);summ={}
 for split,rows,rids in [('validation',sd.loc[ms['validation']],np.flatnonzero(ms['validation'])),('target_test',td.iloc[ids],ids)]:
  q=m.predict(X[split],batch_size=a.batch_size,verbose=0).reshape(-1);pred=(q>=.5).astype(int);summ[split]=metric(y[split],pred,q);o=rows[['url','phishing']].copy();o.insert(0,'row_id',rids);o['prediction']=pred;o['probability']=q;o.to_csv(a.output_dir/f'{a.name}_{split}_predictions.csv',index=False)
 rec={'baseline':'DEPHIDES','source_dataset':str(a.source_dataset.resolve()),'source_dataset_sha256':sha256(a.source_dataset),'source_manifest':str(a.source_manifest.resolve()),'source_manifest_sha256':sha256(a.source_manifest),'target_dataset':str(a.target_dataset.resolve()),'target_dataset_sha256':sha256(a.target_dataset),'target_manifest':str(a.target_manifest.resolve()),'target_manifest_sha256':sha256(a.target_manifest),'model_sha256':sha256(mp),'tokenizer_sha256':sha256(tp),'preprocessing_fit':'source train only','validation_role':'source training-curve monitoring','positive_class':{'value':1,'name':'phishing'},'decision_threshold':.5,'seed':a.seed,'metrics':summ};(a.output_dir/f'{a.name}_metrics.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
