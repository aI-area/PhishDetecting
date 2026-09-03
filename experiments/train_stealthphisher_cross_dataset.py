"""Train StealthPhisher on source rows and evaluate a duplicate-free target cohort."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,random
from pathlib import Path
import joblib,numpy as np,pandas as pd,tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,average_precision_score,confusion_matrix,f1_score,matthews_corrcoef,precision_score,recall_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Concatenate,Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load_module(p):
 s=importlib.util.spec_from_file_location('stealth_transfer_features',p)
 if s is None or s.loader is None:raise ImportError(p)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def seed_all(s):os.environ['PYTHONHASHSEED']=str(s);random.seed(s);np.random.seed(s);tf.random.set_seed(s)
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
def select_features(X,y,names,threshold=.015):
 valid=[c for c in names if X[c].std()>0]
 if not valid:raise ValueError('All source-training features are constant')
 def class_variance(frame):
  if not len(frame):return pd.Series(0.,index=valid)
  scaled=StandardScaler().fit_transform(frame[valid]);ratio=PCA(n_components=min(len(frame),len(valid))).fit(scaled).explained_variance_ratio_[0];return pd.Series(ratio,index=valid)
 score=(class_variance(X[y==0])+class_variance(X[y==1]))/2;chosen=score[score>threshold].index.tolist();return chosen or valid
def build_model(n):
 wide=Input((n,),name='wide_input');wo=Dense(1,activation='sigmoid')(wide);deep=Input((n,),name='deep_input');d=Dense(64,activation='relu')(deep);d=Dense(32,activation='relu')(d);do=Dense(1,activation='sigmoid')(d);out=Dense(1,activation='sigmoid')(Concatenate()([wo,do]));m=Model([wide,deep],out);m.compile(optimizer=Adam(.001),loss='binary_crossentropy',metrics=['accuracy']);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('source_dataset',type=Path);p.add_argument('source_manifest',type=Path);p.add_argument('target_dataset',type=Path);p.add_argument('target_manifest',type=Path);p.add_argument('feature_code',type=Path);p.add_argument('output_dir',type=Path);p.add_argument('--name',required=True);p.add_argument('--epochs',type=int,default=50);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--workers',type=int,default=50);p.add_argument('--seed',type=int,default=42);a=p.parse_args();seed_all(a.seed)
 sd=data(a.source_dataset);td=data(a.target_dataset);sm,ms=source(sd,a.source_manifest);tm,ids=target(td,a.target_manifest);over=set(sd.loc[ms['train']|ms['validation'],'url'])&set(td.iloc[ids].url)
 if over:raise ValueError(f'Target overlaps source by {len(over)} exact URLs')
 feature=load_module(a.feature_code);source_rows=sd.loc[ms['train']|ms['validation']].copy();source_rows['_original_row_id']=source_rows.index;target_data=td.iloc[ids].copy();target_data['_original_row_id']=ids
 sf=feature.extract_features_parallel(source_rows,'url',a.workers);tfet=feature.extract_features_parallel(target_data,'url',a.workers);names=[c for c in feature.StealthPhisherExtractor().feature_columns if c in sf.columns];train_mask=sf._original_row_id.isin(np.flatnonzero(ms['train']));val_mask=sf._original_row_id.isin(np.flatnonzero(ms['validation']));selected=select_features(sf.loc[train_mask,names],sf.loc[train_mask,'phishing'].to_numpy(),names)
 scaler=StandardScaler().fit(sf.loc[train_mask,selected]);Xtr=scaler.transform(sf.loc[train_mask,selected]);Xv=scaler.transform(sf.loc[val_mask,selected]);Xt=scaler.transform(tfet[selected]);ytr=sf.loc[train_mask,'phishing'].to_numpy();yv=sf.loc[val_mask,'phishing'].to_numpy();yt=tfet.phishing.to_numpy();model=build_model(len(selected));hist=model.fit([Xtr,Xtr],ytr,validation_data=([Xv,Xv],yv),epochs=a.epochs,batch_size=a.batch_size,verbose=2,callbacks=[EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)])
 a.output_dir.mkdir(parents=True,exist_ok=True);mp=a.output_dir/f'{a.name}_model.h5';sp=a.output_dir/f'{a.name}_scaler.joblib';fp=a.output_dir/f'{a.name}_selected_features.json';model.save(mp);joblib.dump(scaler,sp);fp.write_text(json.dumps(selected,indent=2));pd.DataFrame(hist.history).to_csv(a.output_dir/f'{a.name}_history.csv',index=False);summ={}
 for split,X,y,rows,rids in [('validation',Xv,yv,sd.loc[ms['validation']],np.flatnonzero(ms['validation'])),('target_test',Xt,yt,td.iloc[ids],ids)]:
  q=model.predict([X,X],batch_size=a.batch_size,verbose=0).reshape(-1);pred=(q>=.5).astype(int);summ[split]=metric(y,pred,q);o=rows[['url','phishing']].copy();o.insert(0,'row_id',rids);o['prediction']=pred;o['probability']=q;o.to_csv(a.output_dir/f'{a.name}_{split}_predictions.csv',index=False)
 rec={'baseline':'StealthPhisher','source_dataset':str(a.source_dataset.resolve()),'source_dataset_sha256':sha256(a.source_dataset),'source_manifest':str(a.source_manifest.resolve()),'source_manifest_sha256':sha256(a.source_manifest),'target_dataset':str(a.target_dataset.resolve()),'target_dataset_sha256':sha256(a.target_dataset),'target_manifest':str(a.target_manifest.resolve()),'target_manifest_sha256':sha256(a.target_manifest),'feature_code_sha256':sha256(a.feature_code),'model_sha256':sha256(mp),'scaler_sha256':sha256(sp),'selected_features_sha256':sha256(fp),'selected_features':selected,'preprocessing_fit':'source train only','feature_mode':'released offline lexical extractor','validation_role':'source early stopping','positive_class':{'value':1,'name':'phishing'},'decision_threshold':.5,'seed':a.seed,'metrics':summ};(a.output_dir/f'{a.name}_metrics.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
