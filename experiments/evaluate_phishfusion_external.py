"""Evaluate a fitted baseline on the complete PhishFusion target without refitting."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,pickle
from pathlib import Path
import joblib,numpy as np,pandas as pd,tensorflow as tf
from sklearn.metrics import accuracy_score,average_precision_score,confusion_matrix,f1_score,matthews_corrcoef,precision_score,recall_score,roc_auc_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def target_order_hash(d):
 h=hashlib.sha256()
 for i,(u,y) in enumerate(zip(d.url,d.phishing)):
  b=str(u).encode('utf-8','surrogatepass');h.update(i.to_bytes(8,'little'));h.update(len(b).to_bytes(8,'little'));h.update(b);h.update(int(y).to_bytes(1,'little'))
 return h.hexdigest()
def module(p):
 s=importlib.util.spec_from_file_location('full_pf_features',p)
 if s is None or s.loader is None:raise ImportError(p)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def metric(y,p,q):
 tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel();return {'n':int(len(y)),'benign':int((y==0).sum()),'phishing':int((y==1).sum()),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'accuracy':float(accuracy_score(y,p)),'precision':float(precision_score(y,p,pos_label=1,zero_division=0)),'recall':float(recall_score(y,p,pos_label=1,zero_division=0)),'f1':float(f1_score(y,p,pos_label=1,zero_division=0)),'roc_auc':float(roc_auc_score(y,q)),'average_precision':float(average_precision_score(y,q)),'mcc':float(matthews_corrcoef(y,p))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',choices=['cnn_fusion','deephides','stealthphisher'],required=True);p.add_argument('--source-name',required=True);p.add_argument('--target',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--preprocessor',type=Path);p.add_argument('--training-audit',type=Path);p.add_argument('--feature-code',type=Path);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--batch-size',type=int,default=512);p.add_argument('--workers',type=int,default=24);a=p.parse_args()
 d=pd.read_csv(a.target,encoding='ISO-8859-1',usecols=['url','phishing']).reset_index(drop=True);d.url=d.url.fillna('').astype(str);d.phishing=d.phishing.astype(int);y=d.phishing.to_numpy();model=tf.keras.models.load_model(a.model,compile=False);extra={}
 if a.baseline=='cnn_fusion':
  if not a.preprocessor:raise ValueError('CNN-Fusion requires vocabulary')
  with a.preprocessor.open('rb') as f:v=pickle.load(f)
  X=pad_sequences([[v[c] for c in u if c in v] for u in d.url],maxlen=150,padding='post',truncating='post');q=model.predict(X,batch_size=a.batch_size,verbose=0).reshape(-1)
 elif a.baseline=='deephides':
  if not a.preprocessor:raise ValueError('DEPHIDES requires tokenizer')
  with a.preprocessor.open('rb') as f:t=pickle.load(f)
  X=pad_sequences(t.texts_to_sequences(d.url.tolist()),maxlen=200,padding='post',truncating='post');q=model.predict(X,batch_size=a.batch_size,verbose=0).reshape(-1)
 else:
  if not (a.preprocessor and a.training_audit and a.feature_code):raise ValueError('StealthPhisher requires scaler, training audit and feature code')
  audit=json.loads(a.training_audit.read_text());selected=audit['selected_features'];features=module(a.feature_code).extract_features_parallel(d.copy(),'url',a.workers);X=joblib.load(a.preprocessor).transform(features[selected]);q=model.predict([X,X],batch_size=a.batch_size,verbose=0).reshape(-1);extra={'feature_code':str(a.feature_code.resolve()),'feature_code_sha256':sha256(a.feature_code),'selected_features':selected}
 pred=(q>=.5).astype(int);a.output_dir.mkdir(parents=True,exist_ok=True);out=d[['url','phishing']].copy();out.insert(0,'row_id',np.arange(len(d)));out['prediction']=pred;out['probability']=q;pp=a.output_dir/f'{a.source_name}_to_full_PhishFusion_predictions.csv';out.to_csv(pp,index=False)
 rec={'baseline':a.baseline,'source_name':a.source_name,'evaluation_mode':'inference only; no refitting or target tuning','target':str(a.target.resolve()),'target_sha256':sha256(a.target),'target_rows':len(d),'target_order_sha256':target_order_hash(d),'target_label_sha256':hashlib.sha256(y.astype(np.int8).tobytes()).hexdigest(),'model':str(a.model.resolve()),'model_sha256':sha256(a.model),'preprocessor':str(a.preprocessor.resolve()) if a.preprocessor else None,'preprocessor_sha256':sha256(a.preprocessor) if a.preprocessor else None,'training_audit':str(a.training_audit.resolve()) if a.training_audit else None,'training_audit_sha256':sha256(a.training_audit) if a.training_audit else None,'predictions_sha256':sha256(pp),'positive_class':{'value':1,'name':'phishing'},'decision_threshold':.5,'metrics':metric(y,pred,q),**extra};ap=a.output_dir/f'{a.source_name}_to_full_PhishFusion_audit.json';ap.write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
