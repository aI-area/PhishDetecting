"""Train CNN-Fusion on source train/validation rows and evaluate a clean target cohort."""

from __future__ import annotations

import argparse, hashlib, json, os, pickle, random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Concatenate, Conv1D, Dense, Dropout, Embedding, GlobalMaxPooling1D, Input, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""): h.update(b)
    return h.hexdigest()


def seed_all(seed):
    os.environ["PYTHONHASHSEED"] = str(seed); random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def read_data(path):
    d = pd.read_csv(path, encoding="ISO-8859-1", usecols=["url", "phishing"]).reset_index(drop=True)
    d["url"] = d.url.fillna("").astype(str); d["phishing"] = d.phishing.astype(int)
    return d


def source_masks(data, path):
    m = pd.read_csv(path)
    if m.row_id.duplicated().any() or not m.row_id.between(0, len(data)-1).all(): raise ValueError("Invalid source row_id")
    ids = m.row_id.astype(int).to_numpy()
    if not np.array_equal(data.iloc[ids].phishing.to_numpy(), m.label.astype(int).to_numpy()): raise ValueError("Source labels disagree with manifest")
    out = {}
    for s in ("train", "validation"):
        out[s] = np.zeros(len(data), dtype=bool); out[s][m.loc[m.split.eq(s), "row_id"].astype(int)] = True
    if any(not x.any() for x in out.values()): raise ValueError("Source train and validation must be non-empty")
    return m, out


def target_rows(data, path):
    m = pd.read_csv(path)
    for c in ("row_id", "label"):
        if c not in m: raise ValueError(f"Target manifest lacks {c}")
    if m.row_id.duplicated().any(): raise ValueError("Duplicate row_id values in target manifest")
    if not m.row_id.between(0, len(data)-1).all(): raise ValueError("Target row_id out of range")
    if "include" in m: m = m[m.include.astype(bool)]
    if "split" in m:
        keep = m.split.astype(str).str.lower().isin(["test", "target_test"])
        if keep.any(): m = m[keep]
    ids = m.row_id.astype(int).to_numpy()
    if not np.array_equal(data.iloc[ids].phishing.to_numpy(), m.label.astype(int).to_numpy()): raise ValueError("Target labels disagree with manifest")
    return m.reset_index(drop=True), ids


def metrics(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {"n":int(len(y)),"benign":int((y==0).sum()),"phishing":int((y==1).sum()),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,pos_label=1,zero_division=0)),"recall":float(recall_score(y,pred,pos_label=1,zero_division=0)),"f1":float(f1_score(y,pred,pos_label=1,zero_division=0)),"roc_auc":float(roc_auc_score(y,prob)),"average_precision":float(average_precision_score(y,prob)),"mcc":float(matthews_corrcoef(y,pred))}


def encode(urls, vocab, length): return pad_sequences([[vocab[c] for c in u if c in vocab] for u in urls], maxlen=length, padding="post", truncating="post")


def build_model(vocab_size, length):
    x = Input((length,), dtype="int32"); emb = Embedding(vocab_size+1,16)(x); branches=[]
    for filters,kernel in ((128,8),(128,10),(256,12)):
        b=Conv1D(filters,kernel,activation="relu")(emb); b=SpatialDropout1D(.4)(b); branches.append(GlobalMaxPooling1D()(b))
    y=Dense(128,activation="relu")(Concatenate()(branches)); y=Dropout(.4)(y); y=Dense(1,activation="sigmoid")(y)
    model=Model(x,y); model.compile(loss="binary_crossentropy",optimizer=Adam(learning_rate=.001,epsilon=1e-8),metrics=["accuracy"]); return model


def main():
    p=argparse.ArgumentParser(); p.add_argument("source_dataset",type=Path); p.add_argument("source_manifest",type=Path); p.add_argument("target_dataset",type=Path); p.add_argument("target_manifest",type=Path); p.add_argument("output_dir",type=Path); p.add_argument("--name",required=True); p.add_argument("--epochs",type=int,default=50); p.add_argument("--batch-size",type=int,default=512); p.add_argument("--max-length",type=int,default=150); p.add_argument("--seed",type=int,default=42); a=p.parse_args(); seed_all(a.seed)
    src=read_data(a.source_dataset); tgt=read_data(a.target_dataset); sm,mask=source_masks(src,a.source_manifest); tm,tids=target_rows(tgt,a.target_manifest)
    train_urls=src.loc[mask["train"],"url"]; vocab={c:i+1 for i,c in enumerate(sorted(set("".join(train_urls))))}
    arrays={"train":encode(train_urls.tolist(),vocab,a.max_length),"validation":encode(src.loc[mask["validation"],"url"].tolist(),vocab,a.max_length),"target_test":encode(tgt.iloc[tids].url.tolist(),vocab,a.max_length)}
    labels={"train":src.loc[mask["train"],"phishing"].to_numpy(),"validation":src.loc[mask["validation"],"phishing"].to_numpy(),"target_test":tgt.iloc[tids].phishing.to_numpy()}
    overlap=set(src.loc[mask["train"]|mask["validation"],"url"]) & set(tgt.iloc[tids].url)
    if overlap: raise ValueError(f"Target cohort overlaps source train/validation by {len(overlap)} exact URLs")
    model=build_model(len(vocab),a.max_length); hist=model.fit(arrays["train"],labels["train"],validation_data=(arrays["validation"],labels["validation"]),epochs=a.epochs,batch_size=a.batch_size,verbose=2,callbacks=[ReduceLROnPlateau(monitor="val_accuracy",patience=3,factor=.8,min_lr=1e-5)])
    a.output_dir.mkdir(parents=True,exist_ok=True); model_path=a.output_dir/f"{a.name}_model.h5"; vocab_path=a.output_dir/f"{a.name}_vocabulary.pickle"; model.save(model_path)
    with vocab_path.open("wb") as f: pickle.dump(vocab,f,pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(hist.history).to_csv(a.output_dir/f"{a.name}_history.csv",index=False); summary={}
    for split,rows,ids in (("validation",src.loc[mask["validation"]],np.flatnonzero(mask["validation"])),("target_test",tgt.iloc[tids],tids)):
        prob=model.predict(arrays[split],batch_size=a.batch_size,verbose=0).reshape(-1); pred=(prob>=.5).astype(int); summary[split]=metrics(labels[split],pred,prob); out=rows[["url","phishing"]].copy(); out.insert(0,"row_id",ids); out["prediction"]=pred; out["probability"]=prob; out.to_csv(a.output_dir/f"{a.name}_{split}_predictions.csv",index=False)
    rec={"baseline":"CNN-Fusion","source_dataset":str(a.source_dataset.resolve()),"source_dataset_sha256":sha256(a.source_dataset),"source_manifest":str(a.source_manifest.resolve()),"source_manifest_sha256":sha256(a.source_manifest),"target_dataset":str(a.target_dataset.resolve()),"target_dataset_sha256":sha256(a.target_dataset),"target_manifest":str(a.target_manifest.resolve()),"target_manifest_sha256":sha256(a.target_manifest),"model_sha256":sha256(model_path),"vocabulary_sha256":sha256(vocab_path),"preprocessing_fit":"source train only","validation_role":"source learning-rate monitoring","positive_class":{"value":1,"name":"phishing"},"decision_threshold":.5,"seed":a.seed,"metrics":summary}
    (a.output_dir/f"{a.name}_metrics.json").write_text(json.dumps(rec,indent=2)); print(json.dumps(rec,indent=2))


if __name__=="__main__": main()
