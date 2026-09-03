"""Generate and verify a row-aligned StealthPhisher feature cache."""
import argparse, hashlib, importlib.util, json
from pathlib import Path
import pandas as pd

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

p=argparse.ArgumentParser(); p.add_argument('dataset',type=Path); p.add_argument('feature_code',type=Path); p.add_argument('output',type=Path); p.add_argument('--workers',type=int,default=18); a=p.parse_args()
spec=importlib.util.spec_from_file_location('stealth_cache_features',a.feature_code); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
d=pd.read_csv(a.dataset,encoding='ISO-8859-1',usecols=['url','phishing']).reset_index(drop=True)
f=mod.extract_features_parallel(d.copy(),url_col='url',workers=a.workers).reset_index(drop=True)
if len(f)!=len(d) or not f.url.astype(str).equals(d.url.astype(str)) or not f.phishing.astype(int).equals(d.phishing.astype(int)): raise ValueError('Extracted cache lost row alignment')
a.output.parent.mkdir(parents=True,exist_ok=True); f.to_csv(a.output,index=False)
meta={'dataset':str(a.dataset.resolve()),'dataset_sha256':digest(a.dataset),'feature_code_sha256':digest(a.feature_code),'output_sha256':digest(a.output),'rows':len(f),'columns':list(f.columns)}
a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); print(json.dumps(meta,indent=2))
