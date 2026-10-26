"""Numerically validated inference port; preserves all learned parameters."""
import ast
import ctypes as ct
import json
import os
from pathlib import Path
import sys
import types
from urllib.parse import urlparse
import re
import math
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'python_packages'))
import tldextract


def pointer(value):
    return value.ctypes.data_as(ct.c_void_p)


class Estimator:
    def __init__(self, spec, arrays, kernels):
        self.spec, self.arrays, self.kernels = spec, arrays, kernels
        self.kind = spec['kind']
        if self.kind in ('RandomForestClassifier', 'DecisionTreeClassifier'):
            trees = spec.get('children', [spec])
            self.nodes = {}
            roots, offset = [], 0
            for i, tree in enumerate(trees):
                roots.append(offset)
                n = len(arrays[tree['feature']])
                for key in ('left', 'right', 'feature', 'threshold', 'probability', 'missing_left'):
                    value = arrays[tree[key]].copy()
                    if key in ('left', 'right'):
                        value[value >= 0] += offset
                    self.nodes.setdefault(key, []).append(value)
                offset += n
            self.nodes = {k:np.ascontiguousarray(np.concatenate(v), dtype=np.uint8 if k=='missing_left' else np.float64 if k in ('threshold','probability') else np.int64) for k,v in self.nodes.items()}
            self.roots = np.asarray(roots,dtype=np.int64)
        elif self.kind == 'StackingClassifier':
            self.children = [Estimator(s, arrays, kernels) for s in spec['children']]
            self.final = Estimator(spec['final'], arrays, kernels)

    def get(self, key):
        return self.arrays[self.spec[key]]

    def predict(self, x):
        if self.kind in ('RandomForestClassifier','DecisionTreeClassifier'):
            x = np.ascontiguousarray(x, dtype=np.float32)
            out = np.empty(len(x), dtype=np.float64)
            self.kernels.forest_probability(pointer(x),len(x),x.shape[1],pointer(self.roots),len(self.roots),
                *[pointer(self.nodes[k]) for k in ('left','right','feature','threshold','probability','missing_left')],pointer(out))
            return out
        if self.kind == 'SVC':
            x = np.ascontiguousarray(x,dtype=np.float64)
            support = np.ascontiguousarray(self.get('support'))
            out = np.empty(len(x),dtype=np.float64)
            self.kernels.svc_probability(pointer(x),len(x),x.shape[1],pointer(support),len(support),pointer(self.get('dual')),
                float(self.get('intercept')[0]),self.spec['gamma'],float(self.get('probA')[0]),float(self.get('probB')[0]),pointer(out))
            return out
        if self.kind == 'LogisticRegression':
            value = np.asarray(x,dtype=np.float64) @ self.get('coef')[0] + self.get('intercept')[0]
            return 1/(1+np.exp(-value))
        if self.kind == 'GaussianNB':
            x = np.asarray(x,dtype=np.float64)
            scores = np.asarray([np.log(prior)-0.5*np.sum(np.log(2*np.pi*var))-0.5*np.sum((x-theta)**2/var,axis=1)
                for theta,var,prior in zip(self.get('theta'),self.get('var'),self.get('prior'))]).T
            scores -= np.max(scores,axis=1,keepdims=True)
            prob = np.exp(scores)
            return prob[:,1]/prob.sum(axis=1)
        if self.kind == 'StackingClassifier':
            return self.final.predict(np.column_stack([child.predict(x) for child in self.children]))
        raise NotImplementedError(self.kind)


def load_scalar_features(path, name):
    # Execute original feature definitions, excluding training/CLI imports only.
    tree = ast.parse(path.read_text())
    allowed = []
    keep_imports = {'__future__','re','math','numpy','urllib','ipaddress','tld','tldextract','os','logging','pathlib'}
    for node in tree.body:
        if isinstance(node, ast.Import):
            if all(n.name.split('.')[0] in keep_imports for n in node.names):
                allowed.append(node)
        elif isinstance(node, ast.ImportFrom):
            if (node.module or '').split('.')[0] in keep_imports:
                allowed.append(node)
        elif not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id=='__name__'):
            allowed.append(node)
    # pd.Series is only used as a dictionary wrapper by the TabNet scalar extractor.
    pd = types.SimpleNamespace(Series=lambda value:value)
    namespace = dict(__name__='arm_features_'+name, __file__=str(path),pd=pd,np=np,
                     tqdm=lambda iterable,**kwargs:iterable)
    exec(compile(ast.Module(body=allowed,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


class BaselineRuntime:
    def __init__(self, name):
        self.name = name
        self.directory = ROOT/'baseline_bundles'/name
        self.meta = json.loads((self.directory/'metadata.json').read_text())
        self.arrays = dict(np.load(self.directory/'parameters.npz'))
        self.kernels = ct.CDLL(str(ROOT/'lib_baseline_kernels.so'))
        self.kernels.forest_probability.argtypes = [ct.c_void_p,ct.c_int,ct.c_int,ct.c_void_p,ct.c_int]+[ct.c_void_p]*7
        self.kernels.svc_probability.argtypes = [ct.c_void_p,ct.c_int,ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p]+[ct.c_double]*4+[ct.c_void_p]
        self.estimator = Estimator(self.meta['estimator'],self.arrays,self.kernels) if name!='MUDS' else None
        suffix = tldextract.TLDExtract(cache_dir=str(ROOT/'baseline_suffix_cache'),
            suffix_list_urls=((ROOT/'bundle/public_suffix_list.dat').as_uri(),),fallback_to_snapshot=False)
        suffix('example.com')
        tldextract.extract = suffix
        old = Path.cwd()
        os.chdir(self.directory)
        try:
            self.features = load_scalar_features(self.directory/'original_features.py',name)
        finally:
            os.chdir(old)
        if name == 'Ebbu':
            state = self.meta['ebbu_state']
            assert not state['has_enchant']
            def frozen_training(gib):
                gib.counts = state['counts']
                gib.threshold = state['threshold']
            self.features['GibberishDetector']._train_improved_model = frozen_training
            self.ebbu_parser = self.features['DomainParser']()
        if name == 'MUDS':
            self.lib = ct.CDLL(str(ROOT/'lightgbm-4.5.0/lib_lightgbm.so'))
            self.lib.LGBM_GetLastError.restype=ct.c_char_p
            self.lib.LGBM_BoosterCreateFromModelfile.argtypes=[ct.c_char_p,ct.POINTER(ct.c_int),ct.POINTER(ct.c_void_p)]
            self.lib.LGBM_BoosterPredictForMat.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int32,ct.c_int32,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_char_p,ct.POINTER(ct.c_int64),ct.POINTER(ct.c_double)]
            self.handle=ct.c_void_p()
            niter=ct.c_int()
            assert self.lib.LGBM_BoosterCreateFromModelfile(str(self.directory/'model.txt').encode(),ct.byref(niter),ct.byref(self.handle))==0

    def extract(self, urls):
        f = self.features
        if self.name == 'Roy':
            return np.asarray([f['extract_one'](u) for u in urls],dtype=np.float64)
        if self.name == 'E2Phish':
            rows = [f['get_url_features'](u) for u in urls]
        elif self.name == 'TabNet':
            rows = [f['extract_features'](u) for u in urls]
        elif self.name == 'Ebbu':
            rules=f['URLRules']()
            state=self.meta['ebbu_state']
            rules.nlp_manager.brands_list=state['brands']
            rules.nlp_manager.keywords_list=state['keywords']
            rows=[]
            for url in urls:
                p=self.ebbu_parser.parse(url)
                assert p['valid'], 'Do not drop URLs from the common cohort'
                rows.append(rules.rules_main(p['domain'],p['tld'],p['subdomain'],p['path'],p['words_raw']))
        elif self.name == 'MUDS':
            rows=[]
            for u in urls:
                rows.append(dict(zip(self.meta['feature_columns'],[
                    f['having_ip_address'](u),f['abnormal_url'](u),u.count('.'),u.count('www'),u.count('@'),
                    f['no_of_dir'](u),f['no_of_embed'](u),f['shortening_service'](u),u.count('https'),u.count('http'),
                    u.count('%'),u.count('?'),u.count('-'),u.count('='),len(u),len(urlparse(u).netloc),
                    f['suspicious_words'](u),f['fd_length'](u),f['tld_length'](f['get_tld'](u,fail_silently=True)),
                    f['digit_count'](u),f['letter_count'](u)])))
        else:
            raise ValueError(self.name)
        return np.asarray([[r.get(k,0) for k in self.meta['feature_columns']] for r in rows],dtype=np.float64)

    def prepare(self, raw):
        if self.name == 'E2Phish':
            return raw[:,[self.meta['feature_columns'].index(k) for k in self.meta['selected_names']]]
        if self.name == 'TabNet':
            x=raw.copy()
            statistics=np.asarray(self.meta['imputer_statistics'])
            x=np.where(np.isnan(x),statistics,x)
            x-=np.asarray(self.meta['scaler_mean'])
            x/=np.asarray(self.meta['scaler_scale'])
            return x[:,self.meta['selected_indices']]
        return raw

    def predict(self, x):
        if self.name != 'MUDS':
            return self.estimator.predict(x)
        x=np.ascontiguousarray(x,dtype=np.float64)
        out=np.empty(len(x),dtype=np.float64)
        length=ct.c_int64()
        code=self.lib.LGBM_BoosterPredictForMat(self.handle,pointer(x),1,len(x),x.shape[1],1,0,0,-1,b'num_threads=1',ct.byref(length),out.ctypes.data_as(ct.POINTER(ct.c_double)))
        if code:
            raise RuntimeError(self.lib.LGBM_GetLastError().decode())
        return out
