"""Inference adapters for the released pretrained models; no fitting operations."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import tldextract

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'artifacts/phreshphish'
SOURCE = ARTIFACTS / 'source_definitions'
sys.path.insert(0, str(SOURCE))
from feature_extraction import FeatureExtractor
from model import FocalLossLGBM
from ngram_processing import NgramProcessor

# The original trusted pickles reference these classes in the entry module.
setattr(sys.modules['__main__'], 'FocalLossLGBM', FocalLossLGBM)
setattr(sys.modules['__main__'], 'NgramProcessor', NgramProcessor)

MODELS = ['frozen_568', 'MUDS', 'Roy', 'E2Phish', 'TabNet', 'Ebbu',
          'original', 'source_nuisance_balanced', 'hostname_independent']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def load_compressed(name):
    spec = json.loads((ARTIFACTS / 'models.json').read_text())[name]
    path = ARTIFACTS / spec['model']
    assert digest(path) == spec['compressed_sha256']
    with gzip.open(path, 'rb') as stream:
        raw = stream.read()
    assert hashlib.sha256(raw).hexdigest() == spec['original_model_sha256']
    with gzip.open(path, 'rb') as stream:
        return joblib.load(stream)


class Baseline:
    def __init__(self, name):
        self.name = name
        directory = ARTIFACTS / 'baselines' / name
        self.meta = json.loads((directory / 'metadata.json').read_text())
        spec = json.loads((ARTIFACTS / 'models.json').read_text())[name]
        assert digest(directory / 'features.py') == spec['feature_sha256']
        module_spec = importlib.util.spec_from_file_location('phresh_' + name, directory / 'features.py')
        self.features = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = self.features
        previous = Path.cwd()
        os.chdir(directory)
        try:
            module_spec.loader.exec_module(self.features)
        finally:
            os.chdir(previous)
        suffix = ROOT / 'experiments/android/bundle/public_suffix_list.dat'
        parser = tldextract.TLDExtract(suffix_list_urls=(suffix.as_uri(),), fallback_to_snapshot=False)
        parser('example.com')
        self.features.tldextract = SimpleNamespace(extract=parser)
        loaded = load_compressed(name)
        self.bundle = loaded if name == 'TabNet' else None
        self.model = loaded['model'] if name == 'TabNet' else loaded
        for key in self.model.get_params(deep=True):
            if key == 'n_jobs' or key.endswith('__n_jobs'):
                self.model.set_params(**{key: 1})
        assert list(self.model.classes_) == [0, 1]
        if name == 'Ebbu':
            state = self.meta['ebbu_state']
            assert not state['has_enchant'] and not self.features.dictionary_en.has_enchant
            def restore(gibberish):
                gibberish.counts = state['counts']
                gibberish.threshold = state['threshold']
            self.features.GibberishDetector._train_improved_model = restore
            original_init = self.features.NLPManager.__init__
            def initialize(manager):
                original_init(manager)
                manager.brands_list = state['brands']
                manager.keywords_list = state['keywords']
            self.features.NLPManager.__init__ = initialize

    def predict(self, urls):
        f = self.features
        if self.name == 'Roy':
            raw = f.extract_features(pd.Series(urls))
        elif self.name == 'MUDS':
            raw = f.extract_features(pd.DataFrame({'url': urls}))
        elif self.name == 'E2Phish':
            raw = pd.DataFrame([f.get_url_features(u) for u in urls])
        elif self.name == 'TabNet':
            raw = pd.Series(urls).apply(f.extract_features).apply(pd.Series).fillna(0)
        else:
            frame = pd.DataFrame({'url': urls, 'phishing': np.zeros(len(urls), dtype=int)})
            raw = f.extract_features_from_dataframe(frame).drop(columns=['labels'])
        raw = raw.loc[:, self.meta['feature_columns']]
        assert len(raw) == len(urls), 'Feature extraction must preserve the common cohort'
        if self.name == 'E2Phish':
            x = raw.loc[:, self.meta['selected_names']].to_numpy()
        elif self.name == 'TabNet':
            x = self.bundle['scaler'].transform(self.bundle['imputer'].transform(raw))
            x = x[:, self.bundle['selected_indices']]
        else:
            x = raw.to_numpy()
        assert np.isfinite(x).all()
        return self.model.predict_proba(x)[:, 1]


class LitePhish:
    def __init__(self, name):
        self.name = name
        if name == 'frozen_568':
            folder = ROOT / 'artifacts/litephish'
            self.bundle = {k: joblib.load(folder / (k + '.pkl')) for k in ['model', 'scaler']}
            self.bundle['processor'] = joblib.load(folder / 'ngram_processor.pkl')
            self.bundle['selected'] = np.asarray(joblib.load(folder / 'selected_features.pkl'), dtype=int)
        else:
            self.bundle = load_compressed(name)

    def predict(self, urls):
        from urllib.parse import urlsplit
        b = self.bundle
        urls = pd.Series(urls)
        hc = np.asarray([FeatureExtractor().generate_features(u) for u in urls], dtype=np.float32)
        text = urls
        if self.name == 'hostname_independent':
            text = urls.map(lambda u: (urlsplit(u).path or '/') + ('?' + urlsplit(u).query if urlsplit(u).query else ''))
        ng = b['processor'].transform(text).toarray()
        if self.name == 'frozen_568':
            x = np.hstack([hc, ng.astype(np.float32)]).astype(np.float32)
        else:
            x = np.hstack([hc[:, b['keep_handcrafted']], ng]).astype(np.float32)
        x = b['scaler'].transform(x[:, b['selected']])
        return b['model'].predict_proba(x, num_threads=2)[:, 1]


def predictor(name):
    return LitePhish(name) if name in MODELS[:1] + MODELS[6:] else Baseline(name)
