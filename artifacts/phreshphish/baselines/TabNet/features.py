# data_preprocessing.py
import pandas as pd
import numpy as np
import math
import re
from urllib.parse import urlparse
import tldextract
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

def calculate_entropy(text):
    """Calculates Shannon entropy of a string."""
    if not text:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(text.count(chr(x))) / len(text)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy

def extract_features(url):
    """
    Extracts the lexical features described in the paper (Figure 4).
    The paper mentions 79 features including lengths, counts, ratios, and entropy.
    """
    try:
        parsed = urlparse(url)
        ext = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}"
        path = parsed.path
        query = parsed.query
        filename = path.split('/')[-1] if '/' in path else ""
        
        feats = {}
        
        # --- 1. Length Features [cite: 328-355] ---
        feats['urlLen'] = len(url)
        feats['domainlength'] = len(domain)
        feats['path_length'] = len(path)
        feats['query_length'] = len(query)
        feats['filename_length'] = len(filename)
        
        # --- 2. Count Features (Symbols) ---
        # "SymbolCount" features in URL, Domain, and Filename
        symbols = ['.', '@', '-', '_', '%', '&', '#', '=', '+', '?', '/', '~', ',']
        for char in symbols:
            feats[f'SymbolCount_{char}_URL'] = url.count(char)
            feats[f'SymbolCount_{char}_Domain'] = domain.count(char)
            feats[f'SymbolCount_{char}_FileName'] = filename.count(char)
            
        # --- 3. Character Continuity & Types ---
        feats['DigitCount_URL'] = sum(c.isdigit() for c in url)
        feats['LetterCount_URL'] = sum(c.isalpha() for c in url)
        feats['DigitCount_Domain'] = sum(c.isdigit() for c in domain)
        feats['LetterCount_Domain'] = sum(c.isalpha() for c in domain)
        feats['DigitCount_FileName'] = sum(c.isdigit() for c in filename)
        
        # --- 4. Ratios [cite: 333, 340, 342, 348] ---
        total_len = len(url) if len(url) > 0 else 1
        path_len = len(path) if len(path) > 0 else 1
        
        feats['pathUrlRatio'] = len(path) / total_len
        feats['domainUrlRatio'] = len(domain) / total_len
        feats['argPathRatio'] = len(query) / path_len
        feats['digit_ratio'] = feats['DigitCount_URL'] / total_len
        feats['argUrlRatio'] = len(query) / total_len
        
        # --- 5. Entropy Features [cite: 332, 354] ---
        feats['Entropy_URL'] = calculate_entropy(url)
        feats['Entropy_Domain'] = calculate_entropy(domain)
        feats['Entropy_Filename'] = calculate_entropy(filename)
        
        # --- 6. Sensitive Words/Heuristics ---
        # Checking for obfuscation tactics or sensitive keywords
        feats['isPortEighty'] = 1 if ':80' in url else 0
        feats['has_www'] = 1 if 'www' in url else 0
        feats['has_login'] = 1 if 'login' in url.lower() else 0
        feats['has_admin'] = 1 if 'admin' in url.lower() else 0
        feats['has_confirm'] = 1 if 'confirm' in url.lower() else 0
        feats['has_account'] = 1 if 'account' in url.lower() else 0
        feats['has_secure'] = 1 if 'secure' in url.lower() else 0
        
        # TLD Count (as per paper "tld" importance)
        feats['tld_length'] = len(ext.suffix)
        
        return pd.Series(feats)
    except Exception as e:
        # Fallback for malformed URLs
        return pd.Series({})

def get_processing_artifacts(df_train):
    """
    Fits the Imputer and Scaler on the training data.
    Returns: X_train_scaled, imputer, scaler, feature_names
    """
    print("--- Extracting Features from Training Data ---")
    feature_df = df_train['url'].apply(extract_features)
    feature_names = feature_df.columns.tolist()
    
    # 1. Imputation: "replacing the NAN values... with a mean value" [cite: 131]
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(feature_df)
    
    # 2. Standardization: "mean of each column 0 and standard deviation 1" [cite: 132]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, imputer, scaler, feature_names

def process_new_data(df_new, imputer, scaler, selected_features=None):
    """
    Applies the PRE-FITTED imputer and scaler to new data.
    If selected_features is provided, it filters the columns.
    """
    print("--- Extracting Features from New Data ---")
    feature_df = df_new['url'].apply(extract_features)
    
    # Ensure all columns from training exist (handle potential missing cols in small batches)
    # The Imputer expects the exact same number of features as training.
    # Note: extracting features from 'url' usually yields consistent columns if extract_features is deterministic.
    
    # 1. Impute
    X_imputed = imputer.transform(feature_df)
    
    # 2. Scale
    X_scaled = scaler.transform(X_imputed)
    
    # 3. Feature Selection (Filtering)
    if selected_features is not None:
        all_features = feature_df.columns.tolist()
        # Find indices of the selected features
        indices = [all_features.index(f) for f in selected_features]
        X_final = X_scaled[:, indices]
        return X_final
    
    return X_scaled