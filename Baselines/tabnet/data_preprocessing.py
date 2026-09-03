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
        
      
        feats['urlLen'] = len(url)
        feats['domainlength'] = len(domain)
        feats['path_length'] = len(path)
        feats['query_length'] = len(query)
        feats['filename_length'] = len(filename)
        
       
        symbols = ['.', '@', '-', '_', '%', '&', '#', '=', '+', '?', '/', '~', ',']
        for char in symbols:
            feats[f'SymbolCount_{char}_URL'] = url.count(char)
            feats[f'SymbolCount_{char}_Domain'] = domain.count(char)
            feats[f'SymbolCount_{char}_FileName'] = filename.count(char)
            
       
        feats['DigitCount_URL'] = sum(c.isdigit() for c in url)
        feats['LetterCount_URL'] = sum(c.isalpha() for c in url)
        feats['DigitCount_Domain'] = sum(c.isdigit() for c in domain)
        feats['LetterCount_Domain'] = sum(c.isalpha() for c in domain)
        feats['DigitCount_FileName'] = sum(c.isdigit() for c in filename)
        
   
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
        
      
        feats['isPortEighty'] = 1 if ':80' in url else 0
        feats['has_www'] = 1 if 'www' in url else 0
        feats['has_login'] = 1 if 'login' in url.lower() else 0
        feats['has_admin'] = 1 if 'admin' in url.lower() else 0
        feats['has_confirm'] = 1 if 'confirm' in url.lower() else 0
        feats['has_account'] = 1 if 'account' in url.lower() else 0
        feats['has_secure'] = 1 if 'secure' in url.lower() else 0
        
      
        feats['tld_length'] = len(ext.suffix)
        
        return pd.Series(feats)
    except Exception as e:
       
        return pd.Series({})

def get_processing_artifacts(df_train):
    """
    Fits the Imputer and Scaler on the training data.
    Returns: X_train_scaled, imputer, scaler, feature_names
    """
    print("--- Extracting Features from Training Data ---")
    feature_df = df_train['url'].apply(extract_features)
    feature_names = feature_df.columns.tolist()
    
 
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(feature_df)
    
   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, imputer, scaler, feature_names

def process_new_data(df_new, imputer, scaler, selected_features=None):
    
    print("--- Extracting Features from New Data ---")
    feature_df = df_new['url'].apply(extract_features)
    
   
    X_imputed = imputer.transform(feature_df)
    
   
    X_scaled = scaler.transform(X_imputed)
  
    if selected_features is not None:
        all_features = feature_df.columns.tolist()
       
        indices = [all_features.index(f) for f in selected_features]
        X_final = X_scaled[:, indices]
        return X_final
    
    return X_scaled