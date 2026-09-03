import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse



def get_url_features(url):
    features = {}
    try:
        # Parse URL
        parsed = urlparse(url)
        hostname = parsed.netloc
        path = parsed.path
        
       
        features['NumDots'] = url.count('.')
        features['SubdomainLevel'] = features['NumDots'] - 1 if features['NumDots'] > 1 else 0
        features['PathLevel'] = path.count('/')
        features['UrlLength'] = len(url)
        features['NumDash'] = url.count('-')
        features['NumDashInHostname'] = hostname.count('-')
        features['AtSymbol'] = 1 if '@' in url else 0
        features['TildeSymbol'] = 1 if '~' in url else 0
        features['NumUnderscore'] = url.count('_')
        features['NumPercent'] = url.count('%')
        features['NumQueryComponents'] = len(parsed.query.split('&')) if parsed.query else 0
        features['NumAmpersand'] = url.count('&')
        features['NumHash'] = url.count('#')
        features['NumNumericChars'] = sum(c.isdigit() for c in url)
        features['NoHttps'] = 1 if parsed.scheme != 'https' else 0
        features['RandomString'] = 1 if re.search(r'[0-9a-f]{5,}', url) else 0  
        features['IpAddress'] = 1 if re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', hostname) else 0
        features['DomainInSubdomains'] = 1 if hostname.count('.') > 2 else 0
        features['DomainInPaths'] = 1 if hostname in path else 0
        features['HttpsInHostname'] = 1 if 'https' in hostname else 0
        features['HostnameLength'] = len(hostname)
        features['PathLength'] = len(path)
        features['QueryLength'] = len(parsed.query)
        features['DoubleSlashInPath'] = 1 if '//' in path else 0
        
        sensitive_words = ['secure', 'account', 'webscr', 'login', 'ebay', 'signin', 'banking', 'confirm']
        features['NumSensitiveWords'] = sum(1 for w in sensitive_words if w in url.lower())
        
     
        
        features['IframeOrFrame'] = 0
        features['MissingTitle'] = 0
        features['ImagesOnlyInForm'] = 0
        features['PctExtHyperlinks'] = 0
        features['PctNullSelfRedirectHyperlinks'] = 0
        
  
        features['SubdomainLevelRT'] = 1 if features['SubdomainLevel'] > 1 else -1
        features['UrlLengthRT'] = 1 if len(url) > 54 else (0 if len(url) > 75 else -1)
        features['PctExtResourceUrlsRT'] = 0 
        features['AbnormalExtFormActionR'] = 0 
        features['ExtMetaScriptLinkRT'] = 0 
        features['PctExtNullSelfRedirectHyperlinksRT'] = 0 
            
       
        defaults = [
            'EmbeddedBrandName', 'PctExtResourceUrls', 'ExtFavicon', 'InsecureForms',
            'RelativeFormAction', 'ExtFormAction', 'AbnormalFormAction', 
            'FrequentDomainNameMismatch', 'FakeLinkInStatusBar', 'RightClickDisabled',
            'PopUpWindow', 'SubmitInfoToEmail'
        ]
        for d in defaults:
            features[d] = 0

    except Exception:
   
        return None

    return features

def extract_features_from_dataframe(df):
    """Iterates over a dataframe and extracts features for every URL."""
    print("Extracting features (Lexical only)... This should be fast.")
    features_list = []
    total = len(df)
    

    for index, row in df.iterrows():
        if index % 5000 == 0:
            print(f"Processed {index}/{total}")
            
        feats = get_url_features(row['url'])
        if feats:
            feats['labels'] = row['phishing']
            features_list.append(feats)
            
    return pd.DataFrame(features_list)