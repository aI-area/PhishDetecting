import pandas as pd
import numpy as np
import re
import zlib
import math
import requests
import tldextract
import concurrent.futures
import warnings
import time
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Try importing tqdm for a nice progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

warnings.filterwarnings('ignore')

class StealthPhisherExtractor:
    def __init__(self, timeout=1):
        # Timeout set to 2 seconds to fail fast on dead/slow sites
        self.timeout = timeout
        self.headers = {'User-Agent': 'StealthPhisher-Bot/1.0'}
        # Exact feature list from the paper's final model
        self.feature_columns = [
            'LengthOfURL', 'URLComplexity', 'CharacterComplexity', 'DomainLengthOfURL', 
            'IsDomainIP', 'TLDLength', 'LetterCntInURL', 'URLLetterRatio', 
            'DigitCntInURL', 'URLDigitRatio', 'EqualCharCntInURL', 'QuesMarkCntInURL', 
            'AmpCharCntInURL', 'OtherSpclCharCntInURL', 'URLOtherSpclCharRatio', 
            'NumberOfHashtags', 'NumberOfSubdomains', 'HavingPath', 'PathLength', 
            'HavingQuery', 'HavingFragment', 'HavingAnchor', 'HasSSL', 'IsUnreachable', 
            'LineOfCode', 'LongestLineLength', 'HasTitle', 'HasFavicon'
        ]

    def kolmogorov_complexity(self, text):
        if not text: return 0.0
        encoded = text.encode('utf-8')
        compressed = zlib.compress(encoded)
        return len(compressed) / len(text)

    def shannon_entropy(self, text):
        if not text: return 0.0
        prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
        return -sum([p * math.log(p) / math.log(2.0) for p in prob])

    def process_url(self, url):
        features = {}
        try:
            url_str = str(url).strip()
            if not url_str: return [0] * len(self.feature_columns)

            if not url_str.startswith(('http://', 'https://')):
                target_url = 'http://' + url_str
            else:
                target_url = url_str

            parsed = urlparse(target_url)
            ext = tldextract.extract(target_url)
            domain = f"{ext.domain}.{ext.suffix}"
            path = parsed.path

            # --- Lexical Features (Fast) ---
            features['LengthOfURL'] = len(target_url)
            features['DomainLengthOfURL'] = len(domain)
            features['TLDLength'] = len(ext.suffix)
            features['URLComplexity'] = self.kolmogorov_complexity(target_url)
            features['CharacterComplexity'] = self.shannon_entropy(target_url)
            features['IsDomainIP'] = 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ext.domain) else 0
            
            letters = sum(c.isalpha() for c in target_url)
            digits = sum(c.isdigit() for c in target_url)
            special = len(target_url) - letters - digits
            
            features['LetterCntInURL'] = letters
            features['URLLetterRatio'] = letters / len(target_url) if len(target_url) > 0 else 0
            features['DigitCntInURL'] = digits
            features['URLDigitRatio'] = digits / len(target_url) if len(target_url) > 0 else 0
            
            features['EqualCharCntInURL'] = target_url.count('=')
            features['QuesMarkCntInURL'] = target_url.count('?')
            features['AmpCharCntInURL'] = target_url.count('&')
            features['NumberOfHashtags'] = target_url.count('#')
            
            features['OtherSpclCharCntInURL'] = max(0, special - (features['EqualCharCntInURL'] + features['QuesMarkCntInURL'] + features['AmpCharCntInURL'] + features['NumberOfHashtags']))
            features['URLOtherSpclCharRatio'] = features['OtherSpclCharCntInURL'] / len(target_url) if len(target_url) > 0 else 0
            
            features['NumberOfSubdomains'] = len(ext.subdomain.split('.')) if ext.subdomain else 0
            features['HavingPath'] = 1 if path and path != '/' else 0
            features['PathLength'] = len(path)
            features['HavingQuery'] = 1 if parsed.query else 0
            features['HavingFragment'] = 1 if parsed.fragment else 0
            features['HavingAnchor'] = 1 if '#' in target_url else 0
            features['HasSSL'] = 1 if parsed.scheme == 'https' else 0

            # --- Network Features (HTML) ---
            # These cause the delay. If the site is down, it waits for timeout.
            features['IsUnreachable'] = 0
            features['LineOfCode'] = 0
            features['LongestLineLength'] = 0
            features['HasTitle'] = 0
            features['HasFavicon'] = 0

            try:
                response = requests.get(target_url, headers=self.headers, timeout=self.timeout)
                if response.status_code == 200:
                    content = response.text
                    # Only parse if HTML
                    if 'text/html' in response.headers.get('Content-Type', ''):
                        soup = BeautifulSoup(content, 'html.parser')
                        lines = content.splitlines()
                        features['LineOfCode'] = len(lines)
                        features['LongestLineLength'] = max([len(line) for line in lines]) if lines else 0
                        features['HasTitle'] = 1 if soup.title else 0
                        features['HasFavicon'] = 1 if soup.find("link", rel=lambda x: x and 'icon' in x.lower()) else 0
                else:
                    features['IsUnreachable'] = 1
            except Exception:
                features['IsUnreachable'] = 1

        except Exception:
            # Fallback for parsing errors: fill 0
            for col in self.feature_columns:
                if col not in features: features[col] = 0
        
        # Ensure correct column order
        return [features.get(col, 0) for col in self.feature_columns]

def extract_features_parallel(df, url_col='url', workers=50):
    """
    Extracts features using ThreadPoolExecutor with a progress bar.
    """
    total_urls = len(df)
    print(f"\n[INFO] Initializing extraction for {total_urls} URLs...")
    print(f"[INFO] Using {workers} concurrent workers. (Network requests will slow this down)")
    
    extractor = StealthPhisherExtractor(timeout=2)
    
    # Pre-allocate results list to maintain order (Critical!)
    results = [None] * total_urls
    urls = df[url_col].tolist()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks and map futures to their original index
        future_to_index = {executor.submit(extractor.process_url, url): i for i, url in enumerate(urls)}
        
        # Use TQDM to show progress as tasks complete
        if HAS_TQDM:
            for future in tqdm(concurrent.futures.as_completed(future_to_index), total=total_urls, unit="url"):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = [0] * len(extractor.feature_columns)
        else:
            # Fallback if TQDM is missing
            completed = 0
            print("[INFO] (tqdm not installed, showing simple progress)")
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = [0] * len(extractor.feature_columns)
                completed += 1
                if completed % 1000 == 0:
                    print(f"\rProgress: {completed}/{total_urls}", end="")
            print("")
    
    print(f"\n[INFO] Extraction complete.")
    feature_df = pd.DataFrame(results, columns=extractor.feature_columns)
    df_reset = df.reset_index(drop=True)
    return pd.concat([df_reset, feature_df], axis=1)


