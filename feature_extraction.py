
import numpy as np
import urllib.parse
import re
import logging
import math
from collections import Counter
from tldextract import extract as extract_tld_parts
from utils import UrlUtils
from constants import (
    SUSPICIOUS_TERMS, RARE_TLDS, SUSPICIOUS_DOMAINS,
    MAJOR_LEGITIMATE_SITES, TRUSTWORTHY_DOMAINS
)

class FeatureExtractor:
    """Class for extracting handcrafted features from URLs."""

    FEATURE_NAMES = [
        'URL Length', 'Hostname Length', 'Domain Length', 'TLD Length', 'Subdomain Length',
        'Path Length', 'Query Length', 'Max Token Length', 'Min Token Length', 'Mean Token Length',
        'Digit Count', 'Subdomain Depth', 'Domain Hierarchy', 'TLD Structure Depth', 'Encoded Hostname Indicators',
        'Slash Count', 'Dot Count', 'Hyphen in Hostname', 'Query Param Count', 'Path Depth',
        'Port Number', 'Has Port', 'Traversal Depth',
        'Domain to URL Ratio', 'Subdomain to URL Ratio', 'Hostname to URL Ratio', 'Path to URL Ratio',
        'Query to URL Ratio', 'Path to Domain Ratio', 'Query to Domain Ratio', 'Query to Path Ratio',
        'Uppercase Ratio', 'Subdomain Digit Ratio',
        'Hostname Entropy', 'Path Entropy', 'Query Entropy', 'URL 2-Gram Entropy', 'URL 3-Gram Entropy',
        'Hostname 2-Gram Entropy', 'Domain Entropy', 'Path Entropy Variance', 'Rolling Entropy Range',
        'Subdomain Entropy Gradient', 'Cookie Param Entropy',
        'Digits in Hostname', 'Alphabetic in Hostname', 'Punctuation in URL', 'Non-ASCII in URL',
        'Alpha Ratio in Hostname', 'Digit Ratio in Hostname', 'Special Char Ratio in Hostname',
        'Homoglyph Score',
        'Letter-Digit-Letter Count', 'Digit-Letter-Digit Count', 'Alpha-Digit Transitions',
        'Digit-Alpha Transitions', 'Consecutive Chars', 'Consecutive Digits', 'Consecutive Letters',
        'Special Char Cluster', 'Pattern Density',
        'Suspicious Keyword', 'URL Shortener', 'Digits in Subdomain', 'Redirect in Path',
        'Long Numeric Domain', 'TLD Risk Score', 'Suspicious Hosting', 'Numeric Domain Pattern',
        'Hex-Encoded IP', 'TLD-Domain Mismatch', 'Port Mismatch',
        'IP as Hostname', 'Non-HTTP Scheme', 'WWW Prefix', 'TLD is .com', 'Punycode Domain',
        'Char Diversity in URL', 'Char Diversity in Hostname',
        'Obfuscation Chars', 'Encoding Keywords', 'Redirect Keywords', 'Redirect Chain Count',
        'Consecutive Special Chars', 'Encoding Bypass Ratio', 'Redirect TLD Disparity'
    ]

    def __init__(self):
        self.utils = UrlUtils()
        self.logger = logging.getLogger(__name__)

    
    def calculate_shannon_entropy(self, text):
        if not text: return 0
        freq = Counter(text)
        total = len(text)
        return -sum((f/total) * math.log2(f/total) for f in freq.values() if f>0)

    def compute_ngram_entropy(self, text, n=2):
        if len(text) < n: return 0
        counts = Counter(text[i:i+n] for i in range(len(text)-n+1))
        total = sum(counts.values())
        return -sum((c/total) * math.log2(c/total) for c in counts.values() if total>0)

    def check_ip_presence(self, hostname):
        parts = hostname.split('.')
        return len(parts)==4 and all(p.isdigit() and 0<=int(p)<=255 for p in parts)

    def calculate_tld_risk_score(self, domain, tld):
        full = f"{domain}.{tld}".lower() if domain and tld else (domain or tld or "").lower()
        if any(s in full for s in MAJOR_LEGITIMATE_SITES + TRUSTWORTHY_DOMAINS): return 0.0
        risk = 0.9 if tld and ('.'+tld.lower()) in RARE_TLDS else 0.0
        return 1.0 if any(s in full for s in SUSPICIOUS_DOMAINS) else risk

    def detect_url_shortener(self, url):
        netloc = urllib.parse.urlparse(url).netloc.lower()
        shorts = {"bit.ly", "goo.gl", "tinyurl.com", "t.co", "is.gd", "buff.ly", "adf.ly", "bc.vc"}
        if netloc in shorts: return 1
        return 1 if (len(netloc)<=5 and '.' not in netloc) else 0

    def detect_homoglyph_clusters(self, h):
        return sum(1 for c in h if c in {'?','?','?','?','?'})

    def calculate_encoding_bypass(self, url):
        seq = re.findall(r'%[0-9a-fA-F]{2}', url)
        unusual = [s for s in seq if s.upper() not in {'%20','%21','%2F','%3A','%3F','%3D'}]
        return len(unusual)/len(url) if url else 0

    def path_segment_entropy_variance(self, path):
        segs = [s for s in path.split('/') if s]
        return np.var([self.calculate_shannon_entropy(s) for s in segs]) if len(segs)>=2 else 0

    def special_char_clusters(self, url):
        m = re.findall(r'[^\w\./-]{2,}', url)
        return max(len(x) for x in m) if m else 0

    def rolling_entropy(self, url, w=5):
        if len(url)<w: return 0
        e = [self.calculate_shannon_entropy(url[i:i+w]) for i in range(len(url)-w)]
        return max(e)-min(e) if e else 0

    def tld_domain_mismatch(self, d, t):
        exp = {'com':(3,12), 'net':(3,10), 'org':(3,8), 'io':(2,6)}.get(t,(0,0))
        return 0 if exp[0] <= len(d) <= exp[1] else 1

    def subdomain_entropy_gradient(self, sub):
        p = sub.split('.')
        return (self.calculate_shannon_entropy(p[0]) - self.calculate_shannon_entropy(p[-1])) if len(p)>=2 else 0

    def port_service_mismatch(self, p):
        return 1 if p and p not in {80,443,8080,8000,3000} else 0

    def path_traversal_depth(self, p):
        return p.count('../') + p.count('..\\')

    def mixed_case_obfuscation(self, u):
        return sum(1 for c in u if c.isupper())/len(u) if u else 0

    def subdomain_token_anomaly(self, sub):
        t = re.findall(r'[a-z]+|\d+', sub)
        return sum(1 for x in t if x.isdigit())/len(t) if t else 0

    def sliding_pattern_density(self, u, w=5):
        if len(u)<w: return 0
        return np.mean([len(set(u[i:i+w])) for i in range(len(u)-w)])

    def cookie_param_entropy(self, q):
        if not q: return 0.0
        try:
            p = urllib.parse.parse_qs(q)
            relevant = []
            for k,v in p.items():
                if k.lower() in ["sessionid","sessid","sid","token","auth","uid","user_id","cookie"]:
                    relevant.extend(v)
            return self.calculate_shannon_entropy("".join(relevant)) if relevant else 0.0
        except: return 0.0

    def redirect_tld_disparity(self, q, main_tld):
        if not q or not main_tld: return 0
        try:
            p = urllib.parse.parse_qs(q)
            redir_url = None
            for k,v in p.items():
                if k.lower() in ["redirect","url","return","goto","dest"] and v:
                    if v[0].startswith(('http','//')): 
                        redir_url = v[0]
                        break
            if redir_url:
                if redir_url.startswith('//'): redir_url='http:'+redir_url
                t = extract_tld_parts(redir_url).suffix.lower()
                if t and t!=main_tld.lower(): return 1
        except: pass
        return 0

    def detect_hex_ip(self, h):
        return 1 if re.search(r'(0x[a-f0-9]{8})|(\d+\.\d+\.\d+\.\d+)', h) else 0

    def generate_features(self, url):
        """Generate only handcrafted features for a URL."""
        try:
            parsed = urllib.parse.urlparse(url)
            tld_res = extract_tld_parts(url)

            url_lower = url.lower()
            hostname = parsed.netloc.lower()
            path = parsed.path
            query = parsed.query
            scheme = parsed.scheme.lower()
            port = parsed.port
            subdomain = tld_res.subdomain.lower()
            domain = tld_res.domain.lower()
            tld = tld_res.suffix.lower()

            f = []

            # Lexical/Structural
            url_len = len(url)
            tokens = url.split('/')
            f.append(url_len)
            f.append(len(hostname))
            f.append(len(domain))
            f.append(len(tld))
            f.append(len(subdomain))
            f.append(len(path))
            f.append(len(query))
            f.append(max(len(t) for t in tokens) if tokens else 0)
            f.append(min(len(t) for t in tokens if t) if tokens else 0)
            f.append(np.mean([len(t) for t in tokens if t]) if tokens else 0)
            f.append(sum(c.isdigit() for c in url))
            f.append(subdomain.count('.')+1 if subdomain else 0)
            f.append(domain.count('.')+1 if domain else 0)
            f.append(tld.count('.')+1 if tld else 0)
            f.append(sum(not c.isalnum() for c in hostname))
            f.append(url.count('/'))
            f.append(url.count('.'))
            f.append(hostname.count('-'))
            f.append(len(query.split('&')) if query else 0)
            f.append(path.count('/'))
            f.append(port if port else -1)
            f.append(1 if port else 0)
            f.append(self.path_traversal_depth(path))

            # Ratios
            f.append(len(domain)/url_len if url_len else 0)
            f.append(len(subdomain)/url_len if url_len else 0)
            f.append(len(hostname)/url_len if url_len else 0)
            f.append(len(path)/url_len if url_len else 0)
            f.append(len(query)/url_len if url_len else 0)
            f.append(len(path)/len(domain) if domain else 0)
            f.append(len(query)/len(domain) if domain else 0)
            f.append(len(query)/len(path) if path else 0)
            f.append(self.mixed_case_obfuscation(url))
            f.append(self.subdomain_token_anomaly(subdomain))

            # Entropy
            f.append(self.calculate_shannon_entropy(hostname))
            f.append(self.calculate_shannon_entropy(path))
            f.append(self.calculate_shannon_entropy(query))
            f.append(self.compute_ngram_entropy(url, 2))
            f.append(self.compute_ngram_entropy(url, 3))
            f.append(self.compute_ngram_entropy(hostname, 2))
            f.append(self.calculate_shannon_entropy(domain))
            f.append(self.path_segment_entropy_variance(path))
            f.append(self.rolling_entropy(url))
            f.append(self.subdomain_entropy_gradient(subdomain))
            f.append(self.cookie_param_entropy(query))

            # Composition
            dig_host = sum(c.isdigit() for c in hostname)
            f.append(dig_host)
            f.append(sum(c.isalpha() for c in hostname))
            f.append(sum(1 for c in url if c in "!@#$%^&*()"))
            f.append(sum(ord(c)>127 for c in url))
            f.append(dig_host/len(hostname) if hostname else 0) 
            f.append(dig_host/len(hostname) if hostname else 0)
            f.append(sum(not c.isalnum() for c in hostname)/len(hostname) if hostname else 0)
            f.append(self.detect_homoglyph_clusters(hostname))

            # Transitions
            url_nz = url_len if url_len else 1
            f.append(sum(1 for i in range(1, url_len-1) if url[i].isdigit() and url[i-1].isalpha() and url[i+1].isalpha()))
            f.append(sum(1 for i in range(1, url_len-1) if url[i].isalpha() and url[i-1].isdigit() and url[i+1].isdigit()))
            f.append(sum(1 for i in range(url_len-1) if url[i].isalpha() and url[i+1].isdigit())/url_nz)
            f.append(sum(1 for i in range(url_len-1) if url[i].isdigit() and url[i+1].isalpha())/url_nz)
            f.append(sum(1 for i in range(url_len-1) if url[i]==url[i+1]))
            f.append(sum(1 for i in range(url_len-1) if url[i].isdigit() and url[i+1].isdigit()))
            f.append(sum(1 for i in range(url_len-1) if url[i].isalpha() and url[i+1].isalpha()))
            f.append(self.special_char_clusters(url))
            f.append(self.sliding_pattern_density(url))

            # Behavioral
            f.append(1 if any(k in url_lower for k in SUSPICIOUS_TERMS) else 0)
            f.append(self.detect_url_shortener(url))
            f.append(1 if any(c.isdigit() for c in subdomain) else 0)
            f.append(1 if ("//" in path or "redirect" in path.lower()) else 0)
            f.append(1 if domain and len(domain)>8 and any(c.isdigit() for c in domain) else 0)
            f.append(self.calculate_tld_risk_score(domain, tld))
            f.append(1 if any(s in hostname for s in SUSPICIOUS_DOMAINS) else 0)
            f.append(1 if domain and (sum(c.isdigit() for c in domain)/len(domain)>0.3 or sum(c.isdigit() for c in domain)>3) else 0)
            f.append(self.detect_hex_ip(hostname))
            f.append(self.tld_domain_mismatch(domain, tld))
            f.append(self.port_service_mismatch(port))

            # Protocol
            f.append(1 if self.check_ip_presence(hostname) else 0)
            f.append(0 if scheme=="http" else 1)
            f.append(1 if hostname.startswith("www.") else 0)
            f.append(1 if tld=="com" else 0)
            f.append(1 if domain.startswith("xn--") else 0)

            # Diversity
            f.append(len(set(url))/url_len if url_len else 0)
            f.append(len(set(hostname))/len(hostname) if hostname else 0)

            # Obfuscation
            f.append(1 if "%" in url or any(ord(c)>127 for c in url) else 0)
            f.append(1 if re.search(r"(base64|hex)[^a-zA-Z0-9]", url, re.IGNORECASE) else 0)
            f.append(1 if (query and any(r in query.lower() for r in ["redirect","url=","goto="])) else 0)
            f.append(url.count("//")-1 if "://" in url else url.count("//"))
            f.append(sum(1 for i in range(url_len-1) if not url[i].isalnum() and url[i]==url[i+1]))
            f.append(self.calculate_encoding_bypass(url))
            f.append(self.redirect_tld_disparity(query, tld))

            return f

        
        except Exception as e:
            self.logger.error(f"Error features for '{url}': {e}")
            return [0] * len(self.FEATURE_NAMES)
        
