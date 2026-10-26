# utils.py

import math
import re
from collections import Counter
import urllib.parse
from tldextract import extract as extract_tld_parts
import logging
from constants import MAJOR_LEGITIMATE_SITES, TRUSTWORTHY_DOMAINS, SUSPICIOUS_DOMAINS, RARE_TLDS

class UrlUtils:
    """Utility class for URL processing (entropy, TLD risk, shortener detection)."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_shannon_entropy(self, text):
        """Shannon entropy of characters in text."""
        if not text:
            return 0
        freqs = Counter(text)
        total = len(text)
        probs = [f / total for f in freqs.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def compute_ngram_entropy(self, text, n=2):
        """Compute n-gram entropy of text (match standalone)."""
        if len(text) < n:
            return 0
        counts = Counter(text[i:i+n] for i in range(len(text) - n + 1))
        total = sum(counts.values())
        return -sum(
            (c / total) * math.log2(c / total)
            for c in counts.values() if total > 0
        )

    def check_ip_presence(self, hostname):
        """Returns True if hostname is an IPv4 address."""
        parts = hostname.split('.')
        if len(parts) == 4 and all(part.isdigit() for part in parts):
            return all(0 <= int(p) <= 255 for p in parts)
        return False

    def split_domain_parts(self, hostname):
        """Split into (subdomain, domain, tld)."""
        parts = hostname.split('.')
        if len(parts) > 2:
            return '.'.join(parts[:-2]), parts[-2], parts[-1]
        elif len(parts) == 2:
            return '', parts[0], parts[1]
        else:
            return '', hostname, ''

    def calculate_tld_risk_score(self, domain, tld):
        """Match standalone’s TLD risk logic exactly."""
        full_domain = ""
        if domain and tld:
            full_domain = f"{domain}.{tld}".lower()
        elif domain:
            full_domain = domain.lower()
        elif tld:
            full_domain = tld.lower()

        # If it’s a major legitimate site or a trustworthy TLD
        if any(legit == full_domain for legit in MAJOR_LEGITIMATE_SITES) or \
           any(kw in full_domain for kw in TRUSTWORTHY_DOMAINS):
            return 0.0

        tld_risk_val = 0.0
        if tld and ('.' + tld.lower()) in RARE_TLDS:
            tld_risk_val = 0.9  # match standalone’s emphasis on rare TLDs

        if any(susp in full_domain for susp in SUSPICIOUS_DOMAINS):
            return 1.0

        return tld_risk_val

    def detect_url_shortener(self, url):
        """Detect common URL shorteners or very short hostnames (match standalone)."""
        try:
            parsed = urllib.parse.urlparse(url)
            netloc_lower = parsed.netloc.lower()
            shorteners = {"bit.ly", "goo.gl", "tinyurl.com", "t.co", "is.gd", "buff.ly", "adf.ly", "bc.vc"}
            if netloc_lower in shorteners:
                return 1
            # fallback short‐hostname heuristics
            patterns = [
                lambda d: len(d) <= 5 and '.' not in d,
                lambda d: bool(re.match(r'^[a-z][0-9][a-z]{1,2}\.[a-z]{2,3}$', d)),
                lambda d: bool(re.match(r'^[a-z]{1,3}\.[a-z]{2,3}$', d)) and d.split('.')[0] not in ['com','org','net','edu','gov']
            ]
            return 1 if any(p(netloc_lower) for p in patterns) else 0
        except Exception as e:
            self.logger.error(f"Error detecting URL shortener: {e}")
            return 0
