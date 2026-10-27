import pandas as pd
import re
import numpy as np
from urllib.parse import urlparse
from tld import get_tld

# --- Feature Extraction Logic (Faithful to Notebook Cells 7-21) ---

def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.0x[0-9a-fA-F]{1,2}\\.0x[0-9a-fA-F]{1,2}\\.0x[0-9a-fA-F]{1,2}\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        return 1
    else:
        return 0

def abnormal_url(url):
    try:
        hostname = urlparse(url).hostname
        hostname = str(hostname)
        match = re.search(hostname, url)
        if match:
            return 1
        else:
            return 0
    except:
        return 0

def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')

def shortening_service(url):
    # Exact regex list from the paper's code
    match = re.search('bit\\.ly|goo\\.gl|shorte\\.st|go2l\\.ink|x\\.co|ow\\.ly|t\\.co|tinyurl|tr\\.im|is\\.gd|cli\\.gs|'
                      'yfrog\\.com|migre\\.me|ff\\.im|tiny\\.cc|url4\\.eu|twit\\.ac|su\\.pr|twurl\\.nl|snipurl\\.com|'
                      'short\\.to|BudURL\\.com|ping\\.fm|post\\.ly|Just\\.as|bkite\\.com|snipr\\.com|fic\\.kr|loopt\\.us|'
                      'doiop\\.com|short\\.ie|kl\\.am|wp\\.me|rubyurl\\.com|om\\.ly|to\\.ly|bit\\.do|t\\.co|lnkd\\.in|'
                      'db\\.tt|qr\\.ae|adf\\.ly|goo\\.gl|bitly\\.com|cur\\.lv|tinyurl\\.com|ow\\.ly|bit\\.ly|ity\\.im|'
                      'q\\.gs|is\\.gd|po\\.st|bc\\.vc|twitthis\\.com|u\\.to|j\\.mp|buzurl\\.com|cutt\\.us|u\\.bb|yourls\\.org|'
                      'x\\.co|prettylinkpro\\.com|scrnch\\.me|filoops\\.info|vzturl\\.com|qr\\.net|1url\\.com|tweez\\.me|v\\.gd|'
                      'tr\\.im|link\\.zip\\.net',
                      url)
    if match:
        return 1
    else:
        return 0

def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0

def fd_length(url):
    urlpath = urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits

def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters

def extract_features(df):
    """
    Applies the exact 21 feature engineering steps.
    Expects a DataFrame with a 'url' column.
    """
    # Create a copy to avoid SettingWithCopy warnings
    data = df.copy()
    
    # Ensure URL is string
    data['url'] = data['url'].astype(str)

    # Feature Generation
    data['use_of_ip'] = data['url'].apply(lambda i: having_ip_address(i))
    data['abnormal_url'] = data['url'].apply(lambda i: abnormal_url(i))
    data['count.'] = data['url'].apply(lambda i: i.count('.'))
    data['count-www'] = data['url'].apply(lambda i: i.count('www'))
    data['count@'] = data['url'].apply(lambda i: i.count('@'))
    data['count_dir'] = data['url'].apply(lambda i: no_of_dir(i))
    data['count_embed_domian'] = data['url'].apply(lambda i: no_of_embed(i)) # Typo from original code preserved
    data['short_url'] = data['url'].apply(lambda i: shortening_service(i))
    data['count-https'] = data['url'].apply(lambda i : i.count('https'))
    data['count-http'] = data['url'].apply(lambda i : i.count('http'))
    data['count%'] = data['url'].apply(lambda i: i.count('%'))
    data['count?'] = data['url'].apply(lambda i: i.count('?'))
    data['count-'] = data['url'].apply(lambda i: i.count('-'))
    data['count='] = data['url'].apply(lambda i: i.count('='))
    data['url_length'] = data['url'].apply(lambda i: len(str(i)))
    data['hostname_length'] = data['url'].apply(lambda i: len(urlparse(i).netloc))
    data['sus_url'] = data['url'].apply(lambda i: suspicious_words(i))
    data['fd_length'] = data['url'].apply(lambda i: fd_length(i))
    
    # TLD extraction using tld library
    data['tld'] = data['url'].apply(lambda i: get_tld(i, fail_silently=True))
    data['tld_length'] = data['tld'].apply(lambda i: tld_length(i))
    
    data['count-digits'] = data['url'].apply(lambda i: digit_count(i))
    data['count-letters'] = data['url'].apply(lambda i: letter_count(i))
    
    # Select specific features for training
    feature_cols = [
        'use_of_ip','abnormal_url', 'count.', 'count-www', 'count@',
        'count_dir', 'count_embed_domian', 'short_url', 'count-https',
        'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
        'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
        'count-letters'
    ]
    
    return data[feature_cols]