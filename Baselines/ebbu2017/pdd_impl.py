import pandas as pd
import numpy as np
import re
import math
import tldextract
import os
from urllib.parse import urlparse
from tqdm import tqdm


try:
    import editdistance
except ImportError:
   
    def editdistance_eval(s1, s2):
        if len(s1) < len(s2): return editdistance_eval(s2, s1)
        if len(s2) == 0: return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    class editdistance:
        @staticmethod
        def eval(a, b): return editdistance_eval(a, b)

--
def load_list_from_file(filename):
    possible_names = [filename, filename.replace('s.txt', '.txt'), filename.replace('.txt', 's.txt')]
    for name in possible_names:
        if os.path.exists(name):
            try:
                with open(name, 'r', encoding='utf-8', errors='ignore') as f:
                  
                    return set(line.strip().lower() for line in f if line.strip())
            except: pass
    return None


loaded_brands = load_list_from_file("allbrands.txt")
loaded_keywords = load_list_from_file("keywords.txt")
loaded_alexa = load_list_from_file("alexa_tld.txt") # NEW: Load Alexa


ALL_BRANDS = loaded_brands if loaded_brands else {
    'google', 'facebook', 'amazon', 'paypal', 'apple', 'microsoft', 'netflix', 
    'instagram', 'whatsapp', 'linkedin', 'yahoo', 'twitter', 'dropbox', 'ebay', 'adobe'
}
KEYWORDS = loaded_keywords if loaded_keywords else {
    'login', 'secure', 'account', 'update', 'verify', 'signin', 'banking', 
    'confirm', 'service', 'password', 'user', 'admin', 'webscr', 'support', 'bill'
}

ALEXA_DOMAINS = loaded_alexa if loaded_alexa else set()

if not ALEXA_DOMAINS:
    print("WARNING: 'alexa_tld.txt' not found or empty. Accuracy may drop.")
else:
    print(f"SUCCESS: Loaded {len(ALEXA_DOMAINS)} domains from Alexa list.")


class DictionaryChecker:
    def __init__(self):
        try:
            import enchant
            self.d = enchant.Dict("en_US")
            self.has_enchant = True
        except ImportError:
            self.has_enchant = False
            self.fallback_set = ALL_BRANDS.union(KEYWORDS).union({
                'home', 'index', 'html', 'content', 'site', 'web', 'org', 'net', 'com', 
                'info', 'contact', 'about', 'news', 'shop', 'store', 'blog', 'mail', 
                'server', 'app', 'mobile', 'free', 'online', 'video', 'image', 'help'
            })

    def check(self, word):
        if not word: return False
        word = word.lower()
        if self.has_enchant: return self.d.check(word)
        return word in self.fallback_set or len(word) < 3

dictionary_en = DictionaryChecker()

class GibberishDetector:
    def __init__(self):
        self.accepted_chars = 'abcdefghijklmnopqrstuvwxyz '
        self.pos = dict([(char, idx) for idx, char in enumerate(self.accepted_chars)])
        self.counts = [[10 for i in range(27)] for i in range(27)]
        self.threshold = 0.0
        self._train_improved_model()

    def _ngram(self, n, l):
        l = [c.lower() for c in l if c.lower() in self.accepted_chars]
        for start in range(0, len(l) - n + 1):
            yield ''.join(l[start:start + n])

    def _train_improved_model(self):
        text = " ".join(list(ALL_BRANDS)) + " " + " ".join(list(KEYWORDS))
        text += " the quick brown fox jumps over the lazy dog " * 50
        text += " information contact privacy policy terms support service account login home page search results " * 20
        for a, b in self._ngram(2, text):
            self.counts[self.pos[a]][self.pos[b]] += 1
        for row in self.counts:
            s = float(sum(row))
            for j in range(len(row)):
                row[j] = math.log(row[j] / s)
        self.threshold = self.avg_transition_prob("askjdfhlaskdjfh")

    def avg_transition_prob(self, l):
        log_prob = 0.0
        transition_ct = 0
        for a, b in self._ngram(2, l):
            log_prob += self.counts[self.pos[a]][self.pos[b]]
            transition_ct += 1
        return math.exp(log_prob / (transition_ct or 1))

    def is_gibberish(self, word):
        return self.avg_transition_prob(word) <= self.threshold


class WordSplitterClass:
    def _split(self, gt7_word_list):
        return_word_list = []
        for word in gt7_word_list:
            clean_word = re.sub(r"\d+", "", word)
            if not clean_word: continue
            if dictionary_en.check(clean_word):
                return_word_list.append(clean_word)
                continue
            found_split = False
            for number in range(len(clean_word), 3, -1):
                if found_split: break
                for l in range(0, len(clean_word) - number + 1):
                    sub = clean_word[l:l + number]
                    if dictionary_en.check(sub):
                        return_word_list.append(sub)
                        found_split = True
                        break
            if not found_split:
                return_word_list.append(clean_word)
        return return_word_list

class NLPManager:
    def __init__(self):
        self.word_splitter = WordSplitterClass()
        self.gib_detect = GibberishDetector()
        self.brands_list = list(ALL_BRANDS)
        self.keywords_list = list(KEYWORDS)

    def parse(self, words_raw):
        len_gt_7 = [w for w in words_raw if len(w) > 7]
        len_lt_7 = [w for w in words_raw if len(w) <= 7]
        
        splitted_words = self.word_splitter._split(len_gt_7)
        all_words_processed = len_lt_7 + splitted_words
        
        keyword_count = 0
        brand_count = 0
        similar_keyword_count = 0
        similar_brand_count = 0
        random_word_count = 0
        other_words_count = 0

        for w in all_words_processed:
            w_lower = w.lower()
            w_len = len(w_lower)
            
            # 1. Exact Match
            if w_lower in KEYWORDS:
                keyword_count += 1
                continue
            if w_lower in ALL_BRANDS:
                brand_count += 1
                continue
            
            other_words_count += 1
            
            # 2. Typosquatting
            is_similar = False
            for b in self.brands_list:
                if abs(w_len - len(b)) > 1: continue 
                if editdistance.eval(w_lower, b) == 1:
                    similar_brand_count += 1
                    is_similar = True
                    break
            if is_similar: continue
            
            for k in self.keywords_list:
                if abs(w_len - len(k)) > 1: continue 
                if editdistance.eval(w_lower, k) == 1:
                    similar_keyword_count += 1
                    is_similar = True
                    break
            if is_similar: continue

            # 3. Random Check
            if self.gib_detect.is_gibberish(w_lower):
                random_word_count += 1

        lengths = [len(w) for w in words_raw]
        
        target_brand_count = brand_count + similar_brand_count
        target_keyword_count = keyword_count + similar_keyword_count

        return {
            'raw_word_count': len(words_raw),
            'average_word_length': np.mean(lengths) if lengths else 0,
            'longest_word_length': max(lengths) if lengths else 0,
            'shortest_word_length': min(lengths) if lengths else 0,
            'std_word_length': np.std(lengths) if lengths else 0,
            'adjacent_word_count': len(len_gt_7),
            'average_adjacent_word_length': np.mean([len(w) for w in len_gt_7]) if len_gt_7 else 0,
            'separated_word_count': len(splitted_words),
            'keyword_count': keyword_count,
            'brand_name_count': brand_count,
            'similar_keyword_count': similar_keyword_count,
            'similar_brand_count': similar_brand_count,
            'random_word_count': random_word_count,
            'target_brand_name_count': target_brand_count,
            'target_keyword_count': target_keyword_count,
            'other_words_count': other_words_count,
        }


class URLRules:
    def __init__(self):
        self.nlp_manager = NLPManager()

    def rules_main(self, domain, tld, subdomain, path, words_raw):

        features = self.nlp_manager.parse(words_raw)
        

        full_url = str(domain) + str(subdomain) + str(path)
        features['char_dot'] = full_url.count('.')
        features['char_hyphen'] = full_url.count('-')
        features['char_underscore'] = full_url.count('_')
        features['char_slash'] = full_url.count('/')
        features['char_at'] = full_url.count('@')
        features['char_question'] = full_url.count('?')
        features['char_ampersand'] = full_url.count('&')
        features['char_equal'] = full_url.count('=')

     
        features['digit_domain'] = sum(c.isdigit() for c in domain)
        features['digit_subdomain'] = sum(c.isdigit() for c in subdomain)
        features['digit_path'] = sum(c.isdigit() for c in path)

        
        features['len_domain'] = len(domain)
        features['len_subdomain'] = len(subdomain)
        features['len_path'] = len(path)

       
        features['brand_check_domain'] = 1 if domain in ALL_BRANDS else 0
        features['is_random_domain'] = 1 if self.nlp_manager.gib_detect.is_gibberish(domain) else 0
        features['is_known_tld'] = 1 if tld in {'com', 'net', 'org', 'edu', 'gov', 'uk', 'de'} else 0

       
        features['is_www'] = 1 if 'www' in words_raw else 0
        features['is_com'] = 1 if 'com' in words_raw else 0
        features['is_punycode'] = 1 if 'xn--' in domain else 0
        
       
        cons_repeat = 0
        for i in range(len(full_url)-2):
            if full_url[i] == full_url[i+1] == full_url[i+2]:
                cons_repeat = 1
                break
        features['consecutive_char_repeat'] = cons_repeat

        
        full_domain = f"{domain}.{tld}" if tld else domain
        features['alexa_tld_check'] = 1 if full_domain in ALEXA_DOMAINS else 0
       
        features['alexa_no_tld_check'] = 1 if domain in ALEXA_DOMAINS else 0
        
        return features

class DomainParser:
    def parse(self, url):
        url = str(url).strip().replace('"', "").replace("'", "")
        if not url.startswith(('http', 'https')): url = 'http://' + url
        try:
            ext = tldextract.extract(url)
            parsed = urlparse(url)
            path = parsed.path
            if parsed.query: path += "?" + parsed.query
            
            raw_text = f"{ext.domain} {ext.subdomain} {path}".lower()
            words_raw = re.split(r"[\-\.\/\?\=\@\&\%\:\_ ]+", raw_text)
            words_raw = [w for w in words_raw if w]
            
            return {
                'domain': ext.domain, 'subdomain': ext.subdomain,
                'tld': ext.suffix, 'path': path, 'words_raw': words_raw, 'valid': True
            }
        except: return {'valid': False}

def extract_features_from_dataframe(df):
    parser = DomainParser()
    rules = URLRules()
    data_list = []
    print("Extracting features with Alexa Check...")
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing URLs"):
        p = parser.parse(row['url'])
        if not p['valid']: continue
        feats = rules.rules_main(p['domain'], p['tld'], p['subdomain'], p['path'], p['words_raw'])
        feats['labels'] = row['phishing']
        data_list.append(feats)
    return pd.DataFrame(data_list)