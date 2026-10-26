# Datasets

CSV columns are `url` and `phishing`, with phishing encoded as 1. The study uses PhishFusion, Ebbu2017, PhishStorm, and PhishTank. Labels and URLs are not rewritten to improve model results.

| File | Rows | Purpose |
|---|---:|---|
| `PhishFusion.csv` | 105,752 | Source model training and domain-disjoint evaluation |
| `ebbu2017.csv` | 73,575 | Internal and transfer comparisons |
| `phishstorm.csv` | 95,911 | Internal and transfer comparisons |
| `PhishTank.csv` | 93,788 | External PhishTank/Common Crawl detection evaluation |
| `merged.csv` | 256,409 | Descriptive exact-URL/label union, not model evaluation |

Phone resource measurements use a fixed **5,000-URL workload**, containing 2,705 benign and 2,295 phishing URLs. The frozen cohort and its sampling metadata are retained in [`experiments/android/bundle`](../experiments/android/bundle); the corresponding resource input is `Phishtank/PhishTank.csv`. The ordered row-index SHA-256 (int64 bytes) is `66fcfd4badd3d32a5aa54555d68a912c9130f4edebc618e1ede0e150d365dd79`.

## Descriptive union

| Class | Component records | Repeated occurrences removed | Union URLs | Union root domains |
|---|---:|---:|---:|---:|
| Phishing | 137,834 | 12,324 | 125,510 | 44,025 |
| Benign | 137,404 | 6,505 | 130,899 | 31,739 |

`merged.csv` retains the first occurrence of each literal `(url, phishing)` pair across PhishFusion, Ebbu2017, and PhishStorm in that order. No case folding or URL rewriting is applied. Root domains use the offline ICANN Public Suffix List. PhishTank is excluded from this union.

Treat URL strings as untrusted data. No workflow needs to visit the listed pages. Existing source attributions remain applicable; inclusion here does not grant ownership of third-party content or a right to redistribute it under an unrelated software license.
