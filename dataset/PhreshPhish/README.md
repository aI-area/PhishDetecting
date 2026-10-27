# PhreshPhish URL metadata

URL metadata from the public PhreshPhish test split, pinned to version
`eca603e0ecd4f698f8d4d226c088659e72681239`. HTML is not included.

| Population | Benign | Phishing | Total |
|---|---:|---:|---:|
| Released test split | 91,260 | 76,800 | 168,060 |
| Eligible URL-disjoint cohort | 89,686 | 76,541 | 166,227 |
| Source-domain-disjoint subset | 67,421 | 74,202 | 141,623 |

`test_url_metadata.csv.gz` preserves the release order, URL strings, source IDs,
labels, collection dates, language and shard provenance. `source_label` is
`benign` or `phish`; inference results use 0 and 1, respectively. Eligibility
excludes invalid/non-HTTP(S) entries, normalized-label conflicts, later normalized
duplicates and exact/normalized PhishFusion URL overlap. The stricter subset also
excludes source registered-domain overlap. The classifier receives the original
URL string. These are two analysis populations from one release.

Benign URLs combine browsing-derived and search-derived observations. Released
labels are retained; the cohort is not a probability sample of consumer browsing.
This dataset is separate from the 93,788-URL PhishTank cohort and the 5,000-URL
phone workload.

## Attribution and license

Thomas Dalton, Hemanth Gowda, Girish Rao, Sachin Pargi, Alireza Hadj Khodabakhshi,
Joseph Rombs, Stephan Jou and Manish Marwah. *PhreshPhish: A Real-World,
High-Quality, Large-Scale Phishing Website Dataset and Benchmark* (2025),
[arXiv:2507.10854](https://arxiv.org/abs/2507.10854).

Source: [PhreshPhish on Hugging Face](https://huggingface.co/datasets/phreshphish/phreshphish).
The upstream dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
for anti-phishing research. This export selects URL metadata without altering
the upstream URLs or labels. It contains no webpage content. Do not open dataset
URLs as part of evaluation.
