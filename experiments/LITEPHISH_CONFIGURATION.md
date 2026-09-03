# LitePhish experimental configuration

The focal parameters were selected using phishing-class F1 on the source-domain validation partition.

| Source dataset | Gamma | Alpha |
|---|---:|---:|
| PhishStorm | 0.55 | 0.90 |
| Ebbu2017 | 0.30 | 0.95 |
| PhishFusion | 0.55 | 0.95 |

Character 2-/3-gram selection uses `min_df=5`, `max_df=1.0`, variance threshold `1e-5`, equal mutual-information and L1-logistic weights, and a 5,000-feature candidate limit. SDCS uses 20 stability subsamples, sampling fraction 0.905, L1-logistic inverse-regularization `C=0.191`, stability threshold 0.705, mutual-information weight 0.318, stability weight 0.767, correlation threshold 0.956, and a maximum of 1,000 retained features.
