# External-evaluation models

Trained inference components for the common-cohort PhreshPhish comparisons.
`models.json` records compressed-file hashes and the SHA-256 of the unchanged,
decompressed joblib bytes. Load these trusted models only in the documented
reference environment. Compression does not alter learned parameters.

The five baseline folders contain the trained desktop classifier, feature
definitions and preprocessing metadata. Ebbu includes its lexical resources and
saved gibberish-scoring state. Roy et al. uses the 17-feature, 100-tree Random
Forest fitted on the canonical PhishFusion partition. TabNet denotes descriptor
selection followed by stacking. `representations/` contains the three controlled
source-trained LitePhish variants. The primary 568-feature LitePhish artifact is
in `artifacts/litephish`.

`source_definitions/` retains the training definitions needed to deserialize the
fixed models and reconstruct their input representation. The public evaluator
checks reference probabilities before full inference. No external labels are
used to fit or select any model.
