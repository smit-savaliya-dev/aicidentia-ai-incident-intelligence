# Model card: AICIDENTIA risk-domain classifier and precedent search

_Structure follows Mitchell et al. (2019), "Model Cards for Model Reporting". Every number has its
`reports/results.json` key in brackets; values are for AIID snapshot 2026-10-26._

## 1. Model details

| | Risk-domain classifier (deployed) | Fallback classifier | Precedent search (deployed) |
|---|---|---|---|
| What | Frozen `sentence-transformers/all-mpnet-base-v2` embeddings + multinomial logistic regression, temperature-scaled [`rq3.deployable`, `rq3.calibration_method`] | TF-IDF (1-2-grams) + logistic regression, temperature-scaled [`rq3.fallback_model`] | BM25 (k1 1.5, b 0.75) over title + description [`rq4.deployed`] |
| File | `models/rq3_domain_classifier.joblib` (git-ignored) | `models/rq3_domain_classifier_fallback.joblib` | built at run time from `data/processed/incidents_clean.parquet` |
| Output | One of 7 MIT risk domains with a calibrated probability for each | same | Ranked incident ids (earlier incidents only in the Precedent Finder) |
| Trained on | 1,496 MIT-labelled incidents [`rq3.labelled`] | same | 1,775 incidents (no training) [`rq4.indexed_incidents`] |
| Selection rule | Best cross-validated macro-F1, kept only if significantly better than TF-IDF + LR (paired bootstrap) | used when the embedding model is not in the local cache | Best MRR, kept only if significantly better than BM25; the hybrid was not, so BM25 |
| Author / date | Project author, Nov 2026 | | |

Other models compared but not deployed: majority class, TF-IDF + linear SVM, MiniLM embeddings + LR, DistilBERT
fine-tuned on the local GPU (same folds), TF-IDF/MiniLM/mpnet/hybrid retrieval.

## 2. Intended use

- **Intended:** suggest a likely MIT risk domain for a new incident description, and surface earlier AIID
  incidents that resemble it, to support analysts triaging or studying AI incidents (the dashboard's Precedent Finder).
- **Users:** analysts, students and researchers who read the suggestion together with the source incidents.
- **Out of scope:** automatic labelling without human review; judging responsibility or legal fault; any use on
  individuals; non-English text; incidents unlike AIID's news-based descriptions.

## 3. Factors

- **Domain size.** Performance is much lower for the two smallest domains (Human-Computer Interaction, 42 incidents;
  Socioeconomic & Environmental Harms, 24) [`rq3.class_counts`, `reports/tables/rq3_per_class.csv`].
- **Time.** The mix of incidents shifts after 2022 (Malicious Actors & Misuse rises from 12% to 52% of labelled
  incidents [`story.chatgpt_malicious_share_after`]), so performance on future incidents is best estimated by the
  time split.

## 4. Metrics and evaluation data

- Classifier: macro-F1 (primary; classes are unequal), accuracy, weighted-F1, per-class F1; stratified 5-fold
  cross-validation with nested tuning, and a time split (train before 2024, test 2024+); bootstrap 95% CIs;
  calibration by expected calibration error (ECE).
- Search: MRR (primary), nDCG@10, P@5, Hit@5 against 447 queries with editor-marked similar incidents [`rq4.queries`];
  paired bootstrap tests against BM25.

## 5. Results

| Model | Macro-F1, 5-fold CV [95% CI] | Macro-F1, time split 2024+ |
|---|---|---|
| mpnet embeddings + LR (deployed) | 0.651 [0.616, 0.684] | 0.581 |
| TF-IDF + linear SVM | 0.629 | 0.483 |
| DistilBERT fine-tuned (local GPU) | 0.628 [0.590, 0.664] | 0.551 |
| TF-IDF + LR (fallback) | 0.614 [0.570, 0.652] | 0.492 |
| MiniLM embeddings + LR | 0.605 | 0.494 |
| Majority class | 0.078 | 0.026 |

[`rq3.metrics`]. mpnet + LR beats TF-IDF + LR by +0.037 (95% CI +0.003 to +0.074) [`rq3.significance`].
Calibration: ECE 0.050 uncalibrated, 0.020 after temperature scaling [`rq3.ece`].

Search: BM25 MRR 0.489, hybrid 0.510 (not significantly better, +0.022, CI -0.002 to +0.046) [`rq4.mrr`,
`rq4.comparisons_all`]. MRR@3 0.446 against 0.034 for AIID's built-in suggestions [`rq4.mrr3`, `rq4.aiid_nlp_rr3`].
For incidents since 2024 that have an editor-linked earlier precedent, BM25 finds one in the top 5 for 76%
(95% CI 69-82%) when searching only earlier incidents [`rq4.repeat`].

## 6. Training data and its labels

The target is the **MIT risk domain, which AIID generated automatically**, not a human label. Its reliability was
checked with independent signals (`docs/rq3_classification.md`, section 5): substantial agreement with human CSETv1
annotation for discrimination (kappa 0.74), 75% same-domain rate on editor-linked similar pairs against 30% by
chance, and an estimated 15.4% label noise by confident learning [`rq3rel.*`]. The classifier therefore learns, and
reproduces, an automatic labeller with some noise; a high score is agreement with that labeller.

## 7. Ethical considerations

- Organisations are *alleged* developers or deployers; neither model attributes blame.
- No named persons are used as features or shown as entities (rule D3); incident texts are AIID's own.
- AIID over-represents English-language, high-profile news; suggestions inherit that bias.

## 8. Caveats and recommendations

- Show the calibrated confidence and the top alternatives (the dashboard does); treat confidence below about 50% as
  "unclear between domains".
- Misuse vs misinformation is the most frequent confusion and is partly built into the taxonomy (`docs/decisions.md` D8).
- Retrain after a new AIID snapshot (`scripts/run_all.py`); the classifier was fitted on incidents up to Oct 2026.
- The embedding model must be in the local cache for the deployed classifier; otherwise the fallback is used and
  the dashboard says so.
