# Decisions and rules

Every rule applied to the data or the analysis, why it was chosen, and its effect in numbers. Numbers come
from `reports/results.json` (key in brackets) so they can be cited and re-checked. Earlier analytic rules
(model-selection rules, deployment rules) are stated in the RQ documents; the ones that shape the final results
are repeated here so the report can cite one place.

## D1. No manual label audit

**What.** The planned blind audit (150 MIT-labelled incidents hand-annotated by the author) was dropped. The
template, its git-ignored answer key, the annotation app, the label guidelines, the dashboard's
"hide labels while the audit is open" masking, the refresh step that waited for annotations, and their tests were
removed. None of the 150 items had been annotated.

**Why.** A single annotator gives weak evidence: with one person there is no inter-annotator agreement, so a
disagreement with the MIT label cannot be attributed to the label rather than to the annotator's own errors or
reading of the taxonomy. Reliability is instead assessed with independent signals that already exist in AIID
(human CSETv1 and GMF annotations, editor-linked incident pairs, and confident learning on out-of-fold
predictions; see D6-D8).

**Effect.** RQ3's second half is reframed as "How reliable are the AI-generated risk labels, judged against
independent signals already in AIID?". The dashboard shows every MIT label. The small-cell rule for network
communities (domain mix hidden for communities with fewer than 20 incidents) is kept as a general disclosure
rule, not as audit protection.

## D2. Duplicate pairs merged only by a strict rule

**What.** AIID's own duplicate list stays applied (13 pairs [`prep.aiid_merge_pairs`]). For the extra
near-duplicate candidates (TF-IDF cosine >= 0.6 on title + description), a pair is merged only when ALL hold:
(1) the two incidents share at least one deployer entity, (2) their incident dates are at most 7 days apart,
(3) text similarity >= 0.90. Every other pair is `keep_separate`. Implemented in `aicidentia.data.rules`
(`scripts/apply_rules.py`, also the first step of `scripts/refresh_all.py`); each row's `notes` says which
criteria failed.

**Why.** AIID deliberately records separate incidents for similar events (the same system failing again, the
same scam against different victims). Merging on similarity alone would erase exactly the repeated failures
this project studies. Requiring the same deployer and the same week as well as near-identical text targets
true double entries only.

**Effect.** 8 candidate pairs [`rules.duplicates.pairs`]; 0 merged [`rules.duplicates.merged`], 8 kept
separate. 7 pairs fail the similarity criterion [`rules.duplicates.fail_similarity`], 5 fail the 7-day
criterion [`rules.duplicates.fail_dates`], none fails the deployer criterion. The only pair above 0.90 (cosine
0.946) describes two different wrongful arrests by the same police department 183 days apart. The incident set
is unchanged (1,775 incidents [`prep.incidents_clean`]), so folds and the DistilBERT input are unchanged.

## D3. Every named person is excluded

**What.** All 1,346 person-like entity names flagged in Phase 2 [`rules.entities.candidates`] were read. Every
named person, private or public, is excluded (`is_named_person = yes`); names that are not persons are kept and
re-typed in `config/entity_types.yaml` (`overrides`): organisations (companies, media outlets, government
bodies, threat-actor groups, venues), products/systems, and population groups. The switch
`privacy.exclude_unreviewed_individual_candidates` is now on, so any person-like name added by a future
snapshot is excluded until reviewed.

**Why.** RQ2 and the dashboard are about organisations, systems and groups. Distinguishing private from
public persons is a judgement with privacy risk and no analytic value here, so no person is shown at all.

**Effect.** 1,150 named persons excluded [`rules.entities.named_persons_excluded`]; 196 non-persons kept
[`rules.entities.not_persons_kept`]: 177 organisations, 11 products/systems, 8 population groups
[`rules.entities.kept_as_*`]. Entity mentions fall from 18,797 to 17,418 [`prep.incident_entities_rows`];
55 incidents lose their only named deployer (1,720 of 1,775 keep one [`prep.incidents_with_deployer`]).
The entity network gains 9 nodes (1,005 -> 1,014 [`rq2.entity_network_nodes`]) and 9 organisations
(181 -> 190 [`rq2.organisations_in_network`]); Leiden still finds 11 communities with modularity 0.362
[`rq2.chosen_modularity`]. Incident titles and descriptions are not edited (they come from AIID and the
dashboard links back to AIID), so names can still appear inside incident text.

## D4. Topic labels

**What.** `final_label` in `data/manual/topic_labels.csv` was first copied from `suggested_label`, then every
label was replaced by a readable name, because the suggested labels are only the three top terms
("video / depict / circulate"). Each name was proposed from the topic's top 10 terms and its five most
representative incidents (highest topic weight), plus a random sample of 12 member titles for the broad
topics, and reviewed by the author.

| Topic | Suggested (top terms) | Final label |
|---|---|---|
| 0 | video / depict / circulate | Deepfake videos and AI hoaxes |
| 1 | data / agent / openai | LLM chatbots and agents |
| 2 | court / citation / legal | Hallucinated legal citations |
| 3 | vehicle / autonomous / drive | Autonomous vehicles and robotaxis |
| 4 | student / school / image | AI nude images in schools |
| 5 | facial / recognition / arrest | Facial recognition and false arrests |
| 6 | election / campaign / disinformation | AI election disinformation |
| 7 | scam / impersonate / deepfake | Deepfake impersonation scams |
| 8 | content / facebook / moderation | Content moderation failures |
| 9 | tesla / autopilot / crash | Tesla Autopilot crashes |
| 10 | plate / license / reader | License-plate reader errors |
| 11 | google / bias / image | Algorithmic bias in search and images |

**Genuinely mixed topics.** Topic 1 "LLM chatbots and agents" (311 incidents, the largest) is a catch-all for
LLM chatbot and agent failures, data exposure and training-data problems. Topic 11 "Algorithmic bias in search
and images" (181) has a clear bias core (image search, image generation, translation, hiring ads) plus other
search-product failures. Both names describe the majority, not every member.

**Effect.** No two topics share a label, so no themes are merged; Phase 4 is no longer provisional
[`rq1b.provisional` = false, `rq1b.labels`]. Trend results are unchanged except for the names
(5 emerging themes [`rq1b.emerging_themes`]).

## D5. DistilBERT trained on the local GPU

**What.** DistilBERT (`distilbert-base-uncased`) was fine-tuned on the local NVIDIA GeForce RTX 3050 (6 GB)
with `scripts/train_distilbert_local.py`, which imports the training code from `scripts/colab_distilbert_code.py`
(the same file the Colab notebook embeds): the same 5 folds and time split from
`reports/colab/rq3_colab_input.csv`, seed 42 + fold, 4 epochs, lr 5e-5, batch 16, max length 128,
class-weighted loss, fp32. Predictions go to `reports/colab/rq3_distilbert_predictions.csv`, where Phase 6
expects them; device, versions and run time are in `reports/colab/rq3_distilbert_run.json` (7.8 minutes).
The Colab notebook stays as the documented alternative.

**Why.** A local GPU was available; identical code and folds keep the comparison with the other models fair.

**Effect.** DistilBERT scores macro-F1 0.628 [0.590, 0.664] in cross-validation and 0.551 on the 2024+ time
split [`rq3.metrics`]. It does not beat TF-IDF + logistic regression in cross-validation (+0.015, 95% CI
[-0.020, +0.048]) but does on the time split (+0.059 [+0.014, +0.104]) [`rq3.significance`]. The best model
remains mpnet embeddings + logistic regression (0.651) [`rq3.best_model`]. GPU kernels are not bit-for-bit
deterministic, so a retrain can differ slightly; the saved predictions file is what the pipeline scores.

## D6. Label reliability check (a): mappings from MIT domains to human fields

**What.** Each MIT domain is compared with the human CSETv1/GMF field closest to it in meaning, as a yes/no
pair, on incidents that carry both (CSETv1: 207 MIT-labelled incidents, all from 2023 or earlier; GMF: 41)
[`rq3rel.cset_mit_overlap`, `rq3rel.gmf_mit_overlap`]. Human answers 'maybe', 'unclear' or empty are excluded.
Cohen's kappa with a percentile bootstrap CI (2,000 resamples) [`rq3rel.mappings`]:

| MIT domain (yes/no) | Human field (yes/no) | n | Kappa [95% CI] |
|---|---|---|---|
| Discrimination and Toxicity | CSETv1 Protected Characteristic = yes | 196 | 0.74 [0.64, 0.83] |
| Discrimination and Toxicity | CSETv1 Harm Distribution Basis not 'none' | 194 | 0.73 [0.63, 0.83] |
| AI system safety, failures, and limitations | CSETv1 Tangible Harm occurred or imminent | 197 | 0.40 [0.27, 0.53] |
| AI system safety, failures, and limitations | GMF a Technical Failure is recorded | 41 | -0.00 [-0.29, 0.29] |
| Misinformation or Malicious Actors & Misuse | CSETv1 Detrimental Content = yes | 193 | 0.31 [0.13, 0.47] |

**Why.** No human field in AIID is a risk-domain label, so only partial mappings are defensible. Each was chosen
from the field definitions in the snapshot's taxonomy records (for example, CSET defines detrimental content as
"deepfakes, identity misrepresentation, ... misinformation, ... scam emails"). These are convergent-validity
checks, not ground truth: agreement is capped by definitional mismatch.

**Effect.** Where the concepts coincide (discrimination) agreement is substantial; where they only overlap it is
moderate (safety vs tangible harm) or fair (misinformation/misuse vs detrimental content); GMF has too few
incidents to conclude anything. Coverage limits these checks to pre-2024 incidents.

## D7. Label reliability check (b): editor-linked pairs

**What.** For the 325 incident pairs AIID editors marked as similar (the RQ4 ground truth), the share with the
same MIT domain, among the 249 pairs where both are labelled; bootstrap CI over pairs. Baseline: 2,000
permutations in which each incident of a pair is replaced by a random MIT-labelled incident from the same year.
The same is computed for the 349 editor-marked dissimilar pairs [`rq3rel.pairs_similar`,
`rq3rel.pairs_dissimilar`].

**Why.** Editors' links are an existing human judgement about which incidents belong together; if the automatic
labels are consistent, linked incidents should share a domain far more often than chance. Matching on year
removes the effect of the domain mix changing over time.

**Effect.** 75.1% of editor-similar pairs share a domain (95% CI 69.5-80.3%) against 29.7% for year-matched
random pairs (p = 0.0005; 2.5 times the baseline). Editor-dissimilar pairs: 28.1%, at their own baseline (26.1%,
p = 0.21). The most frequent mixed pair is Malicious Actors & Misuse with Misinformation (13 pairs).

## D8. Label reliability check (c): confident learning, and the cleaned-label retrain

**What.** cleanlab 2.9.0 (the latest stable release; it installed cleanly) with default settings
(`prune_by_noise_rate`) on the temperature-scaled out-of-fold probabilities of the best classifier
(mpnet embeddings + logistic regression). It estimates the joint distribution of given and latent labels,
the noise rate per domain, and flags likely mislabelled incidents [`rq3rel.cl_*`]. The sensitivity check
retrains the same model without flagged incidents. To keep it leakage-free, flags are recomputed inside every
outer training part (inner 5-fold cross-validation, then confident learning on training incidents only), and
the untouched outer test fold is scored against the original MIT labels [`rq3rel.retrain_*`].

**Why.** Confident learning (Northcutt, Jiang & Chuang, 2021) estimates label noise without any extra
annotation. A first, non-nested version flagged incidents once on all data and appeared to raise macro-F1 by
0.041; that design lets test labels influence which training incidents are removed, so it was replaced by the
nested design, which is the only version reported.

**Effect.** Estimated share of wrong labels 15.4%; 214 of 1,496 incidents (14.3%) flagged
[`rq3rel.cl_noise_rate`, `rq3rel.cl_flagged`]. Highest estimated noise: Human-Computer Interaction 26.2% and
Socioeconomic & Environmental Harms 25.0%; lowest: Malicious Actors & Misuse 12.5%
[`rq3rel.cl_noise_by_domain`]. Largest exchange: Malicious Actors & Misuse vs Misinformation (52 estimated
incidents, 40 of them one way), read as taxonomy overlap. Retraining without flagged incidents changes macro-F1
by +0.003 (95% CI [-0.023, +0.026], p = 0.81) [`rq3rel.retrain_diff`]: no effect on performance.
