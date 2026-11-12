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
