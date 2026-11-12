# AICIDENTIA Intelligence: dashboard and one-command refresh (Phase 8)

## Run the dashboard

```bash
uv run streamlit run app/Home.py
```

The dashboard only *reads* saved results (`reports/`, `data/processed/`, `models/`); it never recomputes an analysis and never downloads a model.

| Page | What it shows | Main caveat shown on the page |
|---|---|---|
| Home | Key numbers from `reports/results.json` and how to read the dashboard | AIID records reported incidents only |
| 1 Overview | Incidents per year (partial year hatched), MIT domain distribution, incident browser with search and filters | MIT labels are machine-generated |
| 2 Themes and Trends (RQ1) | Theme x year heatmap, theme lines (share or count, partial year hollow), Mann-Kendall / Sen / BH table, sensitivity to the measure, MIT-domain trends, forecasting verdict, incidents per theme | Theme names are provisional until you label topics; counts track database growth |
| 3 Risk Network (RQ2) | Interactive entity network with a year slider (cumulative), node-type filter, entity profile panel (communities, PageRank, recent incidents), community table, top organisations | Organisations are *alleged*; centrality is not blame; communities depend on hubs |
| 4 Classify and Compare (RQ3, RQ4) | Free-text risk-domain suggestion with calibrated confidence; similar past incidents (method selectable); model comparison; blind-audit status or result | The classifier reproduces machine-generated labels |
| 5 Method and Limitations | **Active classifier and retrieval method, and whether a fallback is in use**; method summary; status of your manual files; the refresh command; limitations; software versions | |

### Which models are active (offline-first with fallbacks)

* **Classifier.** The deployed model is mpnet embeddings + logistic regression with temperature scaling (`models/rq3_domain_classifier.joblib`). It needs `sentence-transformers/all-mpnet-base-v2` in the local Hugging Face cache (`HF_HOME`, set in `.env`); the dashboard loads it with `local_files_only=True`. If that fails, it automatically uses **TF-IDF + logistic regression** (`models/rq3_domain_classifier_fallback.joblib`, also temperature-scaled). The Method page says which one is active and why.
* **Similar-incident search.** The Phase 7 deployed method (BM25). Embedding methods and the hybrid can be selected when their query encoder is in the local cache; otherwise only TF-IDF and BM25 are offered. Corpus embeddings come from the disk cache in `data/interim/`.
* Nothing is ever downloaded at run time.

### Blind label audit

While `data/manual/label_audit_sample.csv` is not fully annotated, the MIT label of every audit incident is shown as *hidden (blind audit in progress)* in the incident browser, the network panel's domain mix is withheld, and community domain mixes are suppressed for communities with fewer than 20 incidents (permanently, as a small-cell rule). When all 150 items are annotated, labels reappear after "Reload data".

## Refresh everything from your manual files

```bash
uv run python scripts/refresh_all.py
```

It prints the status of your manual files, then runs, in dependency order:

| Step | What it re-applies | Typical time (no inputs changed) |
|---|---|---|
| phase2 | `entity_review.csv` (privacy filter), `duplicate_review.csv` (merges) | 11 s |
| phase3 | topic models (cached; refits only if phase2 changed the incidents) | 6 s |
| phase4 | `topic_labels.csv` (theme names; same label = merged theme) | 44 s |
| phase5 | network (uses phase2's entities) | 15 s |
| phase6 | classifiers; scores `reports/colab/rq3_distilbert_predictions.csv` if present and valid | 24 s |
| phase6_audit | `label_audit_sample.csv` (kappa etc.); runs only when **all** items are annotated | 5 s |
| phase7 | retrieval | 43 s |
| dashboard | renders every page headlessly | 15 s |

Total about 2.5 minutes. A failed or blocked step blocks only the steps that depend on it; the summary is printed and written to `reports/refresh_summary.md`; the exit code is 1 if anything failed. Options: `--check` (status only), `--only phase4 phase6_audit`, `--from phase4`, `--partial-audit`, `--no-dashboard-check`. In a running dashboard press **Reload data after a refresh** in the sidebar.

### Recommended order for your manual work

Some inputs change what later inputs refer to, so this order avoids redoing work:

1. **Duplicate review and entity review first**, then run the refresh. A merge removes an incident, which changes (a) the topic-model corpus, so the topic solution may change and your topic labels would no longer apply (the refresh then *blocks* Phase 4 and writes `topic_labels.proposed.csv`), and (b) the cross-validation folds, so DistilBERT predictions made before the merge are rejected (Phase 6 then runs without them and says so).
2. **Topic labels**, the **Colab DistilBERT run** (re-export `reports/colab/rq3_colab_input.csv` comes from the refresh), and the **blind annotation**, in any order.
3. Run the refresh again.

If you make no merges (all `keep_separate`), steps 1 and 2 can happen in any order.

## Reproducibility notes

* Re-running a phase with unchanged inputs reproduces its outputs byte for byte (figures carry no save date and fixed SVG ids; network exports and layouts no longer depend on Python's per-process string hashing).
* Machine-specific settings live in `.env` (git-ignored; template `.env.example`): `HF_HOME` for models. The uv cache location is set in the user-level `uv.toml`.
