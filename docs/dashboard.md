# AICIDENTIA Intelligence: the dashboard

```bash
uv run streamlit run app/Home.py
```

The dashboard only *reads* saved results (`reports/`, `data/processed/`, `models/`); it never recomputes an
analysis and never downloads a model, so it works offline. In a real browser every page reaches its footer in
0.2-2.0 s (`scripts/capture_dashboard.py`); the test suite checks that each page renders in under 3 s after a
warm-up (`tests/test_phase8.py`).

## Design

Insight first, calm and readable from the back of a room: one theme in `.streamlit/config.toml` (light, Okabe-Ito
colour-blind-safe palette, one accent colour `#0072B2`), a header with the AICIDENTIA name and tagline on every
page, KPI cards, a caption under every chart, axis titles with units, and a footer "Source: AIID snapshot
2026-10-26 · reported incidents only".

**Global filters** (sidebar): year range, MIT risk domain, CSETv1 sector. They are kept in session state and
applied on every page that lists incidents; each page says which of its elements they affect. Only 207 incidents
have a CSETv1 sector, so a sector filter keeps only those (the page says so). Tables of incidents link to
`https://incidentdatabase.ai/cite/<id>` and have a CSV download.

## Pages

| Page | What it shows | Screenshot |
|---|---|---|
| Home / Key Findings | The five findings from `docs/key_findings.md` as cards (number, sentence, limitation, link to the page with the evidence), five headline KPIs, how to use | `reports/figures/dashboard/01_home.png` |
| Themes and Trends | Theme x year heatmap (filtered), emerging-themes table with Sen's slope, q-value and an EMERGING badge, before/after-ChatGPT slope charts for domains and themes, representative incidents with AIID links | `02_themes_and_trends.png` |
| Risk Priority Matrix | Trend (Sen's slope) x severity (CSETv1 tangible harm, injuries or deaths) bubble chart with quadrant labels; themes with fewer than 30 CSETv1 annotations in grey with their n; click a bubble to list its incidents | `03_risk_priority_matrix.png` |
| Risk Network | Search an organisation, system or group; radial ego network (the entity and its top-N co-occurring neighbours), colour by community or node type with legend, year slider, top-N slider; side panel with risk domains, sectors, incident count, roles and linked incidents | `04_risk_network.png` |
| Precedent Finder | Paste an incident: predicted MIT risk domain with calibrated confidence and the top 3 alternatives; the 5 most similar EARLIER incidents with dates and links; "what went wrong before" with their descriptions, MIT subdomain and GMF technical failure; Demo mode with 3 real incidents from Sep-Oct 2026 (`rq4.demo_cases`) | `05_precedent_finder.png` (after pressing the first demo case) |
| Label Reliability | The three RQ3 checks in plain language: kappa against CSETv1/GMF fields, same-domain rate on editor-linked pairs vs chance, confident-learning noise per domain and the confusion structure, most suspicious labels | `06_label_reliability.png` |
| Method and Limitations | Pipeline summary, active classifier and search method and why (decided from the local model cache without loading anything), model comparison tables with CIs, ethics (alleged framing, excluded persons, reporting bias), limitations, data snapshot and citations | `07_method_and_limitations.png` |

## Which models are active (offline-first, with fallbacks)

* **Classifier.** mpnet embeddings + logistic regression with temperature scaling (`models/rq3_domain_classifier.joblib`);
  it needs `sentence-transformers/all-mpnet-base-v2` in the local Hugging Face cache (`HF_HOME` in `.env`). If the model is
  missing, the dashboard uses TF-IDF + logistic regression (`models/rq3_domain_classifier_fallback.joblib`).
* **Search.** The Phase 7 deployed method (BM25), which needs no model. The Precedent Finder searches only incidents
  earlier than the case (for typed text: earlier than the snapshot date) and excludes the case itself.
* Models are loaded on the first Precedent Finder query and then cached for the session.

## Screenshots and the backup video

```bash
uv run python scripts/capture_dashboard.py   # PNG of every page, 1600 px wide at 2x, to reports/figures/dashboard/
uv run python scripts/record_demo.py         # 4-minute click path to reports/demo/backup_demo.webm (git-ignored)
```

Both start the dashboard on a free local port and drive it with Playwright, using Playwright's Chromium if installed,
otherwise the Chrome or Edge already on the machine. Video recording needs Playwright's ffmpeg, installed
automatically into `PLAYWRIGHT_BROWSERS_PATH` (set in `.env`).

## Refresh everything

```bash
uv run python scripts/refresh_all.py
```

Steps: rules (D2/D3) -> phase2 -> phase3 -> phase4 -> phase5 -> phase6 (+ DistilBERT predictions, label reliability)
-> phase7 -> story -> dashboard check. A failed or blocked step blocks only the steps that depend on it; the summary
is printed and written to `reports/refresh_summary.md`. In a running dashboard press **Reload data after a refresh**.
`scripts/run_all.py` regenerates everything from `data/raw` (see the README).
