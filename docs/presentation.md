# Presentation: 4-minute talk

Six slides, 4:00 in total. Figures are in `reports/figures/` (300 dpi PNG), dashboard screenshots in
`reports/figures/dashboard/`, the pipeline picture is `docs/methods_overview.svg`. Every number below has its
`reports/results.json` key in brackets, so you can check it before the talk.

## Slide outline

### 1. Title and hook (0:00-0:25)
- **On the slide:** "AICIDENTIA: Learning from Failure" · tagline · `reports/figures/dashboard/01_home.png` (cropped to
  the five KPI cards).
- **Speaker notes (25 s):** "Since 2024, about 1,060 AI incidents have been reported, roughly 60% of everything in the AI
  Incident Database. One in six of them repeats a failure that had already happened. Today: what is growing, who is
  involved, whether the AI-generated labels can be trusted, and whether we could have found the earlier case."
  [`rq4.repeat`]

### 2. Problem and research questions (0:25-0:55)
- **On the slide:** four RQ lines; small inset `fig_audit_1_incidents_per_year.png`.
- **Speaker notes (30 s):** "AIID holds 1,775 reported incidents and 8,023 reports [`audit.n_incidents`,
  `audit.n_reports`]. RQ1: which risk themes emerge over time? RQ2: which organisations, systems and groups cluster
  together? RQ3: can we classify risk domains, and how reliable are AIID's AI-generated labels? RQ4: can we retrieve
  similar, earlier incidents? Caveat up front: these are reported incidents, not all AI harm."

### 3. Method in one picture (0:55-1:25)
- **On the slide:** `docs/methods_overview.svg`.
- **Speaker notes (30 s):** "CRISP-DM. One frozen snapshot, cleaned with written rules: strict duplicate merging, and
  every named person removed. Then four analyses, each chosen by a rule fixed before looking at results: NMF topics
  with trend tests; a Leiden network; classifiers with nested cross-validation plus three independent checks of the
  labels; and retrieval scored against links AIID's own editors made. Everything feeds one offline dashboard, and one
  command rebuilds it all."

### 4. Live demo (1:25-2:55, 90 s)
- **On the slide:** switch to the dashboard (`uv run streamlit run app/Home.py`, opened before the talk).
  Backup: `reports/demo/backup_demo.webm` (same path, slower), or screenshots 01-07.
- **Click path:** see the timed path below.

### 5. Key results and business value (2:55-3:35)
- **On the slide:** `fig_story_1_chatgpt_shift.png` (left) and `fig_rq3_5_label_reliability.png` (right); one line per finding.
- **Speaker notes (40 s):** "After ChatGPT, deliberate misuse rose from 12% to 52% of labelled incidents, the biggest
  shift in the mix [`story.chatgpt_malicious_share_*`]. Five of twelve themes are emerging: deepfakes, deepfake scams,
  hallucinated legal citations, AI nude images in schools, election disinformation [`rq1b.emerging_themes`]. The
  AI-generated labels agree with human editors far above chance, 75% against 30%, but about 15% are probably wrong
  [`rq3rel.pairs_similar`, `rq3rel.cl_noise_rate`]. And search finds a known precedent for 76% of recent repeats, where
  AIID's built-in suggestions score MRR 0.03 against our 0.45 [`rq4.repeat`, `rq4.mrr3`]. Business value: a risk team
  can triage a new incident in seconds and read what went wrong last time."

### 6. Limitations and close (3:35-4:00)
- **On the slide:** three limitations, closing line, QR/link to the repository.
- **Speaker notes (25 s):** "Three honest limits: reporting bias; human severity data stop in 2023, so we cannot yet rank
  the new themes by harm; and organisations are alleged, not found at fault. The point of AICIDENTIA is simple: most
  lessons are already in the data; we made them findable. Thank you."

## Demo click path (90 s, live)

| Time | Action | What to say |
|---|---|---|
| 1:25 | **Home**: point at the five finding cards | "Five findings, each with its number and its limitation." |
| 1:35 | **Themes and Trends**: heatmap, then the EMERGING badges in the table | "Darker = larger share; these five rise significantly." |
| 1:50 | Scroll to the **before/after ChatGPT** slope chart | "Blue rose after Nov 2022, red fell: misuse up, system failures down." |
| 2:05 | **Risk Network**: Meta is preselected; point at the ring and the domain bars | "Its neighbours and its incidents' risk domains; alleged involvement, no persons." |
| 2:20 | **Precedent Finder**: press the first **Demo** button (21 Oct 2026 deepfake case) | "A real incident from last week of the snapshot." |
| 2:30 | Point at the predicted domain (85% calibrated) and the five earlier incidents | "Predicted misuse, and five earlier deepfake cases from 2024-2026." |
| 2:40 | Scroll to **What went wrong before** | "What happened last time, one click from AIID." |
| 2:50 | **Label Reliability**: the three KPI cards | "And this is how far you can trust the labels." |

The recorded backup (`scripts/record_demo.py`) follows the same pages in the same order, with longer pauses and a
Tesla search on the network page, about 4 minutes.

## Likely examiner questions

1. **"AIID only has reported incidents; isn't every trend just media attention?"**
   Partly, and we say so. Trends are tested on *shares* of each year's incidents, not counts, which removes overall
   database growth; reporting-bias sensitivity (report-weighted shares, a later start year) is in `docs/rq1_trends.md`.
   Five themes stay significant on shares [`rq1b.emerging_themes`, `rq1b.themes_window_sensitive`].
2. **"Why NMF and not BERTopic?"**
   A rule fixed in advance ranked 26 configurations x 5 seeds on C_v, NPMI, diversity and stability; NMF k=12 won on all
   four and under every alternative weighting (C_v 0.794, stability 0.64) [`rq1a.selection.sensitivity`,
   `rq1a.chosen.c_v_mean`, `rq1a.chosen.stability`].
3. **"The MIT labels are generated by an AI. Why trust a classifier trained on them?"**
   We do not treat them as ground truth. Three independent checks: kappa 0.74 with human CSETv1 for discrimination,
   75% same-domain on editor-linked pairs against 30% by chance, and confident learning estimating 15.4% noise
   [`rq3rel.mapping_kappa`, `rq3rel.pairs_similar`, `rq3rel.cl_noise_rate`].
4. **"Did you hand-label anything?"**
   No. One annotator cannot separate their own errors from label errors; we used signals already in AIID instead
   (`docs/decisions.md` D1).
5. **"Why is DistilBERT not your best model?"**
   It was trained on the same folds and scores 0.628 macro-F1, below mpnet + LR at 0.651; it is better than TF-IDF on
   the 2024+ time split (+0.059, CI +0.014 to +0.104) but not in cross-validation [`rq3.metrics`, `rq3.significance`].
   With 1,496 short texts, frozen embeddings are enough.
6. **"Does removing the suspected wrong labels improve the classifier?"**
   No: +0.003 macro-F1 (CI -0.023 to +0.026) with a nested design in which test labels never decide what is removed
   [`rq3rel.retrain_diff`, `rq3rel.retrain_ci`]. A non-nested version looked better (+0.041) but leaked; we report the
   nested one (D8).
7. **"Why deploy BM25 when the hybrid scores higher?"**
   The hybrid's MRR gain (+0.022) is not significant by the pre-specified paired bootstrap (CI -0.002 to +0.046), and
   BM25 needs no model and runs offline [`rq4.comparisons_all`, `rq4.deployed_reason`].
8. **"How do you know the 'precedent' is real and not your own definition?"**
   It is AIID editors' own similar-incident links. 183 of 1,062 incidents since 2024 have an editor-linked earlier
   incident (a lower bound), and BM25 finds one in the top 5 for 76% (CI 69-82%) [`rq4.repeat`].
9. **"Your network names companies. Is that fair?"**
   Every organisation is an *alleged* developer or deployer as AIID records it; centrality measures how often an
   organisation is reported, not blame; all 1,150 named persons are excluded [`rules.entities.named_persons_excluded`].
   Communities change when hub entities are removed (ARI 0.32) [`rq2.hub_robustness`], which we report.
10. **"Can the risk matrix tell us which new risk is most severe?"**
    Not yet: human severity annotation (CSETv1) stops in 2023, so only 2 of 12 themes have the 30 annotations we
    require [`story.matrix_themes_supported`]. We show the gap instead of guessing; it is a clear next step for AIID.
