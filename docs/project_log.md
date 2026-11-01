# Project log

Keep one row per work session. Time spent is in hours.

| Date | Task | Time spent (h) | Notes |
|------|------|----------------|-------|
| 2026-11-01 | Project skeleton, environment (uv, Python 3.13.12), dependencies | | Scaffold only; no data downloaded, no analysis yet |
| 2026-11-01 | Phase 1: data audit (Excel + snapshot), datasheet, audit figures/tables, results registry | | Snapshot 20261026. MIT labels are machine-generated; MIT coverage falls to 34.7% for 2026; editor-similar links give 447 RQ4 queries. Time spent: fill in. |
| 2026-11-01 | Phase 2: data preparation (clean parquet tables, entity aliases/types, validation, review templates, EDA figures) | | 22 validation checks pass; 1,346 person-like entities and 8 near-duplicate pairs await review. Time spent: fill in. |
| 2026-11-01 | Phase 3: RQ1a risk themes (LDA vs NMF vs BERTopic, 26 configs x 5 seeds), topic-label template | | NMF k=12 selected by a pre-specified rule (robust to 4 alternative weightings); BERTopic did not beat it on coherence. Time spent: fill in. |
| 2026-11-01 | Phase 4: RQ1b trends (Mann-Kendall + Sen + BH, reporting-bias sensitivity, forecasting evaluation) | | PROVISIONAL until topic labels are confirmed. Freed 6.9 GB on C: (pip cache, with user permission) after the disk filled. Time spent: fill in. |
| 2026-11-01 | Phase 5: RQ2 network (typed graph, entity co-occurrence network, centrality, Louvain vs Leiden vs label propagation, temporal evolution, dashboard JSON) | | Leiden chosen by pre-specified rule (near tie with Louvain); communities are hub-dependent (ARI 0.35 without hubs). Time spent: fill in. |
| 2026-11-01 | Phase 6: RQ3 classification (5 models, nested CV, bootstrap CIs, paired tests, calibration), Colab DistilBERT notebook, blind audit sample + analysis code | | mpnet+LR best (macro-F1 0.651); temperature scaling selected; audit awaiting annotation, DistilBERT awaiting Colab run. Time spent: fill in. |
|  |  |  |  |
