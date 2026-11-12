"""AICIDENTIA Intelligence - Method and Limitations."""

import streamlit as st

from aicidentia.dashboard import charts, data, models, status, ui
from aicidentia.models import retrieval as R

ui.setup(
    "Method and Limitations",
    "How the numbers were produced, which models are running now, and what the data cannot tell you.",
    filters=False,
)
f = data.fact
info = data.snapshot_info()

st.markdown("### Pipeline (CRISP-DM)")
st.markdown(
    f"""
**Data:** AI Incident Database snapshot **{info["snapshot_iso"]}** (Excel export + full database dump): {f("audit.n_incidents", 0):,} incidents,
{f("audit.n_reports", 0):,} reports. **Preparation:** AIID duplicates merged, rule-based duplicate review (D2), every named person excluded (D3),
entity aliases and types. **RQ1** themes: LDA vs NMF vs BERTopic, NMF k=12 chosen by a fixed rule; trends: Mann-Kendall + Sen's slope with
Benjamini-Hochberg. **RQ2** network: entity co-occurrence, centrality, Louvain vs Leiden vs label propagation (Leiden chosen). **RQ3**
classification: TF-IDF, sentence embeddings and DistilBERT with nested 5-fold CV and a time split; label reliability with three independent checks.
**RQ4** retrieval: TF-IDF, BM25, embeddings and a hybrid, scored against editor-marked similar incidents. Every number comes from
`reports/results.json`; every rule is in `docs/decisions.md`.
"""
)

st.markdown("### Active models (offline-first, with fallbacks)")
active = models.expected_status()
st.dataframe(active, hide_index=True, width="stretch")
st.caption(
    f"Classifier: {f('rq3.deployable_reason', '')}. Search: {f('rq4.deployed_reason', '')}. "
    "Models load only from the local cache; nothing is downloaded at run time."
)

st.markdown("### Model comparison")
a, b = st.columns(2)
with a:
    st.markdown("**RQ3 classifiers: macro-F1 with 95% CI**")
    metrics = ui.cached_table("rq3_metrics")
    st.plotly_chart(charts.model_comparison(metrics), width="stretch")
    show = metrics[~metrics["model"].str.contains("ablation")][
        ["model", "setting", "n", "macro_f1", "ci_low", "ci_high", "accuracy"]
    ]
    st.dataframe(show.round(3), hide_index=True, width="stretch")
    ui.download(show, "rq3_model_comparison")
with b:
    st.markdown("**RQ4 search methods: mean reciprocal rank with 95% CI**")
    summary = ui.cached_table("rq4_summary_all")
    st.plotly_chart(charts.retrieval_comparison(summary, R.LABELS), width="stretch")
    show4 = summary[["method", "queries", "rr", "rr_low", "rr_high", "ndcg@10", "p@5", "hit@5"]]
    st.dataframe(show4.round(3), hide_index=True, width="stretch")
    st.caption(
        f"MRR@3 against AIID's own suggestions: ours {f('rq4.mrr3', {}).get(f('rq4.deployed', 'bm25'), float('nan')):.3f}, AIID {f('rq4.aiid_nlp_rr3', float('nan')):.3f}."
    )
    ui.download(show4, "rq4_method_comparison")

st.markdown("### Ethics")
st.markdown(
    f"""
- **Alleged, not proven.** Organisations appear as *alleged* developers or deployers, as AIID records them. Network centrality reflects how often
  an organisation is reported, not responsibility or blame.
- **No named persons.** All {f("rules.entities.named_persons_excluded", 0):,} person names (private or public) found among the entities are excluded
  from the network, the search lists and every table built from entities (rule D3). Incident titles are AIID's own text and link back to AIID.
- **Reporting bias.** AIID holds incidents that journalists covered and editors accepted; English-language, US and high-profile cases are
  over-represented. Trends describe the database, not the real-world frequency of AI harm.
- **Machine-generated labels.** MIT risk domains were assigned automatically; about {ui.fmt_pct(f("rq3rel.cl_noise_rate", 0))} are estimated to be wrong.
"""
)

st.markdown("### Limitations")
st.markdown(
    """
- **Short series:** eleven complete years (2015-2025) give trend tests little power; forecasts are not shown because they did not beat a naive forecast.
- **Severity data stop in 2023:** CSETv1 annotation covers 207 incidents, so the risk priority matrix can place only 2 of 12 themes.
- **Sparse similarity links:** editors link only some incidents, so retrieval scores and the share of repeated failures are lower bounds.
- **Hub-dependent communities:** network communities change when a few generic hub entities are removed.
- **Before/after ChatGPT** is a shift in reported incidents, not a causal effect.
"""
)

with st.expander("Manual inputs and the refresh command"):
    st.dataframe(status.manual_status(), hide_index=True, width="stretch")
    st.code("uv run python scripts/refresh_all.py", language="bash")

st.markdown("### Data and citations")
st.markdown(
    f"""
- McGregor, S. (2021). *Preventing repeated real world AI failures by cataloging incidents: The AI Incident Database.* AAAI 35(17), 15458-15463.
  Data: AI Incident Database, snapshot {info["snapshot_iso"]}, https://incidentdatabase.ai.
- Slattery, P. et al. (2024). *The AI Risk Repository: A comprehensive meta-review, database, and taxonomy of risks from artificial intelligence.* arXiv:2408.12622 (MIT risk domains).
- CSET AI Harm Taxonomy for AIID (CSETv1), Center for Security and Emerging Technology.
- Goals, Methods and Failures taxonomy (GMF), Pittaras, N. & McGregor, S. (2022), arXiv:2211.07280.
- Northcutt, C., Jiang, L. & Chuang, I. (2021). *Confident learning: Estimating uncertainty in dataset labels.* JAIR 70, 1373-1411.
"""
)
ui.footer()
