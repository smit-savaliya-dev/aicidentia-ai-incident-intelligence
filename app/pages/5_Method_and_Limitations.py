"""AICIDENTIA Intelligence - Method and Limitations page."""

import importlib.metadata as md

import pandas as pd
import streamlit as st

from aicidentia.dashboard import data, status, ui
from aicidentia.models import retrieval as R

ui.setup(
    "Method and Limitations",
    "How the analysis was done, which models are active right now, and what to keep in mind.",
)
f = data.fact

st.subheader("Active models")
clf = ui.cached_classifier()
ret = ui.cached_retriever()
rows = [
    {
        "component": "Risk-domain classifier",
        "active": clf.name if clf else "none",
        "fallback": "yes" if clf and clf.is_fallback else "no",
        "why": clf.reason if clf else "no model files: run Phase 6",
        "calibration": (clf.calibration if clf else "") or "none",
    },
    {
        "component": "Similar-incident search",
        "active": R.LABELS[ret.method],
        "fallback": "yes" if ret.is_fallback else "no",
        "why": ret.reason + (f"; unavailable: {', '.join(ret.missing)}" if ret.missing else ""),
        "calibration": "",
    },
]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
if clf and clf.is_fallback:
    st.warning(
        "The embedding model is not in the local Hugging Face cache (HF_HOME), so the TF-IDF + logistic regression fallback is in use. "
        f"Its cross-validated macro-F1 is {f('rq3.fallback_macro_f1', float('nan')):.3f}."
    )
st.caption("Models are loaded from the local cache only; the dashboard never downloads anything.")

st.subheader("Method (CRISP-DM)")
st.markdown(
    f"""
- **Data:** frozen AIID snapshot {data.snapshot_info()["snapshot"]} (Excel export + MongoDB backup); {f("audit.n_incidents", 0):,} incidents after cleaning.
  Text = title + description. Stub reports removed, entities normalised, person-like entities kept out of the network.
- **RQ1 themes:** LDA vs NMF vs BERTopic on one shared corpus; NMF chosen by coherence, diversity and stability. **Trends:** Mann-Kendall + Sen's slope on yearly shares,
  Benjamini-Hochberg correction, sensitivity to counts / report weighting / start year; forecasting evaluated by rolling-origin MASE.
- **RQ2 network:** typed incident graph and weighted entity co-occurrence network; centrality (degree, betweenness, PageRank, eigenvector);
  Louvain vs Leiden vs label propagation over 10 seeds, chosen by a pre-specified rule.
- **RQ3 classification:** TF-IDF / embedding models with nested 5-fold CV, a time split, bootstrap CIs and paired tests; temperature-scaled confidences;
  blind human audit of 150 machine-generated labels (Cohen's kappa).
- **RQ4 retrieval:** TF-IDF, BM25, sentence embeddings and a hybrid against editor-marked similar incidents (MRR, nDCG@10, P@5).
- Every number is produced by code in `src/aicidentia` and stored in `reports/results.json`; random seed {data.snapshot_info()["seed"]}.
"""
)

st.subheader("Your manual inputs")
st.dataframe(status.manual_status(), hide_index=True, width="stretch")
st.code("uv run python scripts/refresh_all.py", language="bash")
st.caption(
    "Re-runs every phase that depends on these files, then press 'Reload data' in the sidebar."
)

st.subheader("Limitations")
st.markdown(
    """
- **Reported, not all, incidents.** AIID reflects what journalists covered and editors accepted; trends describe the database, not the prevalence of AI harm.
- **Machine-generated labels.** MIT risk domains were assigned automatically; the classifier learns to reproduce them. The blind audit measures how far a human agrees.
- **Short series.** 11 complete years give trend tests little power; a 'no trend' result is not evidence of stability. Forecasts are not shown because they did not beat a naive forecast.
- **Alleged involvement.** Organisations are alleged developers/deployers; network centrality reflects frequency in reports, not responsibility. Communities depend on a few generic hubs.
- **Sparse similarity judgements.** Editors mark few similar pairs, so retrieval scores are lower bounds.
- **Themes** are dominant-topic assignments from a model fitted mostly on recent incidents; names are provisional until confirmed.
"""
)

st.subheader("Software")
versions = {
    p: md.version(p)
    for p in ["pandas", "scikit-learn", "sentence-transformers", "networkx", "streamlit", "plotly"]
}
st.write(", ".join(f"{k} {v}" for k, v in versions.items()))
