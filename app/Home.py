"""AICIDENTIA Intelligence - Streamlit entry point.

Run with:  uv run streamlit run app/Home.py
Everything shown is read from the saved pipeline outputs; refresh them with
`uv run python scripts/refresh_all.py`, then press "Reload data" in the sidebar.
"""

import streamlit as st

from aicidentia.dashboard import data, ui

ui.setup("Home", data.snapshot_info()["title"])
info = data.snapshot_info()

st.subheader(info["tagline"])

f = data.fact
c1, c2, c3, c4 = st.columns(4)
c1.metric("Incidents analysed", f"{f('audit.n_incidents', 0):,}")
c2.metric("Complete years", f"{f('rq1b.window', ['?', '?'])[0]}-{f('rq1b.window', ['?', '?'])[1]}")
c3.metric("Emerging themes", len(f("rq1b.emerging_themes", []) or []))
c4.metric("Network communities", f("rq2.n_communities", "-"))
c5, c6, c7, c8 = st.columns(4)
cv_scores = [m["macro_f1"] for m in (f("rq3.metrics", []) or []) if m.get("setting") == "cv"]
c5.metric("Best classifier macro-F1 (CV)", f"{max(cv_scores):.3f}" if cv_scores else "-")
c6.metric(
    "Deployed classifier",
    "mpnet + LR" if "mpnet" in str(f("rq3.deployable", "")) else str(f("rq3.deployable", "-")),
)
c7.metric(
    "Retrieval MRR (deployed)",
    f"{(f('rq4.mrr', {}) or {}).get(f('rq4.deployed', 'bm25'), float('nan')):.3f}",
)
c8.metric("Forecasting justified", "yes" if f("rq1b.forecast_justified") else "no")

st.markdown(
    """
**Pages**

1. **Overview** - the AI Incident Database at a glance, and an incident browser
2. **Themes and Trends** (RQ1) - risk themes from topic modelling and how their share changes over time
3. **Risk Network** (RQ2) - organisations, AI systems and harmed groups linked by shared incidents
4. **Classify and Compare** (RQ3, RQ4) - classify a new incident description and find similar past incidents
5. **Method and Limitations** - how the analysis was done, which models are active, and what to keep in mind

**How to read this dashboard.** AIID records *reported* incidents. Counts reflect media attention and
editorial effort as well as harm; organisations are *alleged* developers and deployers; the MIT risk
labels are machine-generated. Each page states the main caveat for what it shows.
"""
)
