"""AICIDENTIA Intelligence - Precedent Finder (RQ3 classifier + RQ4 retrieval)."""

import numpy as np
import pandas as pd
import streamlit as st

from aicidentia.dashboard import charts, data, models, ui
from aicidentia.models import retrieval as R

flt = ui.setup(
    "Precedent Finder",
    "Paste a new incident: get its likely risk domain with calibrated confidence and the five most similar "
    "EARLIER incidents, with what went wrong in them. Models run locally; nothing is sent anywhere.",
)
f = data.fact
inc = ui.cached_incidents()
ss = st.session_state
ss.setdefault("pf_text", "")
ss.setdefault("pf_before", None)
ss.setdefault("pf_exclude", None)

demo = data.demo_cases(inc)
st.markdown("**Demo mode:** three real incidents from the snapshot's most recent months.")
cols = st.columns(len(demo) or 1)
for col, r in zip(cols, demo.itertuples(), strict=False):
    if col.button(
        f"{pd.Timestamp(r.date):%d %b %Y} · {r.title[:70]}…",
        key=f"demo_{r.incident_id}",
        width="stretch",
    ):
        ss.pf_text = f"{r.title}. {r.description}"
        ss.pf_before = pd.Timestamp(r.date)
        ss.pf_exclude = int(r.incident_id)

text = st.text_area(
    "Incident description (title and a sentence or two works best)",
    value=ss.pf_text,
    height=110,
    placeholder="e.g. A bank employee transferred money after a call in which an AI-cloned voice impersonated the CEO.",
)
if text != ss.pf_text:  # typed by hand: search everything before the snapshot date
    ss.pf_text, ss.pf_before, ss.pf_exclude = text, None, None
st.button("Find precedents", type="primary", icon=":material/search:")

if not ss.pf_text.strip():
    st.caption("Type or paste an incident, or press a demo case.")
    ui.footer()
    st.stop()

clf = ui.cached_classifier()
ret = ui.cached_retriever()
left, right = st.columns([2, 3])
with left:
    st.markdown("### Likely risk domain")
    if clf is None:
        st.error("No classifier found: run `uv run python scripts/run_phase6_classification.py`.")
    else:
        pred = models.classify_text(clf, ss.pf_text)
        probs = pred.filter(like="p_").iloc[0].sort_values(ascending=False)
        top, conf = probs.index[0][2:], probs.iloc[0]
        ui.kpi(
            st,
            "Predicted MIT risk domain",
            charts.short(top),
            f"{100 * conf:.0f}% calibrated confidence",
        )
        st.write("")
        alt = pd.DataFrame(
            {
                "alternative": [charts.short(c[2:]) for c in probs.index[1:4]],
                "probability": probs.values[1:4],
            }
        )
        st.dataframe(
            alt,
            hide_index=True,
            width="stretch",
            column_config={
                "probability": st.column_config.ProgressColumn(
                    "probability", format="percent", min_value=0, max_value=1
                )
            },
        )
        st.caption(
            f"Top 3 alternatives. Model: {clf.name}{' (fallback)' if clf.is_fallback else ''}, temperature-scaled: of incidents "
            "shown with about 70% confidence, about 70% matched the MIT label in cross-validation. MIT labels are machine-generated."
        )
with right:
    before = (
        ss.pf_before
        if ss.pf_before is not None
        else pd.Timestamp(data.snapshot_info()["snapshot_iso"])
    )
    st.markdown(f"### 5 most similar earlier incidents (before {before:%d %b %Y})")
    hits = R.search(
        ret.index,
        ret.method,
        ss.pf_text,
        k=60,
        embed=models.query_embedder,
        exclude_ids=[ss.pf_exclude] if ss.pf_exclude else [],
        before=np.datetime64(before),
    )
    details = data.precedent_details(hits, inc)
    allowed = set(data.filter_incidents(inc, flt)["incident_id"])
    details = details[details["incident_id"].isin(allowed)].head(5).reset_index(drop=True)
    details["rank"] = range(1, len(details) + 1)
    ui.incident_table(
        details[["rank", "incident_id", "date", "title", "mit_domain", "url"]], height=230
    )
    st.caption(
        f"Search: {R.LABELS[ret.method]}{' (fallback)' if ret.is_fallback else ' (deployed)'}, earlier incidents only, within the sidebar filters. "
        f"Against editor-marked links it finds a precedent in the top 5 for "
        f"{100 * (f('rq4.repeat', {}) or {}).get('found_in_top5', {}).get(ret.method, {}).get('hit@5', float('nan')):.0f}% of 2024+ incidents that have one."
    )
    if ss.pf_exclude:
        st.caption(f"Demo case #{ss.pf_exclude}: the incident itself is excluded from the results.")

st.markdown("### What went wrong before")
for r in details.itertuples():
    with st.container(border=True):
        a, b = st.columns([4, 1])
        a.markdown(
            f"**{r.rank}. {r.title}**  \n{pd.Timestamp(r.date):%d %b %Y} · {r.mit_domain}{' · ' + r.mit_subdomain if r.mit_subdomain else ''}"
        )
        b.link_button("Open in AIID", r.url, icon=":material/open_in_new:")
        st.write(r.description)
        if len(r.failures):
            st.caption("Technical failure (GMF, human-annotated): " + ", ".join(r.failures))
ui.download(details.drop(columns=["failures"]), "precedents")
ui.footer()
