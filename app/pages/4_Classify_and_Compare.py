"""AICIDENTIA Intelligence - Classify and Compare page (RQ3 and RQ4)."""

import streamlit as st

from aicidentia.config import load_config
from aicidentia.dashboard import charts, data, models, ui
from aicidentia.models import retrieval as R

ui.setup(
    "Classify and Compare",
    "RQ3: suggest a risk domain for a new incident description. RQ4: find similar past incidents. "
    "The classifier learns machine-generated MIT labels, so its output is a suggestion, not a judgement.",
)
f = data.fact
clf = ui.cached_classifier()
ret = ui.cached_retriever()

text = st.text_area(
    "Describe an incident (title and a sentence or two works best)",
    "A bank employee transferred money after a phone call in which an AI-cloned voice impersonated the company's CEO.",
    height=110,
)
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("Suggested risk domain")
    if clf is None:
        st.error("No classifier found: run `uv run python scripts/run_phase6_classification.py`.")
    else:
        st.caption(
            f"Model: **{clf.name}**{' (FALLBACK)' if clf.is_fallback else ''}, calibration: {clf.calibration or 'none'}"
        )
        if text.strip():
            pred = models.classify_text(clf, text)
            st.metric(
                pred["predicted_domain"].iloc[0],
                f"{100 * pred['confidence'].iloc[0]:.0f}% confidence",
            )
            st.plotly_chart(charts.probability_bars(pred), width="stretch")
            st.caption(
                "Confidence is calibrated on held-out data: of incidents shown with about 70% confidence, about 70% agreed with the MIT label."
            )

with c2:
    st.subheader("Similar past incidents")
    methods = models.usable_methods(ret)
    method = st.selectbox(
        "Method",
        methods,
        index=methods.index(ret.method),
        format_func=lambda m: R.LABELS[m] + (" (deployed)" if m == ret.deployed else ""),
    )
    if text.strip():
        hits = R.search(
            ret.index,
            method,
            text,
            k=int(load_config()["retrieval"]["top_k_dashboard"]),
            embed=models.query_embedder,
        )
        years = ui.cached_incidents().set_index("incident_id")["year"]
        hits["year"] = hits["incident_id"].map(years)
        st.dataframe(
            hits[["rank", "incident_id", "year", "title"]],
            hide_index=True,
            width="stretch",
            height=390,
        )
    st.caption(
        f"Evaluated against editor-marked similar incidents: MRR {(f('rq4.mrr', {}) or {}).get(method, float('nan')):.3f} for this method "
        f"(best {R.LABELS.get(f('rq4.best_method', ''), '?')}, {f('rq4.best_mrr', float('nan')):.3f})."
    )

st.divider()
st.subheader("How good is the classifier? (RQ3)")
metrics = ui.cached_table("rq3_metrics")
st.plotly_chart(charts.model_comparison(metrics), width="stretch")
st.markdown(
    f"Macro-F1 averages over the seven domains, so small domains count as much as large ones; a majority-class guess scores about 0.08. "
    f"Deployed: **{f('rq3.deployable', '?')}** ({f('rq3.calibration_method', '?')} calibration). "
    f"DistilBERT: {'scored' if f('rq3.colab_present') else 'pending the Colab run'}."
)
