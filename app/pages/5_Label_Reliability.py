"""AICIDENTIA Intelligence - Label Reliability (RQ3): can the AI-generated MIT labels be trusted?"""

import streamlit as st

from aicidentia.dashboard import charts, data, ui

ui.setup(
    "Label Reliability",
    "AIID's risk-domain labels (MIT taxonomy) were assigned by an AI, not by people. Three independent checks, "
    "using evidence already in AIID, show where they can be trusted and where they cannot.",
)
f = data.fact
sim = f("rq3rel.pairs_similar", {}) or {}
dis = f("rq3rel.pairs_dissimilar", {}) or {}
kap = f("rq3rel.mapping_kappa", {}) or {}

k = st.columns(4)
ui.kpi(
    k[0],
    "Agreement with humans",
    f"κ {kap.get('discrimination_vs_protected_characteristic', float('nan')):.2f}",
    "discrimination vs CSETv1 (substantial)",
)
ui.kpi(
    k[1],
    "Editor-linked pairs",
    ui.fmt_pct(sim.get("same_domain_rate", float("nan"))),
    f"share a domain; {ui.fmt_pct(sim.get('baseline_rate', float('nan')))} by chance",
)
ui.kpi(
    k[2],
    "Estimated wrong labels",
    ui.fmt_pct(f("rq3rel.cl_noise_rate", float("nan"))),
    f"{int(f('rq3rel.cl_flagged', 0)):,} incidents flagged",
)
ui.kpi(
    k[3],
    "Retrain without them",
    f"{f('rq3rel.retrain_diff', 0):+.3f}",
    "macro-F1 change (not significant)",
)
st.caption(
    "These checks use all MIT-labelled incidents; the sidebar filters do not apply on this page."
)

st.markdown("### 1. Do the labels agree with people? (human CSETv1 and GMF annotations)")
a, b = st.columns([3, 2])
mappings = ui.cached_table("rq3rel_mappings")
a.plotly_chart(charts.kappa_forest(mappings), width="stretch")
b.markdown(
    f"""
People at CSET annotated {f("rq3rel.cset_mit_overlap", 0)} of the MIT-labelled incidents (all before 2024). None of their
fields is a risk domain, so each MIT domain is compared with the closest human field.

- **Discrimination**: substantial agreement with "protected characteristic" and "harm distribution".
- **AI-system safety**: moderate agreement with "tangible harm"; the concepts only overlap.
- **Misinformation / misuse**: fair agreement with "detrimental content".
- **GMF technical failure**: only {f("rq3rel.gmf_mit_overlap", 0)} incidents, too few to judge.

These are partial checks, not ground truth: low agreement can come from different definitions.
"""
)

st.markdown("### 2. Do incidents that editors link together get the same label?")
a, b = st.columns([3, 2])
a.plotly_chart(charts.pair_bars(sim, dis), width="stretch")
b.markdown(
    f"""
AIID editors marked **{sim.get("pairs_total", 0)}** pairs of incidents as similar. In **{ui.fmt_pct(sim.get("same_domain_rate", 0))}**
of the {sim.get("pairs_labelled", 0)} labelled pairs both incidents carry the same MIT domain, against
{ui.fmt_pct(sim.get("baseline_rate", 0))} for random pairs from the same years ({sim.get("lift", 0):.1f}× chance, permutation p = {sim.get("p_value", 1):.4f}).
Pairs editors marked as *dissimilar* match only {ui.fmt_pct(dis.get("same_domain_rate", 0))} of the time, at chance level.

**Plain language:** the AI labeller groups incidents the way human editors do.
"""
)

st.markdown("### 3. How many labels are probably wrong? (confident learning)")
a, b = st.columns([3, 2])
noise = ui.cached_table("rq3rel_noise_by_domain").set_index("domain")["estimated_noise_rate"]
a.plotly_chart(
    charts.horizontal_bars(
        noise,
        "",
        "Estimated share of labels that are wrong (%)",
        charts.PALETTE["vermillion"],
        pct=True,
    ),
    width="stretch",
)
b.markdown(
    f"""
A classifier trained on the text predicts each incident's domain without having seen its label. Where it
confidently disagrees with the given label, the label is suspect (Northcutt et al., 2021).

- **{ui.fmt_pct(f("rq3rel.cl_noise_rate", 0))}** of labels are estimated to be wrong overall.
- The small domains (human-computer interaction, socioeconomic harms) are the least reliable.
- Retraining without the {int(f("rq3rel.cl_flagged", 0))} flagged incidents changes macro-F1 by {f("rq3rel.retrain_diff", 0):+.3f}
  (95% CI {f("rq3rel.retrain_ci", [0, 0])[0]:+.3f} to {f("rq3rel.retrain_ci", [0, 0])[1]:+.3f}): no measurable effect.
"""
)
a, b = st.columns([3, 2])
joint = ui.cached_table("rq3rel_estimated_joint")
joint = joint.set_index(joint.columns[0])
a.plotly_chart(charts.label_confusion(joint), width="stretch")
b.markdown(
    """
**Which labels get mixed up?** Most suspected errors are *Malicious actors & misuse* incidents whose text reads as
*Misinformation*. Under MIT's own definitions an AI deepfake used for propaganda or a scam fits both, so this is
mostly taxonomy overlap, not careless labelling. Exchanges between distant domains (AI-system safety vs
discrimination) are more likely genuine errors.
"""
)
with st.expander("Most suspicious labels (top 25)"):
    flagged = ui.cached_table("rq3rel_flagged_incidents").head(25)
    titles = ui.cached_incidents().set_index("incident_id")["title"]
    flagged.insert(1, "title", flagged["incident_id"].map(titles))
    flagged["url"] = flagged["incident_id"].map(data.cite_url)
    ui.incident_table(flagged, height=420)
    ui.download(
        ui.cached_table("rq3rel_flagged_incidents"),
        "flagged_labels",
        "Download all flagged incidents (CSV)",
    )
ui.footer()
