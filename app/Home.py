"""AICIDENTIA Intelligence - Home: the five key findings and headline numbers."""

import streamlit as st

from aicidentia.dashboard import data, ui

PAGES = {
    "Themes and Trends": "pages/1_Themes_and_Trends.py",
    "Risk Priority Matrix": "pages/2_Risk_Priority_Matrix.py",
    "Risk Network": "pages/3_Risk_Network.py",
    "Precedent Finder": "pages/4_Precedent_Finder.py",
    "Label Reliability": "pages/5_Label_Reliability.py",
    "Method and Limitations": "pages/6_Method_and_Limitations.py",
}

ui.setup(
    "Key Findings",
    "What 1,775 reported AI incidents tell us: which harms are growing, who is involved, "
    "how far the AI-generated labels can be trusted, and whether past failures can be found again.",
    filters=False,
)
f = data.fact
inc = ui.cached_incidents()
rep = f("rq4.repeat", {}) or {}
dep = f("rq4.deployed", "bm25")
found = (rep.get("found_in_top5") or {}).get(dep, {})

k = st.columns(5)
ui.kpi(
    k[0],
    "Reported incidents",
    f"{len(inc):,}",
    f"AIID snapshot {data.snapshot_info()['snapshot_iso']}",
)
ui.kpi(
    k[1],
    "Since 2024",
    ui.fmt_pct((inc["year"] >= 2024).mean()),
    f"{int((inc['year'] >= 2024).sum()):,} incidents in under three years",
)
ui.kpi(
    k[2],
    "Emerging themes",
    f"{len(f('rq1b.emerging_themes', []))} of {f('rq1b.n_themes', 12)}",
    "significant rise in share",
)
ui.kpi(
    k[3],
    "Precedent found",
    ui.fmt_pct(found.get("hit@5", float("nan"))),
    "top-5 search hit for repeats since 2024",
)
ui.kpi(
    k[4],
    "Doubtful MIT labels",
    ui.fmt_pct(f("rq3rel.cl_noise_rate", float("nan"))),
    "estimated by confident learning",
)

st.markdown("### The five findings")
findings = f("story.key_findings", []) or []
rows = [findings[:3], findings[3:]]
for row in rows:
    cols = st.columns(3)
    for col, finding in zip(cols, row, strict=False):
        ui.finding_card(col, finding, PAGES.get(finding["page"]))
    st.write("")

st.info(
    "**How to use:** set the year range, risk domain or sector in the sidebar (they apply on every page), "
    "open a finding's page for the evidence, and try the **Precedent Finder** with your own incident or the demo cases.",
    icon=":material/lightbulb:",
)
ui.footer()
