"""AICIDENTIA Intelligence - Risk Priority Matrix (RQ1 trend x CSETv1 severity)."""

import streamlit as st

from aicidentia.dashboard import charts, data, ui

flt = ui.setup(
    "Risk Priority Matrix",
    "Which themes are growing, and which cause tangible harm? Trend from RQ1, severity from human CSETv1 "
    "annotation, bubble size = incidents. Click a bubble to list its incidents.",
)
f = data.fact
inc = ui.cached_incidents()
sub = data.filter_incidents(inc, flt)
ui.filter_note(flt, len(sub), len(inc))
matrix = ui.cached_table("story_risk_matrix")
threshold = float(f("story.matrix_severity_overall", 0.35))
supported = int(matrix["supported"].sum())

k = st.columns(4)
ui.kpi(
    k[0], "Themes placed", f"{supported} of {len(matrix)}", "need >= 30 CSET-annotated incidents"
)
ui.kpi(
    k[1],
    "CSET-annotated incidents",
    f"{int(f('story.matrix_cset_incidents', 0)):,}",
    f"all from {f('story.matrix_cset_year_max', 2023)} or earlier",
)
ui.kpi(k[2], "Severity line", ui.fmt_pct(threshold), "share severe among all CSET-annotated")
ui.kpi(
    k[3],
    "Fastest-rising theme",
    f"{matrix.iloc[0]['trend_pp_per_year']:+.2f} pp/yr",
    matrix.iloc[0]["theme"],
)

st.warning(
    f"**Read with care:** human severity annotation (CSETv1) stops in {f('story.matrix_cset_year_max', 2023)}, so only "
    f"{supported} themes have enough annotated incidents (n >= 30) to be placed. The emerging deepfake, scam and election "
    "themes have 3-13 annotations each and are shown in grey with their n: the data cannot yet rank them by severity.",
    icon=":material/warning:",
)
event = st.plotly_chart(
    charts.risk_matrix(matrix, threshold),
    width="stretch",
    on_select="rerun",
    selection_mode="points",
    key="matrix",
)
st.caption(
    "x: Sen's slope of the theme's yearly share 2015-2025 (percentage points per year). y: share of the theme's "
    "CSETv1-annotated incidents where tangible harm definitively occurred, or with injuries or deaths. Dashed lines: "
    "zero trend and the severity of all CSET-annotated incidents. Bubble area = incidents."
)

points = (event.selection or {}).get("points", []) if event else []
theme = points[0]["customdata"][0] if points else None
options = list(matrix["theme"])
theme = st.selectbox(
    "Incidents of theme (or click a bubble)",
    options,
    index=options.index(theme) if theme in options else 0,
)
row = matrix.set_index("theme").loc[theme]
severity = "n/a" if row["cset_n"] == 0 else f"{100 * row['severity']:.0f}%"
st.markdown(
    f"**{theme}** · {int(row['incidents'])} incidents · trend {row['trend_pp_per_year']:+.2f} pp/yr (q = {row['q_value']:.3f}) · "
    f"severity {severity} of {int(row['cset_n'])} CSET-annotated · **{row['quadrant']}**"
)
rows = sub[sub["theme"] == theme].sort_values("date", ascending=False)
ui.incident_table(
    rows[["incident_id", "date", "title", "mit_domain", "cset_tangible_harm", "url"]], height=330
)
ui.download(rows.drop(columns=["sectors", "failures"]), "risk_matrix_theme")
st.caption("Incidents within the sidebar filters. The matrix itself uses all incidents.")
ui.footer()
