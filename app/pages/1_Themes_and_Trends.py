"""AICIDENTIA Intelligence - Themes and Trends (RQ1)."""

import streamlit as st

from aicidentia.dashboard import charts, data, ui

flt = ui.setup(
    "Themes and Trends",
    "RQ1: twelve risk themes found by topic modelling (NMF), how their share of incidents changes, "
    "and what shifted after ChatGPT's release on 30 Nov 2022.",
)
f = data.fact
inc = ui.cached_incidents()
sub = data.filter_incidents(inc, flt)
ui.filter_note(flt, len(sub), len(inc))

em = data.emerging_table()
k = st.columns(4)
ui.kpi(k[0], "Incidents shown", f"{len(sub):,}", flt.describe())
ui.kpi(k[1], "Themes", f"{f('rq1b.n_themes', 12)}", "NMF k=12, chosen by a fixed rule")
ui.kpi(k[2], "Emerging", f"{int((em['status'] == 'EMERGING').sum())}", "rising share, BH q < 0.05")
ct = f("story.chatgpt_themes", {}) or {}
ui.kpi(
    k[3],
    "Shift after ChatGPT",
    f"V = {ct.get('cramers_v', float('nan')):.2f}",
    "Cramer's V, themes (large)",
)

st.markdown("### Which themes rise and fall")
left, right = st.columns([3, 2])
with left:
    shares = data.theme_year_shares(sub)
    order = list(em["theme"])
    st.plotly_chart(
        charts.theme_heatmap(shares, [t for t in order if t in set(shares["group"])]),
        width="stretch",
    )
    st.caption(
        "Share of each year's incidents (filtered) per theme; darker = larger share. Complete years only; "
        "the partial year is left out. Themes ordered by trend, fastest-rising first."
    )
with right:
    show = em.copy()
    show["status"] = show["status"].replace(
        {"EMERGING": "🔺 EMERGING", "declining": "🔻 declining"}
    )
    st.dataframe(
        show[["theme", "status", "slope_pp_per_year", "q_value", "recent_share"]],
        hide_index=True,
        width="stretch",
        height=470,
        column_config={
            "theme": st.column_config.TextColumn("Theme", width="large"),
            "status": "Status",
            "slope_pp_per_year": st.column_config.NumberColumn("Trend (pp/yr)", format="%+.2f"),
            "q_value": st.column_config.NumberColumn("q-value", format="%.3f"),
            "recent_share": st.column_config.NumberColumn("Recent share (%)", format="%.1f"),
        },
    )
    st.caption(
        "Mann-Kendall test on yearly shares 2015-2025, Sen's slope in percentage points per year, "
        "Benjamini-Hochberg q-values. Emerging = significant rise and recent share above the long-run mean. "
        "Tests use all incidents; the filters do not change them."
    )
    ui.download(show, "emerging_themes")

st.markdown("### What changed after ChatGPT (30 Nov 2022)")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(
        charts.chatgpt_slope(
            ui.cached_table("story_chatgpt_domains"), "MIT risk domains (labelled incidents)"
        ),
        width="stretch",
    )
with c2:
    st.plotly_chart(
        charts.chatgpt_slope(ui.cached_table("story_chatgpt_themes"), "Themes (all incidents)"),
        width="stretch",
    )
cd = f("story.chatgpt_domains", {}) or {}
st.caption(
    f"Incidents from 2015; blue = significantly more common after the release, red = less common, grey = no "
    f"significant change (adjusted residual beyond ±1.96). Domains: chi-square p < 0.001, Cramer's V {cd.get('cramers_v', 0):.2f}, "
    f"n = {cd.get('n', 0):,}; MIT labels cover {100 * cd.get('mit_coverage_before', 0):.0f}% of incidents before and "
    f"{100 * cd.get('mit_coverage_after', 0):.0f}% after. A shift in reported incidents, not a causal effect. Filters do not apply here."
)

st.markdown("### Representative incidents")
themes = [t for t in em["theme"] if t in set(sub["theme"])]
if themes:
    pick = st.selectbox("Theme", themes, index=0)
    reps = data.representative_incidents(sub, pick, 6)
    ui.incident_table(reps[["incident_id", "date", "title", "mit_domain", "url"]], height=260)
    st.caption(
        "Most representative incidents of the theme (highest topic weight) within the filters; links open the AIID record."
    )
    theme_rows = sub[sub["theme"] == pick].sort_values("date", ascending=False)
    ui.download(
        theme_rows.drop(columns=["sectors", "failures"]),
        "theme_incidents",
        f"Download all {len(theme_rows):,} incidents of this theme (CSV)",
    )
else:
    st.warning("No incidents match the filters.")
ui.footer()
