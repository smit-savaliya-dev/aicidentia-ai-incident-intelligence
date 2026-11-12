"""AICIDENTIA Intelligence - Themes and Trends page (RQ1)."""

import streamlit as st

from aicidentia.dashboard import charts, data, ui

ui.setup(
    "Themes and Trends",
    "RQ1: risk themes found by topic modelling (NMF) and whether their share of incidents rises or falls over time.",
)
f = data.fact
info = ui.cached_theme_info()
if info.get("provisional"):
    st.warning(
        "Theme names are automatic placeholders (`T<id>: top words`) until you confirm them in "
        "`data/manual/topic_labels.csv`. The statistics do not depend on the names."
    )

yearly = ui.cached_table("rq1b_yearly_themes")
trends = ui.cached_table("rq1b_trends_themes")
share = trends[trends["basis"] == "share"].sort_values("sen_slope", ascending=False)
window = f("rq1b.window", [2015, 2025])
st.markdown(
    f"**Window:** complete years {window[0]}-{window[1]} (Mann-Kendall test, Sen's slope, Benjamini-Hochberg q = 0.05). "
    f"**Emerging** = significantly increasing share and the last two years above the long-run mean. "
    f"The partial year {f('rq1b.partial_year', '')} is shown hollow and never tested."
)

st.plotly_chart(charts.theme_heatmap(yearly, share["group"].tolist()), width="stretch")

c1, c2 = st.columns([3, 1])
default = f("rq1b.emerging_themes", []) or share["group"].head(3).tolist()
chosen = c1.multiselect("Themes to plot", share["group"].tolist(), default=default)
measure = c2.radio("Measure", ["share", "count"], horizontal=True)
if chosen:
    st.plotly_chart(charts.theme_lines(yearly, chosen, measure), width="stretch")
if measure == "count":
    st.info(
        "Raw counts mostly track the growth of the database itself; the share is the primary measure."
    )

st.subheader("Trend tests (yearly share)")
view = share.assign(
    sen_slope_pp=(share["sen_slope"] * 100).round(2),
    recent_pct=(share["recent_mean"] * 100).round(1),
    longrun_pct=(share["longrun_mean"] * 100).round(1),
)[["group", "direction", "emerging", "sen_slope_pp", "p", "p_bh", "longrun_pct", "recent_pct"]]
st.dataframe(
    view,
    hide_index=True,
    width="stretch",
    column_config={
        "group": "theme",
        "sen_slope_pp": "Sen slope (pp/yr)",
        "p_bh": "p (BH)",
        "longrun_pct": "long-run share %",
        "recent_pct": "last 2 years %",
    },
)

st.subheader("Does the conclusion depend on how we measure?")
cls = ui.cached_table("rq1b_classification_themes")
st.dataframe(cls, hide_index=True, width="stretch")
st.caption(
    f"{f('rq1b.n_themes_disagree', '?')} themes change direction between counts, shares and report-weighted shares. "
    "Report-weighted shares understate recent themes (recent incidents have had less time to attract reports)."
)

with st.expander("MIT risk domains and forecasting"):
    st.dataframe(ui.cached_table("rq1b_classification_domains"), hide_index=True, width="stretch")
    st.markdown(
        f"**Forecasting** quarterly counts (naive vs exponential smoothing vs ARIMA, rolling origin): "
        f"best model {f('rq1b.forecast_best_model', '?')} MASE {f('rq1b.forecast_best_mase', float('nan')):.2f} vs naive "
        f"{f('rq1b.forecast_naive_mase', float('nan')):.2f}: forecasting is "
        f"**{'justified' if f('rq1b.forecast_justified') else 'not justified'}**, so no forecasts are shown."
    )

st.subheader("Incidents in a theme")
theme = st.selectbox("Theme", share["group"].tolist())
inc = ui.cached_incidents()
st.dataframe(
    inc[inc["theme"] == theme].sort_values("date", ascending=False)[
        ["incident_id", "date", "title"]
    ],
    hide_index=True,
    width="stretch",
    height=300,
)
