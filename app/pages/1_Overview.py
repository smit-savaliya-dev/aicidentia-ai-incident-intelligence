"""AICIDENTIA Intelligence - Overview page."""

import streamlit as st

from aicidentia.dashboard import charts, data, ui

ui.setup(
    "Overview",
    "The AI Incident Database at a glance. Counts are of reported incidents, not of all AI harm.",
)

hide = ui.blind_ids()
inc = ui.cached_incidents(tuple(sorted(hide)))
left, right = st.columns([3, 2])
left.plotly_chart(charts.incidents_per_year(data.yearly_by_year()), width="stretch")
right.plotly_chart(charts.domain_bars(data.domain_counts()), width="stretch")
st.caption(
    f"{len(inc):,} incidents; {inc['reports'].sum():,} linked reports. The most recent year is incomplete. "
    "MIT risk labels are machine-generated and missing for some (especially recent) incidents."
)

st.subheader("Browse incidents")
c1, c2, c3 = st.columns([2, 2, 3])
query = c1.text_input("Title contains", "")
years = c2.slider(
    "Years", int(inc["year"].min()), int(inc["year"].max()), (2015, int(inc["year"].max()))
)
themes = c3.multiselect("Themes", sorted(inc["theme"].dropna().unique()))
view = inc[inc["year"].between(*years)]
if query:
    view = view[view["title"].str.contains(query, case=False, regex=False)]
if themes:
    view = view[view["theme"].isin(themes)]
st.write(f"{len(view):,} incidents match.")
st.dataframe(
    view.sort_values("date", ascending=False)[
        ["incident_id", "date", "title", "theme", "mit_domain", "reports"]
    ],
    width="stretch",
    hide_index=True,
    height=380,
    column_config={
        "date": st.column_config.DateColumn("date"),
        "mit_domain": "MIT domain (machine-generated)",
    },
)
pick = st.number_input("Show the description of incident #", min_value=0, value=0, step=1)
if pick:
    row = inc[inc["incident_id"] == pick]
    if row.empty:
        st.warning("No incident with that id in the analysis set.")
    else:
        r = row.iloc[0]
        st.markdown(
            f"**#{r['incident_id']} {r['title']}** ({r['date']:%Y-%m-%d}; theme: {r['theme']}; MIT domain: {r['mit_domain']})"
        )
        st.write(r["description"])
