"""AICIDENTIA Intelligence - Risk Network page (RQ2)."""

import pandas as pd
import streamlit as st

from aicidentia.dashboard import charts, data, ui

ui.setup(
    "Risk Network",
    "RQ2: organisations, AI systems and harmed groups linked when they appear in the same incidents. "
    "Organisations are ALLEGED developers or deployers; centrality means frequent involvement in reported incidents, not blame.",
)
f = data.fact
net = ui.cached_network()
years = sorted({n["first_year"] for n in net["nodes"]})
types = sorted({n["type"] for n in net["nodes"]})

c1, c2 = st.columns([2, 3])
year = c1.slider("Show the network up to the end of", min(years), max(years), max(years))
chosen_types = c2.multiselect(
    "Node types", types, default=types, format_func=lambda t: charts.TYPE_NAME.get(t, t)
)
nodes = sorted(
    (n for n in net["nodes"] if n["first_year"] <= year and n["type"] in chosen_types),
    key=lambda n: -n["weighted_degree"],
)
labels = {n["id"]: f"{n['label']} ({charts.TYPE_NAME.get(n['type'], n['type'])})" for n in nodes}
pick = st.selectbox(
    "Select an entity to inspect",
    [None, *labels],
    format_func=lambda k: "(none)" if k is None else labels[k],
)

left, right = st.columns([3, 1])
left.plotly_chart(charts.network_figure(net, year, chosen_types, pick), width="stretch")
with right:
    st.markdown("**Selected entity**")
    if pick is None:
        st.write("Choose an entity above to see its profile.")
    else:
        node = next(n for n in net["nodes"] if n["id"] == pick)
        prof = data.node_profile(node)
        st.markdown(
            f"**{prof['label']}**  \n{charts.TYPE_NAME.get(prof['type'], prof['type'])}, community {prof['community']}  \n"
            f"{prof['incidents']} incidents, {prof['years']}  \nPageRank {prof['pagerank']:.4f}"
        )
        if prof["domains"]:
            st.dataframe(
                pd.Series(prof["domains"], name="incidents").rename_axis("MIT domain"),
                width="stretch",
            )
        st.markdown("**Recent incidents**")
        for r in prof["recent"]:
            st.markdown(f"- #{r['id']} ({r['year']}) {r['title']}")

st.caption(
    f"{net['caps']['top_n']} most connected entities and at most {net['caps']['max_edges']} strongest links are shown "
    f"(the full network has {f('rq2.entity_network_nodes', '?'):,} entities). Communities: {f('rq2.chosen_method', '?')}, "
    f"modularity {f('rq2.chosen_modularity', float('nan')):.3f}. Community boundaries depend heavily on a few generic hubs "
    "(see Method page), so treat colours as one reading of the network."
)

st.subheader("Communities")
st.dataframe(
    ui.cached_table("rq2_community_profile")[
        [
            "community",
            "entities",
            "incidents",
            "organisations",
            "ai_systems",
            "harmed_groups",
            "domains",
        ]
    ],
    hide_index=True,
    width="stretch",
)
st.subheader("Most central organisations (PageRank)")
st.dataframe(ui.cached_table("rq2_top_organisation_pagerank"), hide_index=True, width="stretch")
