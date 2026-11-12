"""AICIDENTIA Intelligence - Risk Network (RQ2): ego network of an organisation, system or group."""

import streamlit as st

from aicidentia.dashboard import charts, data, ui

flt = ui.setup(
    "Risk Network",
    "RQ2: who appears together in incidents. Pick an organisation, AI system or group to see its neighbours, "
    "its risk domains and its incidents. Organisations are ALLEGED developers or deployers; named persons are excluded.",
)
f = data.fact
inc = ui.cached_incidents()
ents = ui.cached_entities()
links = ui.cached_links()

c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
search = c1.text_input(
    "Search an organisation, system or group", "", placeholder="e.g. Meta, Tesla, deepfake"
)
pool = ents[ents["n_incidents"] >= 2]
if search.strip():
    pool = pool[pool["entity"].str.contains(search.strip(), case=False, regex=False)]
options = list(pool["entity_id"][:300])
labels = dict(
    zip(
        ents["entity_id"],
        ents["entity"]
        + " · "
        + ents["type_name"]
        + " · "
        + ents["n_incidents"].astype(str)
        + " incidents",
        strict=True,
    )
)
default = options.index("meta") if "meta" in options else 0
if not options:
    st.warning("No entity matches the search.")
    ui.footer()
    st.stop()
pick = c1.selectbox("Entity", options, index=default, format_func=lambda e: labels.get(e, e))
year_max = c2.slider("Incidents up to year", flt.years[0], flt.years[1], flt.years[1])
top_n = c3.slider("Neighbours shown (top N)", 5, 40, 20, step=5)
colour_by = c4.radio("Colour by", ["community", "node type"], horizontal=True)

window = data.Filters((flt.years[0], year_max), flt.domains, flt.sectors)
sub = data.filter_incidents(inc, window)
ego = data.ego_network(pick, links, ents, set(sub["incident_id"]), top_n=top_n)
profile = data.entity_profile(pick, links, sub)
ui.filter_note(window, len(sub), len(inc))

left, right = st.columns([3, 2])
with left:
    if ego["incidents"] == 0:
        st.info("This entity has no incidents within the filters.")
    else:
        st.plotly_chart(
            charts.ego_network(ego, "community" if colour_by == "community" else "type"),
            width="stretch",
        )
        st.caption(
            f"{labels.get(pick, pick).split(' · ')[0]} and its {len(ego['nodes']) - 1} most frequent co-occurring entities in "
            f"{ego['incidents']} incidents ({window.years[0]}-{window.years[1]}). Neighbours sit on a ring grouped by community; node size and spoke width = shared incidents; faint lines join neighbours that often appear together; "
            "symbol = type (circle organisation, square AI system, triangle harmed group). Communities come from Leiden on the full network (Phase 5)."
        )
with right:
    row = ents.set_index("entity_id").loc[pick]
    k = st.columns(2)
    ui.kpi(k[0], "Incidents", f"{profile['incidents']:,}", profile["years"])
    comm = "none" if row["community"] != row["community"] else f"{int(row['community'])}"
    ui.kpi(k[1], "Community", comm, row["type_name"])
    st.markdown("**Risk domains of its incidents**")
    if len(profile["domains"]):
        st.plotly_chart(
            charts.horizontal_bars(profile["domains"].head(7), "", "Incidents"), width="stretch"
        )
    else:
        st.caption("No MIT-labelled incidents within the filters.")
    st.markdown("**Sectors (CSETv1)**")
    if len(profile["sectors"]):
        st.write(", ".join(f"{s} ({n})" for s, n in profile["sectors"].head(5).items()))
        st.caption(f"From the {profile['sector_coverage']} of its incidents with a CSETv1 sector.")
    else:
        st.caption("None of its incidents in view has a CSETv1 sector annotation.")
    roles = ", ".join(f"{r} {n}" for r, n in profile["roles"].items())
    st.caption(f"Roles recorded by AIID: {roles or 'none'} (alleged).")

st.markdown("### Linked incidents")
ui.incident_table(profile["linked"], height=300)
ui.download(profile["linked"], f"incidents_{pick}")

with st.expander("Most central organisations in the whole network (PageRank)"):
    top = ui.cached_table("rq2_top_organisation_pagerank")
    st.dataframe(top, hide_index=True, width="stretch")
    st.caption(
        "Centrality reflects how often an organisation appears in reported incidents with others, not responsibility."
    )
ui.footer()
