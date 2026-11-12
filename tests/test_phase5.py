"""Phase 5 tests: network construction, projections, centrality, communities, temporal snapshots, export.

Fixtures are tiny hand-built networks with known structure (two dense groups joined by one weak tie),
not analysis data.
"""

import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from aicidentia.models import network as N
from aicidentia.models import network_outputs as O


def _incidents():
    rows = []
    for i in range(1, 9):
        rows.append(
            {
                "incident_id": i,
                "year": 2020 + (i - 1) // 2,
                "title": f"Incident {i}",
                "mit_risk_domain": "Misinformation" if i <= 4 else "AI system safety",
                "cset_sector_of_deployment": np.array(["transport"])
                if i == 1
                else np.array([], dtype=object),
            }
        )
    return pd.DataFrame(rows)


def _entities():
    """Group A (incidents 1-4): OrgA, SysA, GroupA. Group B (5-8): OrgB, SysB, GroupB. One bridge incident 4 also names OrgB."""
    rows = []

    def add(iid, role, eid, name, etype):
        rows.append(
            {
                "incident_id": iid,
                "role": role,
                "entity_id": eid,
                "entity": name,
                "entity_type": etype,
            }
        )

    for i in range(1, 5):
        add(i, "deployer", "org-a", "OrgA", "organisation")
        add(i, "system", "sys-a", "SysA", "technology_system")
        add(i, "harmed", "grp-a", "GroupA", "population_group")
    for i in range(5, 9):
        add(i, "deployer", "org-b", "OrgB", "organisation")
        add(i, "system", "sys-b", "SysB", "technology_system")
        add(i, "harmed", "grp-b", "GroupB", "population_group")
    add(4, "developer", "org-b", "OrgB", "organisation")  # the weak bridge
    add(1, "harmed", "jane-doe", "Jane Doe", "individual_candidate")  # must be excluded
    add(2, "deployer", "scam", "Scammers", "generic_actor_group")
    add(2, "harmed", "priv", "Privacy", "abstract_value")
    add(1, "developer", "org-a", "OrgA", "organisation")  # same entity, second role in incident 1
    return pd.DataFrame(rows)


@pytest.fixture
def nd():
    return N.prepare(_entities(), _incidents())


# ------------------------------------------------------------------ preparation and graphs
def test_prepare_excludes_individuals_and_types_nodes(nd):
    assert "jane-doe" not in set(nd.links["entity_id"]) and nd.excluded_types == [
        "individual_candidate"
    ]
    types = nd.links.drop_duplicates("entity_id").set_index("entity_id")["node_type"].to_dict()
    assert (
        types["org-a"] == "organisation"
        and types["sys-a"] == "ai_system"
        and types["grp-a"] == "harmed_group"
    )
    assert types["scam"] == "generic_actor" and types["priv"] == "harm_value"
    assert nd.links["year"].notna().all()
    with_people = N.prepare(_entities(), _incidents(), include_individuals=True)
    assert "jane-doe" in set(with_people.links["entity_id"])


def test_build_graph_edge_types_and_roles_merge(nd):
    g = N.build_graph(nd)
    summary = N.graph_summary(g)
    assert (
        summary["nodes_by_type"]["incident"] == 8 and summary["nodes_by_type"]["risk_domain"] == 2
    )
    assert summary["nodes_by_type"]["sector"] == 1
    assert (
        g["incident:1"]["organisation:org-a"]["etype"] == "alleged_deployer+alleged_developer"
    )  # one link, combined role
    assert g["incident:1"]["risk_domain:Misinformation"]["etype"] == "domain"
    assert g["incident:1"]["sector:transport"]["etype"] == "sector"
    assert g["incident:4"]["organisation:org-b"]["etype"] == "alleged_developer"
    assert (
        nx.number_connected_components(g) == 1
    )  # the bridge and shared domain nodes connect everything


def test_build_graph_year_filter_and_min_incidents(nd):
    early = N.build_graph(nd, year_max=2020)
    assert {d["year"] for _, d in early.nodes(data=True) if d["type"] == "incident"} == {2020}
    assert "organisation:org-b" not in early  # not yet mentioned
    pruned = N.build_graph(nd, min_incidents=5)
    assert (
        "organisation:org-a" not in pruned and "organisation:org-b" in pruned
    )  # 4 vs 5 incidents (incl. bridge)


def test_entity_network_weights_years_and_attributes(nd):
    g = N.entity_network(nd, min_incidents=1)
    assert g["org-a"]["sys-a"]["weight"] == 4
    assert g["org-a"]["sys-a"]["by_year"] == {2020: 2, 2021: 2}
    assert g["org-a"]["org-b"]["weight"] == 1  # only incident 4 names both
    assert g.nodes["org-a"]["n_incidents"] == 4 and g.nodes["org-a"]["first_year"] == 2020
    assert "jane-doe" not in g
    assert "scam" not in N.entity_network(nd, min_incidents=2)  # appears once
    assert "org-b" not in N.entity_network(nd, min_incidents=1, year_max=2020)


def test_bipartite_projection_counts_deployer_developer_with_harmed(nd):
    proj = N.bipartite_projection(nd).set_index(["left_id", "right_id"])["weight"]
    assert proj[("org-a", "grp-a")] == 4 and proj[("org-b", "grp-b")] == 4
    assert proj[("org-b", "grp-a")] == 1  # bridge incident: OrgB as developer, GroupA harmed


# ------------------------------------------------------------------ centrality
def _path():
    g = nx.Graph()
    for node in "abcde":
        g.add_node(node, type="organisation", label=node.upper(), n_incidents=1)
    for a, b, w in (("a", "b", 1), ("b", "c", 3), ("c", "d", 1), ("d", "e", 1)):
        g.add_edge(a, b, weight=w, by_year={2020: w})
    return g


def test_centrality_table_known_values():
    ct = N.centrality_table(_path()).set_index("node")
    assert ct.loc["c", "betweenness"] > ct.loc["b", "betweenness"] > ct.loc["a", "betweenness"] == 0
    assert (
        ct.loc["b", "degree"] == 2
        and ct.loc["b", "weighted_degree"] == 4
        and ct.loc["a", "weighted_degree"] == 1
    )
    assert ct["pagerank"].sum() == pytest.approx(1.0)
    assert (ct["eigenvector"] >= 0).all()
    assert set(ct.columns) >= set(N.CENTRALITY)


def test_betweenness_uses_inverse_weights_as_lengths():
    # a-b-c strongly tied, a-c weakly tied: the strong two-step path is shorter, so b lies on a-c's shortest path.
    g = nx.Graph()
    for node in "abc":
        g.add_node(node, type="organisation", label=node, n_incidents=1)
    g.add_edge("a", "b", weight=10, by_year={})
    g.add_edge("b", "c", weight=10, by_year={})
    g.add_edge("a", "c", weight=1, by_year={})
    assert N.centrality_table(g).set_index("node").loc["b", "betweenness"] == pytest.approx(1.0)


def test_kendall_matrix_diagonal_and_range():
    table = pd.DataFrame(
        {"degree": [1, 2, 3, 4], "pagerank": [4, 3, 2, 1], "n_incidents": [1, 2, 3, 4]}
    )
    tau = N.kendall_matrix(table, ("degree", "pagerank", "n_incidents"))
    assert (
        (np.diag(tau) == 1).all()
        and tau.loc["degree", "pagerank"] == -1
        and tau.loc["degree", "n_incidents"] == 1
    )


def test_top_by_type_orders_and_filters():
    table = pd.DataFrame(
        {
            "label": ["x", "y", "z"],
            "type": ["organisation", "organisation", "ai_system"],
            "n_incidents": [1, 2, 3],
            **{m: [0.1, 0.5, 0.9] for m in N.CENTRALITY},
        }
    )
    out = N.top_by_type(table, "pagerank", "organisation", 5)
    assert out["label"].tolist() == ["y", "x"]


# ------------------------------------------------------------------ communities
def _two_cliques():
    g = nx.Graph()
    for prefix in "pq":
        for i in range(6):
            g.add_node(
                f"{prefix}{i}",
                type="organisation",
                label=f"{prefix}{i}",
                n_incidents=3,
                first_year=2020,
                last_year=2021,
            )
        for i in range(6):
            for j in range(i + 1, 6):
                g.add_edge(f"{prefix}{i}", f"{prefix}{j}", weight=5, by_year={2020: 5})
    g.add_edge("p0", "q0", weight=1, by_year={2020: 1})
    return g


@pytest.mark.parametrize("method", ["louvain", "leiden"])
def test_modularity_methods_recover_two_cliques(method):
    g = _two_cliques()
    part = N.detect_communities(g, method, seed=1)
    assert len(set(part.values())) == 2
    assert len({part[f"p{i}"] for i in range(6)}) == 1 and part["p0"] != part["q0"]
    assert N.modularity(g, part) > 0.4
    assert part == N.detect_communities(g, method, seed=1)  # deterministic given the seed


def test_label_propagation_runs_and_unknown_method_raises():
    g = _two_cliques()
    part = N.detect_communities(g, "label_propagation", seed=0)
    assert set(part) == set(g.nodes)
    with pytest.raises(ValueError):
        N.detect_communities(g, "spectral", 0)


def test_compare_methods_summary_and_choice_rule():
    g = _two_cliques()
    runs, summary, assignments = N.compare_community_methods(g, [1, 2, 3])
    assert set(summary["method"]) == set(N.METHODS) and len(runs) == 9 and len(assignments) == 9
    assert summary.set_index("method").loc["louvain", "stability_ari_mean"] == pytest.approx(1.0)
    method, reason = N.choose_method(summary, 0.5)
    assert method in {"louvain", "leiden"} and "modularity" in reason
    stable_only = pd.DataFrame(
        {"method": ["a", "b"], "modularity_mean": [0.9, 0.3], "stability_ari_mean": [0.2, 0.9]}
    )
    assert N.choose_method(stable_only, 0.5)[0] == "b"  # high modularity but unstable is rejected
    unstable = stable_only.assign(stability_ari_mean=[0.1, 0.3])
    assert (
        N.choose_method(unstable, 0.5)[0] == "b"
        and "most stable" in N.choose_method(unstable, 0.5)[1]
    )


def test_medoid_seed_picks_most_representative_partition():
    nodes = ["a", "b", "c", "d"]
    same = {"a": 0, "b": 0, "c": 1, "d": 1}
    outlier = {"a": 0, "b": 1, "c": 0, "d": 1}
    assignments = {("m", 1): same, ("m", 2): same, ("m", 3): outlier}
    assert O.medoid_seed(assignments, "m", [1, 2, 3], nodes) in {1, 2}


def test_profile_communities_assigns_incidents_and_profiles_domains(nd):
    g = N.entity_network(nd, min_incidents=2)
    part = N.detect_communities(g, "louvain", 1)
    profile, assoc = N.profile_communities(g, part, nd, min_incidents_for_mix=1)
    assert profile["incidents"].sum() == assoc["incidents_assigned"]
    assert 0 <= assoc["nmi_community_vs_domain"] <= 1
    top = profile.iloc[0]
    assert (
        top["top_domain_share"] >= 0.5
        and "OrgA" in top["organisations"] + profile.iloc[-1]["organisations"]
    )


def test_assign_incidents_plurality():
    links = pd.DataFrame({"incident_id": [1, 1, 1, 2], "entity_id": ["a", "b", "c", "d"]})
    got = N.assign_incidents(links, {"a": 0, "b": 1, "c": 1, "d": 0})
    assert got.to_dict() == {1: 1, 2: 0}


# ------------------------------------------------------------------ temporal and export
def test_cumulative_snapshots_grow_monotonically(nd):
    snap = N.cumulative_snapshots(nd, [2020, 2021, 2022, 2023])
    assert snap["nodes"].is_monotonic_increasing and snap["edges"].is_monotonic_increasing
    assert snap["new_organisations"].sum() == snap["organisations"].iloc[-1] == 2
    assert snap["new_organisations"].tolist() == [
        1,
        1,
        0,
        0,
    ]  # OrgA in 2020; OrgB first named by the bridge incident 4 (2021)
    assert snap.loc[snap["year"] == 2020, "nodes_incident"].iloc[0] == 2


def test_top_organisations_and_overlap(nd):
    top = N.top_organisations_over_time(nd, [2020, 2021, 2022, 2023], n=2)
    assert set(top["year"]) <= {2020, 2021, 2022, 2023} and top["rank"].min() == 1
    overlap = N.top_set_overlap(top)
    assert overlap["jaccard_with_previous"].between(0, 1).all()


def test_export_graph_json_caps_and_year_data(nd, tmp_path):
    g = N.entity_network(nd, min_incidents=1)
    part = N.detect_communities(g, "louvain", 1)
    ct = N.centrality_table(g)
    info = N.export_graph_json(
        g, part, ct, nd, tmp_path / "net.json", top_n=6, max_edges=5, snapshot="TEST"
    )
    data = json.loads((tmp_path / "net.json").read_text(encoding="utf-8"))
    assert info["nodes"] == len(data["nodes"]) <= 6 and len(data["edges"]) <= 5
    ids = {n["id"] for n in data["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in data["edges"])
    assert all(sum(e["by_year"].values()) == e["weight"] for e in data["edges"])
    assert "alleged" in data["note"] and data["snapshot"] == "TEST"
    node = data["nodes"][0]
    assert {"community", "n_incidents", "first_year", "domains", "recent_incidents"} <= set(node)


def test_small_communities_do_not_reveal_domain_mix(nd):
    g = N.entity_network(nd, min_incidents=2)
    part = N.detect_communities(g, "louvain", 1)
    profile, _ = N.profile_communities(g, part, nd)  # fixture communities have 4-5 incidents (< 20)
    assert profile["domains"].str.startswith("(not shown").all()
    assert profile["top_domain"].eq("").all() and profile["top_domain_share"].isna().all()


def test_export_is_identical_across_python_hash_seeds(tmp_path):
    """Regression: edge order and the max_edges cut must not depend on PYTHONHASHSEED."""
    import subprocess
    import sys

    code = (
        "import sys, json; sys.path.insert(0, r'{tests}');"
        "import test_phase5 as t; from aicidentia.models import network as N;"
        "nd = N.prepare(t._entities(), t._incidents()); g = N.entity_network(nd, 1);"
        "part = N.detect_communities(g, 'louvain', 1); ct = N.centrality_table(g);"
        "N.export_graph_json(g, part, ct, nd, __import__('pathlib').Path(sys.argv[1]), top_n=6, max_edges=4)"
    ).format(tests=str(__import__("pathlib").Path(__file__).parent))
    outputs = []
    for seed in ("1", "2", "3"):
        out = tmp_path / f"g{seed}.json"
        env = {**__import__("os").environ, "PYTHONHASHSEED": seed}
        subprocess.run([sys.executable, "-c", code, str(out)], check=True, env=env)
        outputs.append(out.read_text(encoding="utf-8"))
    assert outputs[0] == outputs[1] == outputs[2]
    data = json.loads(outputs[0])
    assert all(e["source"] < e["target"] for e in data["edges"])
