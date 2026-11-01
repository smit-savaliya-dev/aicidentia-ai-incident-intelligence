"""Phase 5 orchestration: network, centrality, communities, temporal evolution, dashboard export, artefacts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from aicidentia import registry
from aicidentia.config import PROJECT_ROOT, load_config
from aicidentia.data import clean
from aicidentia.models import network as N
from aicidentia.viz import plots, style

logger = logging.getLogger(__name__)
PRODUCER = "aicidentia.models.network_outputs.run_network"
SOURCE = "data/processed/incident_entities.parquet"
GRAPH_JSON = PROJECT_ROOT / "reports" / "network_graph.json"


@dataclass
class NetworkResults:
    """Everything Phase 5 produces."""

    nd: N.NetworkData
    hetero: nx.Graph
    graph: nx.Graph
    centrality: pd.DataFrame
    tau_all: pd.DataFrame
    tau_orgs: pd.DataFrame
    tops: dict[str, dict[str, pd.DataFrame]]
    projection: pd.DataFrame
    runs: pd.DataFrame
    method_summary: pd.DataFrame
    chosen_method: str
    chosen_reason: str
    near_tie: bool
    medoid_seed: int
    assignment: dict[str, int]
    profile: pd.DataFrame
    association: dict[str, float]
    robustness: dict[str, Any]
    snapshots: pd.DataFrame
    top_orgs: pd.DataFrame
    overlap: pd.DataFrame
    export_info: dict[str, Any] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)


def medoid_seed(
    assignments: dict[tuple[str, int], dict[str, int]],
    method: str,
    seeds: list[int],
    nodes: list[str],
) -> int:
    """The seed whose partition is most similar (highest mean ARI) to the other seeds' partitions."""
    labels = {s: [assignments[(method, s)][n] for n in nodes] for s in seeds}
    score = {
        s: np.mean([adjusted_rand_score(labels[s], labels[t]) for t in seeds if t != s])
        for s in seeds
    }
    return int(max(score, key=lambda s: (score[s], -s)))


def hub_robustness(
    graph: nx.Graph,
    nd: N.NetworkData,
    method: str,
    seeds: list[int],
    main: dict[str, int],
    cap_share: float,
) -> dict[str, Any]:
    """Re-detect communities after dropping entities present in more than `cap_share` of all incidents."""
    limit = cap_share * nd.incidents["incident_id"].nunique()
    hubs = [n for n, d in graph.nodes(data=True) if d["n_incidents"] > limit]
    reduced = graph.copy()
    reduced.remove_nodes_from(hubs)
    reduced.remove_nodes_from(list(nx.isolates(reduced)))
    shared = [n for n in reduced.nodes if n in main]
    aris, mods = [], []
    for seed in seeds:
        part = N.detect_communities(reduced, method, seed)
        mods.append(N.modularity(reduced, part))
        aris.append(adjusted_rand_score([main[n] for n in shared], [part[n] for n in shared]))
    return {
        "hubs_removed": [graph.nodes[h]["label"] for h in hubs],
        "n_hubs": len(hubs),
        "limit_incidents": float(limit),
        "nodes_after": reduced.number_of_nodes(),
        "modularity_mean": float(np.mean(mods)),
        "ari_vs_main_mean": float(np.mean(aris)),
        "ari_vs_main_min": float(np.min(aris)),
    }


def run_network() -> NetworkResults:
    """Build the networks and run every Phase 5 analysis."""
    cfg = load_config()["network"]
    base = int(load_config()["random_seed"])
    seeds = list(range(base, base + int(cfg["n_seeds"])))
    nd = N.prepare(
        clean.load_processed("incident_entities"),
        clean.load_processed("incidents_clean"),
        cfg["include_individual_candidates"],
    )
    hetero = N.build_graph(nd)
    graph = N.entity_network(nd, cfg["min_incidents"])
    cent = N.centrality_table(graph)
    tau_all = N.kendall_matrix(cent)
    tau_orgs = N.kendall_matrix(cent[cent["type"] == "organisation"])
    n = cfg["top_n_table"]
    tops = {
        m: {t: N.top_by_type(cent, m, t, n) for t in ("organisation", "ai_system", "harmed_group")}
        for m in ("pagerank", "weighted_degree")
    }
    projection = N.bipartite_projection(nd)

    runs, summary, assignments = N.compare_community_methods(graph, seeds)
    method, reason = N.choose_method(summary, cfg["min_stability_ari"])
    ranked = summary.sort_values("modularity_mean", ascending=False)
    top1, top2 = ranked.iloc[0], ranked.iloc[1]
    near_tie = bool(
        abs(top1["modularity_mean"] - top2["modularity_mean"])
        < max(top1["modularity_sd"], top2["modularity_sd"])
    )
    nodes = list(graph.nodes)
    seed = medoid_seed(assignments, method, seeds, nodes)
    assignment = assignments[(method, seed)]
    profile, assoc = N.profile_communities(graph, assignment, nd)
    robustness = hub_robustness(graph, nd, method, seeds, assignment, cfg["hub_share_cap"])

    complete_last = int(registry.get("audit.last_complete_year"))
    partial_year = int(registry.get("audit.partial_year"))
    years = list(range(2015, partial_year + 1))
    snapshots = N.cumulative_snapshots(nd, years)
    top_orgs = N.top_organisations_over_time(nd, years, n)
    overlap = N.top_set_overlap(top_orgs)

    info = N.export_graph_json(
        graph,
        assignment,
        cent,
        nd,
        GRAPH_JSON,
        cfg["export"]["top_n"],
        cfg["export"]["max_edges"],
        str(load_config()["snapshot_date"]),
    )
    hs = N.graph_summary(hetero)
    comp = community_sizes(assignment)
    res = NetworkResults(
        nd,
        hetero,
        graph,
        cent,
        tau_all,
        tau_orgs,
        tops,
        projection,
        runs,
        summary,
        method,
        reason,
        near_tie,
        seed,
        assignment,
        profile,
        assoc,
        robustness,
        snapshots,
        top_orgs,
        overlap,
        info,
    )
    res.facts = {
        "hetero_nodes": hs["nodes"],
        "hetero_edges": hs["edges"],
        "hetero_nodes_by_type": hs["nodes_by_type"],
        "entity_network_nodes": graph.number_of_nodes(),
        "entity_network_edges": graph.number_of_edges(),
        "entity_network_components": nx.number_connected_components(graph),
        "min_incidents": cfg["min_incidents"],
        "excluded_types": nd.excluded_types,
        "excluded_person_like_mentions": int(
            clean.load_processed("incident_entities")
            .query("entity_type == 'individual_candidate'")
            .shape[0]
        ),
        "chosen_method": method,
        "chosen_reason": reason,
        "near_tie_modularity": near_tie,
        "medoid_seed": seed,
        "chosen_modularity": float(N.modularity(graph, assignment)),
        "n_communities": len(comp),
        "community_sizes": comp,
        "nmi_community_vs_domain": assoc["nmi_community_vs_domain"],
        "hub_robustness": robustness,
        "tau_pagerank_vs_weighted_degree": float(tau_all.loc["pagerank", "weighted_degree"]),
        "tau_min_between_measures": float(
            tau_all.loc[list(N.CENTRALITY), list(N.CENTRALITY)]
            .where(~np.eye(len(N.CENTRALITY), dtype=bool))
            .min()
            .min()
        ),
        "organisations_in_network": int((cent["type"] == "organisation").sum()),
        "last_complete_year": complete_last,
        "partial_year": partial_year,
        "top_org_pagerank": tops["pagerank"]["organisation"]["label"].tolist(),
        "top_org_weighted_degree": tops["weighted_degree"]["organisation"]["label"].tolist(),
        "median_top10_jaccard": float(overlap["jaccard_with_previous"].median())
        if len(overlap)
        else float("nan"),
        "sector_coverage_incidents": int(nd.incidents["sectors"].map(len).gt(0).sum()),
        "export": info,
    }
    return res


def community_sizes(assignment: dict[str, int]) -> dict[int, int]:
    """Community sizes, largest first."""
    sizes = pd.Series(assignment).value_counts()
    return {int(k): int(v) for k, v in sizes.items()}


# --------------------------------------------------------------------------------------
# Artefacts
# --------------------------------------------------------------------------------------


def register(res: NetworkResults) -> None:
    """Record Phase 5 facts under `rq2.*`."""
    for key, value in res.facts.items():
        registry.record(f"rq2.{key}", value, SOURCE, PRODUCER)
    registry.record("rq2.method_summary", res.method_summary, SOURCE, PRODUCER)
    registry.record("rq2.community_profile", res.profile, SOURCE, PRODUCER)
    registry.record("rq2.tau_all", res.tau_all.reset_index(names="measure"), SOURCE, PRODUCER)
    registry.record("rq2.snapshots", res.snapshots, SOURCE, PRODUCER)


def save_tables(res: NetworkResults) -> None:
    """reports/tables/rq2_*.{csv,md} and data/processed/entity_communities.parquet."""
    cent = res.centrality.copy()
    cent["community"] = cent["node"].map(res.assignment)
    for col in ("betweenness", "pagerank", "eigenvector"):
        cent[col] = cent[col].round(6)
    style.save_table(cent.sort_values("pagerank", ascending=False), "rq2_centrality")
    clean.save_processed(cent, "entity_communities")
    for measure, by_type in res.tops.items():
        for node_type, table in by_type.items():
            style.save_table(table.round(5), f"rq2_top_{node_type}_{measure}")
    style.save_table(res.tau_all.round(3), "rq2_kendall_tau_all", index=True)
    style.save_table(res.tau_orgs.round(3), "rq2_kendall_tau_organisations", index=True)
    style.save_table(res.projection.head(100), "rq2_org_harmed_group_projection")
    style.save_table(res.runs.round(4), "rq2_community_runs")
    style.save_table(res.method_summary.round(4), "rq2_community_methods")
    style.save_table(res.profile.round(3), "rq2_community_profile")
    style.save_table(res.snapshots, "rq2_snapshots")
    style.save_table(res.top_orgs.round(5), "rq2_top_organisations_by_year")
    style.save_table(res.overlap.round(3), "rq2_top10_overlap")


def make_figures(res: NetworkResults) -> dict[str, Any]:
    """Six figures."""
    top = res.tops["pagerank"]
    figs = {
        "fig_rq2_1_entity_network": plots.plot_entity_network(res.graph, res.assignment),
        "fig_rq2_2_top_central": plots.plot_top_central(
            {
                "Alleged organisations": top["organisation"],
                "Implicated AI systems": top["ai_system"],
                "Harmed groups": top["harmed_group"],
            },
            "pagerank",
        ),
        "fig_rq2_3_centrality_tau": plots.plot_tau_heatmap(res.tau_all),
        "fig_rq2_4_community_methods": plots.plot_community_methods(
            res.runs, res.method_summary, res.chosen_method
        ),
        "fig_rq2_5_network_growth": plots.plot_network_growth(
            res.snapshots, res.facts["partial_year"]
        ),
        "fig_rq2_6_top_organisations_over_time": plots.plot_rank_bump(res.top_orgs),
    }
    for name, fig in figs.items():
        style.save_figure(fig, name, str(load_config()["snapshot_date"]))
    return figs


def write_all(res: NetworkResults) -> None:
    """Registry, tables, figures and docs/rq2_network.md."""
    from aicidentia.models import network_docs

    register(res)
    save_tables(res)
    make_figures(res)
    network_docs.write_network_doc(res, PROJECT_ROOT / "docs" / "rq2_network.md")
