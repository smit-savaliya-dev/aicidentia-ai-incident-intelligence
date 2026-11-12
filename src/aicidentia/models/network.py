"""RQ2 - the network of incidents, organisations, AI systems, harmed groups, risk domains and sectors.

Two views are built from `incident_entities` (Phase 2) and `incidents_clean`:

* a typed **heterogeneous graph**: incident nodes linked to entity nodes (typed by the role the
  entity plays: alleged deployer, alleged developer, harmed party, implicated system), to a MIT
  risk-domain node and to CSETv1 sector nodes;
* a weighted **entity co-occurrence network** (the one-mode projection onto entities): two entities
  are linked with weight = number of incidents they share. Centrality and communities are computed here.

Organisations are *alleged* deployers/developers as AIID records them; a high centrality means frequent
involvement in reported incidents, never blame. Person-like entities (`individual_candidate`) are
excluded: RQ2 concerns organisations, systems and groups, and unreviewed individuals are not placed in a
network. Risk-domain and sector nodes are kept out of community detection so that profiling communities by
domain is not circular.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import igraph as ig
import leidenalg
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community as nxc
from scipy.stats import kendalltau
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

logger = logging.getLogger(__name__)

NODE_TYPE = {
    "organisation": "organisation",
    "technology_system": "ai_system",
    "population_group": "harmed_group",
    "generic_actor_group": "generic_actor",
    "abstract_value": "harm_value",
    "unknown": "other",
    "individual_candidate": "person",  # only present when include_individuals=True
}
ENTITY_NODE_TYPES = tuple(dict.fromkeys(NODE_TYPE.values()))
ROLE_EDGE = {
    "deployer": "alleged_deployer",
    "developer": "alleged_developer",
    "harmed": "harmed",
    "system": "implicated_system",
}
METHODS = ("louvain", "leiden", "label_propagation")
CENTRALITY = ("degree", "weighted_degree", "betweenness", "pagerank", "eigenvector")


@dataclass
class NetworkData:
    """Inputs of the network: entity links (typed) and the incident attributes needed for profiling."""

    links: pd.DataFrame  # incident_id, role, entity_id, entity, node_type, year
    incidents: pd.DataFrame  # incident_id, year, title, domain, sectors (list)
    excluded_types: list[str] = field(default_factory=list)


def prepare(
    incident_entities: pd.DataFrame, incidents: pd.DataFrame, include_individuals: bool = False
) -> NetworkData:
    """Attach node types and years, drop person-like entities (unless asked), tidy incident attributes."""
    links = incident_entities.copy()
    excluded: list[str] = []
    if not include_individuals:
        links = links[links["entity_type"] != "individual_candidate"]
        excluded.append("individual_candidate")
    links["node_type"] = links["entity_type"].map(NODE_TYPE)
    links = links[links["node_type"].notna()]
    inc = incidents[
        ["incident_id", "year", "title", "mit_risk_domain", "cset_sector_of_deployment"]
    ].rename(columns={"mit_risk_domain": "domain", "cset_sector_of_deployment": "sectors"})
    inc["sectors"] = inc["sectors"].map(lambda v: [str(s) for s in v] if v is not None else [])
    links = links.merge(inc[["incident_id", "year"]], on="incident_id", how="inner")
    return NetworkData(links.reset_index(drop=True), inc.reset_index(drop=True), excluded)


def node_id(kind: str, key: Any) -> str:
    """Stable node id such as 'organisation:openai' or 'incident:23'."""
    return f"{kind}:{key}"


# --------------------------------------------------------------------------------------
# Heterogeneous graph
# --------------------------------------------------------------------------------------


def build_graph(nd: NetworkData, year_max: int | None = None, min_incidents: int = 1) -> nx.Graph:
    """Typed heterogeneous graph of incidents, entities, risk domains and sectors.

    Edge attribute `etype` is the role ('alleged_deployer', 'alleged_developer', 'harmed',
    'implicated_system', joined by '+' if one entity plays several roles in an incident), 'domain' or
    'sector'. `year_max` keeps only incidents up to that year (cumulative snapshots); `min_incidents`
    drops entities in fewer incidents (computed after the year filter).
    """
    links, inc = nd.links, nd.incidents
    if year_max is not None:
        links, inc = links[links["year"] <= year_max], inc[inc["year"] <= year_max]
    support = links.groupby("entity_id")["incident_id"].nunique()
    links = links[links["entity_id"].isin(support[support >= min_incidents].index)]
    graph = nx.Graph()
    for row in inc.itertuples():
        graph.add_node(
            node_id("incident", row.incident_id),
            type="incident",
            label=row.title,
            year=int(row.year),
        )
        if isinstance(row.domain, str) and row.domain:
            d = node_id("risk_domain", row.domain)
            graph.add_node(d, type="risk_domain", label=row.domain)
            graph.add_edge(node_id("incident", row.incident_id), d, etype="domain", weight=1)
        for sector in row.sectors:
            s = node_id("sector", sector)
            graph.add_node(s, type="sector", label=sector)
            graph.add_edge(node_id("incident", row.incident_id), s, etype="sector", weight=1)
    roles: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in links.itertuples():
        roles[(row.incident_id, row.entity_id)].add(ROLE_EDGE[row.role])
        graph.add_node(
            node_id(row.node_type, row.entity_id),
            type=row.node_type,
            label=row.entity,
            entity_id=row.entity_id,
        )
    node_type = links.drop_duplicates("entity_id").set_index("entity_id")["node_type"].to_dict()
    for (incident_id, entity_id), types in roles.items():
        graph.add_edge(
            node_id("incident", incident_id),
            node_id(node_type[entity_id], entity_id),
            etype="+".join(sorted(types)),
            weight=1,
        )
    return graph


def graph_summary(graph: nx.Graph) -> dict[str, Any]:
    """Node counts per type, edge counts per type, density and connected components."""
    nodes = Counter(d["type"] for _, d in graph.nodes(data=True))
    edges = Counter(d["etype"] for _, _, d in graph.edges(data=True))
    return {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "nodes_by_type": dict(nodes),
        "edges_by_type": dict(edges),
        "components": int(nx.number_connected_components(graph)) if graph.number_of_nodes() else 0,
    }


# --------------------------------------------------------------------------------------
# One-mode projections
# --------------------------------------------------------------------------------------


def entity_network(
    nd: NetworkData, min_incidents: int = 2, year_max: int | None = None
) -> nx.Graph:
    """Weighted entity co-occurrence network (projection of the incident graph onto entities).

    Edge weight = number of incidents in which both entities appear; `by_year` holds the per-year counts
    (used for year filtering). Node attributes: type, label, n_incidents, first_year, last_year.
    """
    links = nd.links if year_max is None else nd.links[nd.links["year"] <= year_max]
    support = links.groupby("entity_id")["incident_id"].nunique()
    links = links[links["entity_id"].isin(support[support >= min_incidents].index)]
    graph = nx.Graph()
    meta = links.groupby("entity_id").agg(
        label=("entity", "first"),
        node_type=("node_type", "first"),
        n=("incident_id", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
    )
    for entity_id, row in meta.iterrows():
        graph.add_node(
            entity_id,
            type=row["node_type"],
            label=row["label"],
            n_incidents=int(row["n"]),
            first_year=int(row["first_year"]),
            last_year=int(row["last_year"]),
        )
    year_of = nd.incidents.set_index("incident_id")["year"].to_dict()
    pairs: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    for incident_id, group in links.groupby("incident_id"):
        entities = sorted(set(group["entity_id"]))
        for a, b in itertools.combinations(entities, 2):
            pairs[(a, b)][int(year_of[incident_id])] += 1
    for (a, b), by_year in pairs.items():
        graph.add_edge(a, b, weight=int(sum(by_year.values())), by_year=dict(by_year))
    return graph


def bipartite_projection(
    nd: NetworkData, left: str = "organisation", right: str = "harmed_group"
) -> pd.DataFrame:
    """Weighted organisation-harmed group links: incidents where the organisation is an alleged deployer or
    developer and the group is a harmed party. Columns: left_id, left, right_id, right, weight."""
    links = nd.links
    org = links[(links["node_type"] == left) & links["role"].isin(["deployer", "developer"])][
        ["incident_id", "entity_id", "entity"]
    ]
    grp = links[(links["node_type"] == right) & (links["role"] == "harmed")][
        ["incident_id", "entity_id", "entity"]
    ]
    merged = org.drop_duplicates().merge(
        grp.drop_duplicates(), on="incident_id", suffixes=("_l", "_r")
    )
    out = (
        merged.groupby(["entity_id_l", "entity_l", "entity_id_r", "entity_r"])
        .size()
        .rename("weight")
        .reset_index()
        .rename(
            columns={
                "entity_id_l": "left_id",
                "entity_l": "left",
                "entity_id_r": "right_id",
                "entity_r": "right",
            }
        )
    )
    return out.sort_values("weight", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Centrality
# --------------------------------------------------------------------------------------


def _to_igraph(graph: nx.Graph) -> tuple[ig.Graph, list[str]]:
    names = list(graph.nodes)
    index = {n: i for i, n in enumerate(names)}
    g = ig.Graph(n=len(names), edges=[(index[a], index[b]) for a, b in graph.edges])
    g.es["weight"] = [d["weight"] for _, _, d in graph.edges(data=True)]
    return g, names


def centrality_table(graph: nx.Graph) -> pd.DataFrame:
    """Degree, weighted degree, betweenness, PageRank and eigenvector centrality per node.

    * degree = number of distinct co-occurring entities; weighted degree = total shared incidents;
    * betweenness: exact, weighted (igraph; a link's length is 1 / weight, so strongly tied entities are close);
    * PageRank (damping 0.85, weights = shared incidents);
    * eigenvector centrality is computed on the largest connected component (0 elsewhere).
    """
    ig_graph, names = _to_igraph(graph)
    lengths = [1.0 / w for w in ig_graph.es["weight"]]
    betweenness = ig_graph.betweenness(weights=lengths)
    pagerank = nx.pagerank(graph, weight="weight")
    component = max(nx.connected_components(graph), key=len)
    eig = nx.eigenvector_centrality_numpy(graph.subgraph(component), weight="weight")
    rows = []
    for i, node in enumerate(names):
        attrs = graph.nodes[node]
        rows.append(
            {
                "node": node,
                "label": attrs["label"],
                "type": attrs["type"],
                "n_incidents": attrs["n_incidents"],
                "degree": graph.degree(node),
                "weighted_degree": graph.degree(node, weight="weight"),
                "betweenness": betweenness[i],
                "pagerank": pagerank[node],
                "eigenvector": float(eig.get(node, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def kendall_matrix(
    table: pd.DataFrame, measures: tuple[str, ...] = CENTRALITY + ("n_incidents",)
) -> pd.DataFrame:
    """Pairwise Kendall's tau between centrality measures (and plain incident counts) over the given nodes."""
    out = pd.DataFrame(index=list(measures), columns=list(measures), dtype=float)
    for a, b in itertools.product(measures, repeat=2):
        out.loc[a, b] = 1.0 if a == b else float(kendalltau(table[a], table[b])[0])
    return out


def top_by_type(table: pd.DataFrame, measure: str, node_type: str, n: int = 10) -> pd.DataFrame:
    """Top `n` nodes of one type by one centrality measure."""
    sub = table[table["type"] == node_type].sort_values(measure, ascending=False).head(n)
    return sub[["label", "n_incidents", *CENTRALITY]].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Communities
# --------------------------------------------------------------------------------------


def detect_communities(
    graph: nx.Graph, method: str = "louvain", seed: int | None = None
) -> dict[str, int]:
    """Community id per node with 'louvain' (networkx), 'leiden' (leidenalg, modularity) or 'label_propagation'."""
    if method == "louvain":
        parts = nxc.louvain_communities(graph, weight="weight", seed=seed)
    elif method == "leiden":
        ig_graph, names = _to_igraph(graph)
        partition = leidenalg.find_partition(
            ig_graph, leidenalg.ModularityVertexPartition, weights="weight", seed=seed
        )
        parts = [{names[i] for i in cluster} for cluster in partition]
    elif method == "label_propagation":
        parts = list(nxc.asyn_lpa_communities(graph, weight="weight", seed=seed))
    else:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    ordered = sorted(parts, key=lambda p: (-len(p), sorted(p)[0]))
    return {node: i for i, part in enumerate(ordered) for node in part}


def modularity(graph: nx.Graph, assignment: dict[str, int]) -> float:
    """Weighted modularity of an assignment (computed identically for every algorithm)."""
    groups: dict[int, set[str]] = defaultdict(set)
    for node, comm in assignment.items():
        groups[comm].add(node)
    return float(nxc.modularity(graph, list(groups.values()), weight="weight"))


def compare_community_methods(
    graph: nx.Graph, seeds: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], dict[str, int]]]:
    """Run every method with every seed; returns (per-run table, per-method summary, assignments).

    Summary: mean/sd modularity, mean number of communities, share of nodes in the largest community, and
    stability = mean pairwise adjusted Rand index between the partitions of different seeds.
    """
    assignments: dict[tuple[str, int], dict[str, int]] = {}
    rows = []
    nodes = list(graph.nodes)
    for method in METHODS:
        for seed in seeds:
            part = detect_communities(graph, method, seed)
            assignments[(method, seed)] = part
            sizes = Counter(part.values())
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "modularity": modularity(graph, part),
                    "n_communities": len(sizes),
                    "largest_share": max(sizes.values()) / len(nodes),
                    "singletons": sum(1 for v in sizes.values() if v == 1),
                }
            )
    runs = pd.DataFrame(rows)
    summary = []
    for method in METHODS:
        sub = runs[runs["method"] == method]
        labels = [[assignments[(method, s)][n] for n in nodes] for s in seeds]
        aris = [adjusted_rand_score(a, b) for a, b in itertools.combinations(labels, 2)]
        summary.append(
            {
                "method": method,
                "modularity_mean": sub["modularity"].mean(),
                "modularity_sd": sub["modularity"].std(ddof=1),
                "n_communities_mean": sub["n_communities"].mean(),
                "largest_share_mean": sub["largest_share"].mean(),
                "singletons_mean": sub["singletons"].mean(),
                "stability_ari_mean": float(np.mean(aris)),
                "stability_ari_min": float(np.min(aris)),
            }
        )
    return runs, pd.DataFrame(summary), assignments


def choose_method(summary: pd.DataFrame, min_stability: float = 0.5) -> tuple[str, str]:
    """Pre-specified rule: among methods with mean stability (ARI) >= `min_stability`, the highest mean
    modularity; if none is stable enough, the most stable one. Returns (method, reason)."""
    stable = summary[summary["stability_ari_mean"] >= min_stability]
    if len(stable):
        best = stable.sort_values("modularity_mean", ascending=False).iloc[0]
        return str(
            best["method"]
        ), f"highest mean modularity among methods with mean ARI >= {min_stability}"
    best = summary.sort_values("stability_ari_mean", ascending=False).iloc[0]
    return str(best["method"]), f"no method reached mean ARI {min_stability}; most stable chosen"


def assign_incidents(links: pd.DataFrame, assignment: dict[str, int]) -> pd.Series:
    """Community of each incident = community holding most of its (clustered) entities (ties: lowest id)."""
    sub = links[links["entity_id"].isin(assignment)].copy()
    sub["community"] = sub["entity_id"].map(assignment)
    counts = sub.groupby(["incident_id", "community"]).size().rename("n").reset_index()
    best = counts.sort_values(
        ["incident_id", "n", "community"], ascending=[True, False, True]
    ).drop_duplicates("incident_id")
    return best.set_index("incident_id")["community"].rename("community")


MIN_INCIDENTS_FOR_DOMAIN_MIX = 20


def profile_communities(
    graph: nx.Graph,
    assignment: dict[str, int],
    nd: NetworkData,
    top: int = 4,
    min_incidents_for_mix: int = MIN_INCIDENTS_FOR_DOMAIN_MIX,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """One row per community: size, top organisations / AI systems / harmed groups, and the risk-domain and
    sector mix of the incidents assigned to it. Also returns association scores (NMI) with MIT domain.

    Small-cell rule: for communities with fewer than `min_incidents_for_mix` incidents the domain mix is not
    shown, because it would reveal the labels of individual incidents."""
    incident_comm = assign_incidents(nd.links, assignment)
    inc = nd.incidents.set_index("incident_id").join(incident_comm, how="left")
    rows = []
    sizes = Counter(assignment.values())
    for comm, size in sorted(sizes.items(), key=lambda kv: -kv[1]):
        members = [n for n, c in assignment.items() if c == comm]
        by_type: dict[str, list[str]] = defaultdict(list)
        for n in sorted(
            members, key=lambda n: (-graph.nodes[n]["n_incidents"], graph.nodes[n]["label"], n)
        ):
            by_type[graph.nodes[n]["type"]].append(graph.nodes[n]["label"])
        mine = inc[inc["community"] == comm]
        domains = mine["domain"].dropna().value_counts()
        if len(mine) < min_incidents_for_mix:
            domains = domains.iloc[:0]
        sectors = pd.Series([s for lst in mine["sectors"] for s in lst]).value_counts()
        rows.append(
            {
                "community": comm,
                "entities": size,
                "incidents": len(mine),
                "organisations": "; ".join(by_type["organisation"][:top]),
                "ai_systems": "; ".join(by_type["ai_system"][:top]),
                "harmed_groups": "; ".join(by_type["harmed_group"][:top]),
                "top_domain": domains.index[0] if len(domains) else "",
                "top_domain_share": float(domains.iloc[0] / domains.sum())
                if len(domains)
                else np.nan,
                "domains": "; ".join(
                    f"{d} ({v / domains.sum():.0%})" for d, v in domains.head(3).items()
                )
                if len(mine) >= min_incidents_for_mix
                else f"(not shown: fewer than {min_incidents_for_mix} incidents)",
                "sectors_n": int(sectors.sum()) if len(sectors) else 0,
                "sectors": "; ".join(f"{s} ({v})" for s, v in sectors.head(3).items()),
            }
        )
    labelled = inc[inc["domain"].notna() & inc["community"].notna()]
    assoc = {
        "nmi_community_vs_domain": float(
            normalized_mutual_info_score(labelled["domain"], labelled["community"])
        )
        if len(labelled)
        else float("nan"),
        "incidents_assigned": int(inc["community"].notna().sum()),
        "incidents_without_clustered_entity": int(inc["community"].isna().sum()),
    }
    return pd.DataFrame(rows), assoc


# --------------------------------------------------------------------------------------
# Temporal evolution
# --------------------------------------------------------------------------------------


def cumulative_snapshots(nd: NetworkData, years: list[int]) -> pd.DataFrame:
    """Growth of the full heterogeneous graph (all entities, min_incidents = 1) at the end of each year."""
    rows = []
    seen_orgs: set[str] = set()
    for year in years:
        graph = build_graph(nd, year_max=year)
        summary = graph_summary(graph)
        orgs = {d["entity_id"] for _, d in graph.nodes(data=True) if d["type"] == "organisation"}
        rows.append(
            {
                "year": year,
                "nodes": summary["nodes"],
                "edges": summary["edges"],
                **{f"nodes_{k}": v for k, v in summary["nodes_by_type"].items()},
                "new_organisations": len(orgs - seen_orgs),
                "organisations": len(orgs),
            }
        )
        seen_orgs |= orgs
    out = pd.DataFrame(rows).fillna(0)
    for col in out.columns:
        if col != "year":
            out[col] = out[col].astype(int)
    return out


def top_organisations_over_time(nd: NetworkData, years: list[int], n: int = 10) -> pd.DataFrame:
    """Top `n` organisations by PageRank in the cumulative entity network of each year (long table)."""
    rows = []
    for year in years:
        graph = entity_network(nd, min_incidents=1, year_max=year)
        if graph.number_of_edges() == 0:
            continue
        pr = nx.pagerank(graph, weight="weight")
        orgs = sorted(
            ((v, k) for k, v in pr.items() if graph.nodes[k]["type"] == "organisation"),
            reverse=True,
        )[:n]
        rows += [
            {
                "year": year,
                "rank": r + 1,
                "entity_id": k,
                "organisation": graph.nodes[k]["label"],
                "pagerank": v,
                "n_incidents": graph.nodes[k]["n_incidents"],
            }
            for r, (v, k) in enumerate(orgs)
        ]
    return pd.DataFrame(rows)


def top_set_overlap(top: pd.DataFrame) -> pd.DataFrame:
    """Jaccard overlap of the top-n organisation sets of consecutive years."""
    years = sorted(top["year"].unique())
    sets = {y: set(top.loc[top["year"] == y, "entity_id"]) for y in years}
    return pd.DataFrame(
        [
            {
                "year": b,
                "previous": a,
                "jaccard_with_previous": len(sets[a] & sets[b]) / len(sets[a] | sets[b]),
                "entered": len(sets[b] - sets[a]),
            }
            for a, b in zip(years[:-1], years[1:], strict=True)
        ]
    )


# --------------------------------------------------------------------------------------
# Dashboard export
# --------------------------------------------------------------------------------------


def export_graph_json(
    graph: nx.Graph,
    assignment: dict[str, int],
    centrality: pd.DataFrame,
    nd: NetworkData,
    path: Path,
    top_n: int = 150,
    max_edges: int = 900,
    snapshot: str = "",
) -> dict[str, Any]:
    """Write a dashboard-ready JSON of the top `top_n` nodes by weighted degree.

    Nodes carry type, label, community, incident count, first/last year, centrality and a small profile
    (risk-domain mix, up to 5 recent incidents). Edges carry weight and per-year counts so a year slider can
    show the cumulative network (sum `by_year` up to the chosen year). Capped at `max_edges` strongest edges.
    """
    cent = centrality.set_index("node")
    keep = cent.sort_values("weighted_degree", ascending=False).head(top_n).index
    sub = graph.subgraph(keep)
    # Canonical order (endpoints sorted, ties broken by name): networkx subgraphs iterate in an order that
    # depends on Python's per-process string hashing, which made the export (and the max_edges cut) unstable.
    canonical = [(min(a, b), max(a, b), d) for a, b, d in sub.edges(data=True)]
    edges = sorted(canonical, key=lambda e: (-e[2]["weight"], e[0], e[1]))[:max_edges]
    linked = {n for a, b, _ in edges for n in (a, b)}
    inc = nd.incidents.set_index("incident_id")
    by_entity = nd.links.groupby("entity_id")["incident_id"].apply(lambda s: sorted(set(s)))
    nodes = []
    for n in keep:
        if n not in linked:
            continue
        attrs, c = graph.nodes[n], cent.loc[n]
        ids = by_entity.get(n, [])
        domains = inc.loc[ids, "domain"].dropna().value_counts()
        recent = inc.loc[ids].sort_values("year", ascending=False).head(5)
        nodes.append(
            {
                "id": n,
                "label": attrs["label"],
                "type": attrs["type"],
                "community": int(assignment[n]),
                "n_incidents": attrs["n_incidents"],
                "first_year": attrs["first_year"],
                "last_year": attrs["last_year"],
                "weighted_degree": int(c["weighted_degree"]),
                "pagerank": round(float(c["pagerank"]), 6),
                "betweenness": round(float(c["betweenness"]), 2),
                "domains": {k: int(v) for k, v in domains.head(5).items()},
                "recent_incidents": [
                    {"id": int(i), "title": str(r.title), "year": int(r.year)}
                    for i, r in recent.iterrows()
                ],
            }
        )
    payload = {
        "snapshot": snapshot,
        "note": "Entities are alleged deployers, developers, harmed parties or implicated systems as recorded by AIID; "
        "centrality reflects frequency of involvement in reported incidents, not responsibility.",
        "caps": {"top_n": top_n, "max_edges": max_edges},
        "nodes": nodes,
        "edges": [
            {
                "source": a,
                "target": b,
                "weight": d["weight"],
                "by_year": {str(y): c for y, c in sorted(d["by_year"].items())},
            }
            for a, b, d in edges
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return {"nodes": len(nodes), "edges": len(edges), "bytes": path.stat().st_size}
