"""Data the dashboard shows, read from the saved pipeline outputs (never recomputed here).

Everything is a pure function of files in reports/ and data/processed/, so pages load from precomputed
artefacts and work offline. Filters (year range, MIT risk domain, CSETv1 sector) are applied with
`filter_incidents` on every page that shows incidents.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aicidentia import registry
from aicidentia.config import PROJECT_ROOT, load_config
from aicidentia.data import clean

logger = logging.getLogger(__name__)
TABLES = PROJECT_ROOT / "reports" / "tables"
FIGURES = PROJECT_ROOT / "reports" / "figures"
NETWORK_JSON = PROJECT_ROOT / "reports" / "network_graph.json"
AIID_CITE = "https://incidentdatabase.ai/cite/{}"
NO_LABEL = "(no MIT label)"
NO_SECTOR = "(no CSET sector)"


def results() -> dict[str, Any]:
    """Every registered number, flattened to {key: value}."""
    return {k: v["value"] for k, v in registry.load_results().items()}


def fact(key: str, default: Any = None) -> Any:
    try:
        return registry.get(key)
    except KeyError:
        return default


def table(name: str) -> pd.DataFrame:
    """A saved results table from reports/tables/<name>.csv."""
    return pd.read_csv(TABLES / f"{name}.csv")


def cite_url(incident_id: int) -> str:
    return AIID_CITE.format(int(incident_id))


def snapshot_info() -> dict[str, Any]:
    cfg = load_config()
    snap = str(cfg["snapshot_date"])
    return {
        "snapshot": snap,
        "snapshot_iso": f"{snap[:4]}-{snap[4:6]}-{snap[6:]}",
        "seed": int(cfg["random_seed"]),
        "name": cfg["dashboard_name"],
        "tagline": cfg["tagline"],
        "title": cfg["full_title"],
    }


# --------------------------------------------------------------------------------------
# Incidents and filters
# --------------------------------------------------------------------------------------


def theme_names() -> tuple[pd.DataFrame, dict[str, Any]]:
    """incident_id -> theme (confirmed labels, or placeholder names) and the label info."""
    from aicidentia.models.themes import StaleLabelsError, load_theme_frame

    try:
        tf = load_theme_frame()
        return tf.data[["incident_id", "topic_id", "theme"]], tf.info
    except (StaleLabelsError, FileNotFoundError) as exc:
        topics = clean.load_processed("incident_topics")
        out = topics[["incident_id", "topic_id"]].assign(
            theme="T" + topics["topic_id"].astype(str) + ": " + topics["suggested_label"]
        )
        return out, {"provisional": True, "problem": str(exc)}


def _as_list(value: Any) -> list[str]:
    if value is None or (not isinstance(value, list | np.ndarray) and pd.isna(value)):
        return []
    return [str(v) for v in value]


def incidents() -> pd.DataFrame:
    """One row per incident with everything the pages filter and show."""
    inc = clean.load_processed("incidents_clean")
    themes, _ = theme_names()
    out = inc.merge(themes[["incident_id", "theme"]], on="incident_id", how="left")
    out["mit_domain"] = out["mit_risk_domain"].fillna(NO_LABEL)
    out["mit_subdomain"] = out["mit_risk_subdomain"].fillna("")
    out["sectors"] = out["cset_sector_of_deployment"].map(_as_list)
    out["failures"] = out["gmf_technical_failure"].map(_as_list)
    out["url"] = out["incident_id"].map(cite_url)
    keep = [
        "incident_id",
        "date",
        "year",
        "title",
        "description",
        "report_count_clean",
        "theme",
        "mit_domain",
        "mit_subdomain",
        "sectors",
        "failures",
        "has_cset",
        "cset_tangible_harm",
        "url",
    ]
    return out[keep].rename(columns={"report_count_clean": "reports"})


@dataclass(frozen=True)
class Filters:
    years: tuple[int, int]
    domains: tuple[str, ...] = ()
    sectors: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return bool(self.domains or self.sectors)

    def describe(self) -> str:
        parts = [f"{self.years[0]}-{self.years[1]}"]
        if self.domains:
            parts.append(f"{len(self.domains)} risk domain(s)")
        if self.sectors:
            parts.append(f"{len(self.sectors)} sector(s), CSETv1-annotated incidents only")
        return "; ".join(parts)


def filter_incidents(df: pd.DataFrame, flt: Filters) -> pd.DataFrame:
    """Year range (inclusive), MIT domains (any of), CSETv1 sectors (any of; incidents without CSETv1 sector
    annotation are dropped when a sector is chosen)."""
    out = df[(df["year"] >= flt.years[0]) & (df["year"] <= flt.years[1])]
    if flt.domains:
        out = out[out["mit_domain"].isin(flt.domains)]
    if flt.sectors:
        wanted = set(flt.sectors)
        out = out[out["sectors"].map(lambda s: bool(wanted & set(s)))]
    return out


def filter_options(df: pd.DataFrame) -> dict[str, Any]:
    domains = sorted(d for d in df["mit_domain"].unique() if d != NO_LABEL)
    sectors = sorted({s for row in df["sectors"] for s in row if s not in ("unclear", "other")})
    return {
        "years": (int(df["year"].min()), int(df["year"].max())),
        "default_years": (2015, int(df["year"].max())),
        "domains": domains,
        "sectors": sectors,
        "sector_coverage": int(df["sectors"].map(bool).sum()),
    }


def yearly_by_year(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Incidents per year with complete / partial flag."""
    df = incidents() if df is None else df
    partial = int(fact("audit.partial_year", df["year"].max()))
    counts = df.groupby("year").size().rename("incidents").reset_index()
    counts["partial"] = counts["year"] == partial
    return counts


def theme_year_shares(df: pd.DataFrame) -> pd.DataFrame:
    """group, year, n, share (of the year's incidents in `df`), partial_year."""
    partial = int(fact("audit.partial_year", df["year"].max()))
    counts = df.groupby(["year", "theme"]).size().rename("n").reset_index()
    totals = df.groupby("year").size()
    counts["share"] = counts["n"] / counts["year"].map(totals)
    counts["partial_year"] = counts["year"] == partial
    return counts.rename(columns={"theme": "group"})


def domain_counts(df: pd.DataFrame | None = None) -> pd.Series:
    df = incidents() if df is None else df
    return df.loc[df["mit_domain"] != NO_LABEL, "mit_domain"].value_counts()


def emerging_table() -> pd.DataFrame:
    """Share-based trend tests per theme (2015-2025): slope in percentage points per year, q-value, badge."""
    t = table("rq1b_trends_themes")
    t = t[t["basis"] == "share"].copy()
    t["slope_pp_per_year"] = t["sen_slope"] * 100
    t["recent_share"] = t["recent_mean"] * 100
    t["status"] = np.where(
        t["emerging"], "EMERGING", np.where(t["direction"] == "decreasing", "declining", "")
    )
    cols = ["group", "status", "slope_pp_per_year", "p_bh", "recent_share", "n"]
    return (
        t[cols]
        .rename(columns={"group": "theme", "p_bh": "q_value", "n": "years"})
        .sort_values("slope_pp_per_year", ascending=False)
        .reset_index(drop=True)
    )


def representative_incidents(df: pd.DataFrame, theme: str, n: int = 5) -> pd.DataFrame:
    """The topic's representative incidents (highest topic weight) that pass the filters."""
    labels = pd.read_csv(PROJECT_ROOT / "data" / "manual" / "topic_labels.csv", dtype=str)
    labels["name"] = (
        labels["final_label"]
        .fillna("")
        .where(labels["final_label"].fillna("") != "", labels["suggested_label"])
    )
    row = labels[labels["name"] == theme]
    ids = (
        [int(i) for i in row["representative_incident_ids"].iloc[0].split(";")] if len(row) else []
    )
    topics = clean.load_processed("incident_topics")
    best = topics[topics["incident_id"].isin(df["incident_id"])].merge(
        df[["incident_id", "theme"]], on="incident_id"
    )
    best = best[best["theme"] == theme].sort_values("weight", ascending=False)
    ordered = list(
        dict.fromkeys([i for i in ids if i in set(best["incident_id"])] + list(best["incident_id"]))
    )
    return df.set_index("incident_id").loc[ordered[:n]].reset_index()


# --------------------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------------------

ENTITY_TYPE_NAME = {
    "organisation": "Organisation (alleged)",
    "technology_system": "AI system / product",
    "population_group": "Harmed group",
    "generic_actor_group": "Generic actor",
    "abstract_value": "Harm / value",
    "unknown": "Other",
}


def entity_index() -> pd.DataFrame:
    """Searchable entities (named persons are already excluded in Phase 2) with community and PageRank."""
    ent = clean.load_processed("entity_table")[
        ["entity_id", "entity", "entity_type", "n_incidents"]
    ]
    comm = clean.load_processed("entity_communities")[["node", "community", "pagerank"]]
    out = ent.merge(comm, left_on="entity_id", right_on="node", how="left").drop(columns="node")
    out["type_name"] = out["entity_type"].map(ENTITY_TYPE_NAME).fillna("Other")
    return out.sort_values(["n_incidents", "entity"], ascending=[False, True]).reset_index(
        drop=True
    )


def incident_entities() -> pd.DataFrame:
    return clean.load_processed("incident_entities")[["incident_id", "role", "entity_id", "entity"]]


def ego_network(
    entity_id: str,
    links: pd.DataFrame,
    entities: pd.DataFrame,
    incident_ids: set[int],
    top_n: int = 25,
    seed: int = 42,
) -> dict[str, Any]:
    """A node plus its `top_n` most frequent co-occurring neighbours within `incident_ids`, edges weighted by
    shared incidents (among all shown nodes), and a fixed radial layout."""
    links = links[links["incident_id"].isin(incident_ids)][["incident_id", "entity_id"]]
    links = links.drop_duplicates()
    own = set(links.loc[links["entity_id"] == entity_id, "incident_id"])
    co = links[links["incident_id"].isin(own) & (links["entity_id"] != entity_id)]
    weights = co.groupby("entity_id")["incident_id"].nunique().sort_values(ascending=False)
    weights = weights.reindex(sorted(weights.index, key=lambda e: (-weights[e], e)))
    keep = [entity_id, *weights.index[:top_n]]
    sub = links[links["entity_id"].isin(keep) & links["incident_id"].isin(own)]
    pairs = sub.merge(sub, on="incident_id")
    pairs = pairs[pairs["entity_id_x"] < pairs["entity_id_y"]]
    edges = pairs.groupby(["entity_id_x", "entity_id_y"])["incident_id"].nunique().reset_index()
    edges.columns = ["source", "target", "weight"]
    # Radial layout: the entity in the centre, neighbours on a ring grouped by community, then by shared incidents
    info = entities.set_index("entity_id")

    def community_of(e: str) -> float:
        c = info["community"].get(e, np.nan) if e in info.index else np.nan
        return 99.0 if pd.isna(c) else float(c)

    ring = sorted(keep[1:], key=lambda e: (community_of(e), -int(weights.get(e, 0)), e))
    pos = {entity_id: (0.0, 0.0)}
    for i, e in enumerate(ring):
        angle = 2 * np.pi * i / max(len(ring), 1) + np.pi / 2
        pos[e] = (float(np.cos(angle)), float(np.sin(angle)))
    nodes = []
    for e in keep:
        row = info.loc[e] if e in info.index else None
        nodes.append(
            {
                "id": e,
                "label": str(row["entity"]) if row is not None else e,
                "type": str(row["entity_type"]) if row is not None else "unknown",
                "type_name": str(row["type_name"]) if row is not None else "Other",
                "community": None
                if row is None or pd.isna(row["community"])
                else int(row["community"]),
                "shared": int(len(own)) if e == entity_id else int(weights.get(e, 0)),
                "centre": e == entity_id,
                "x": float(pos[e][0]),
                "y": float(pos[e][1]),
            }
        )
    return {"nodes": nodes, "edges": edges.to_dict(orient="records"), "incidents": len(own)}


def entity_profile(entity_id: str, links: pd.DataFrame, df: pd.DataFrame) -> dict[str, Any]:
    """Risk domains, sectors, incident count and linked incidents of an entity within the filtered incidents."""
    ids = set(links.loc[links["entity_id"] == entity_id, "incident_id"]) & set(df["incident_id"])
    sub = df[df["incident_id"].isin(ids)].sort_values("date", ascending=False)
    roles = links[(links["entity_id"] == entity_id) & links["incident_id"].isin(ids)]["role"]
    return {
        "incidents": len(sub),
        "years": f"{int(sub['year'].min())}-{int(sub['year'].max())}" if len(sub) else "",
        "roles": roles.value_counts().to_dict(),
        "domains": sub.loc[sub["mit_domain"] != NO_LABEL, "mit_domain"].value_counts(),
        "sectors": sub["sectors"].explode().dropna().value_counts(),
        "sector_coverage": int(sub["sectors"].map(bool).sum()),
        "linked": sub[["incident_id", "date", "title", "mit_domain", "url"]],
    }


def network() -> dict[str, Any]:
    """The Phase 5 dashboard export with a fixed 2-D layout (kept for the method page and tests)."""
    import networkx as nx

    data = json.loads(NETWORK_JSON.read_text(encoding="utf-8"))
    graph = nx.Graph()
    graph.add_nodes_from(n["id"] for n in data["nodes"])
    graph.add_weighted_edges_from((e["source"], e["target"], e["weight"]) for e in data["edges"])
    pos = nx.spring_layout(
        graph,
        weight="weight",
        seed=int(load_config()["random_seed"]),
        iterations=200,
        k=1.6 / max(len(graph), 1) ** 0.5,
    )
    for node in data["nodes"]:
        node["x"], node["y"] = (float(v) for v in pos[node["id"]])
    return data


def node_profile(node: dict[str, Any]) -> dict[str, Any]:
    """What a panel shows for a node of the Phase 5 export."""
    return {
        "label": node["label"],
        "type": node["type"],
        "community": node["community"],
        "incidents": node["n_incidents"],
        "years": f"{node['first_year']}-{node['last_year']}",
        "pagerank": node["pagerank"],
        "domains": node.get("domains", {}),
        "recent": list(node.get("recent_incidents", [])),
    }


# --------------------------------------------------------------------------------------
# Precedent finder
# --------------------------------------------------------------------------------------


def demo_cases(df: pd.DataFrame) -> pd.DataFrame:
    """The three preloaded real incidents (most recent months of the snapshot; see rq4.demo_cases)."""
    ids = [int(i) for i in fact("rq4.demo_cases", []) or []]
    return (
        df.set_index("incident_id")
        .loc[[i for i in ids if i in set(df["incident_id"])]]
        .reset_index()
    )


def precedent_details(hits: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Search hits joined with date, description, MIT domain/subdomain, GMF failures and link."""
    cols = [
        "incident_id",
        "date",
        "title",
        "description",
        "mit_domain",
        "mit_subdomain",
        "failures",
        "url",
    ]
    return hits[["rank", "incident_id", "score"]].merge(df[cols], on="incident_id", how="left")
