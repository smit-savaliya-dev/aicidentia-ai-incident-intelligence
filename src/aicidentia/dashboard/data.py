"""Data the dashboard shows, read from the saved pipeline outputs (never recomputed here).

Blind label audit: while `data/manual/label_audit_sample.csv` is not fully annotated, the MIT label and any
model prediction for the 150 audit incidents are replaced by HIDDEN, and per-entity domain mixes in the network
panel are withheld, so the dashboard cannot reveal those labels.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from aicidentia import registry
from aicidentia.config import PROJECT_ROOT, load_config
from aicidentia.data import clean
from aicidentia.models import label_audit

logger = logging.getLogger(__name__)
TABLES = PROJECT_ROOT / "reports" / "tables"
FIGURES = PROJECT_ROOT / "reports" / "figures"
NETWORK_JSON = PROJECT_ROOT / "reports" / "network_graph.json"
HIDDEN = "hidden (blind audit in progress)"


def results() -> dict[str, Any]:
    """Every registered number, flattened to {key: value}."""
    return {k: v["value"] for k, v in registry.load_results().items()}


def fact(key: str, default: Any = None) -> Any:
    try:
        return registry.get(key)
    except KeyError:
        return default


def blind_ids() -> set[int]:
    """Incident ids whose MIT label must stay hidden right now (empty once the audit is fully annotated)."""
    return label_audit.audit_incident_ids() if label_audit.blind_audit_active() else set()


def table(name: str) -> pd.DataFrame:
    """A saved results table from reports/tables/<name>.csv."""
    return pd.read_csv(TABLES / f"{name}.csv")


def theme_names() -> tuple[pd.DataFrame, dict[str, Any]]:
    """incident_id -> theme (your labels, or PROVISIONAL placeholders) and the label info."""
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


def incidents(hide: set[int] | None = None) -> pd.DataFrame:
    """Incident table for browsing: id, date, year, title, description, reports, theme, MIT domain (masked)."""
    inc = clean.load_processed("incidents_clean")[
        [
            "incident_id",
            "date",
            "year",
            "title",
            "description",
            "report_count_clean",
            "mit_risk_domain",
        ]
    ]
    themes, _ = theme_names()
    out = inc.merge(themes[["incident_id", "theme"]], on="incident_id", how="left")
    hide = blind_ids() if hide is None else hide
    out["mit_domain"] = out["mit_risk_domain"].fillna("(no MIT label)")
    out.loc[out["incident_id"].isin(hide), "mit_domain"] = HIDDEN
    return out.drop(columns=["mit_risk_domain"]).rename(columns={"report_count_clean": "reports"})


def yearly_by_year() -> pd.DataFrame:
    """Incidents per year with complete / partial flag."""
    inc = clean.load_processed("incidents_clean")
    partial = int(fact("audit.partial_year", inc["year"].max()))
    counts = inc.groupby("year").size().rename("incidents").reset_index()
    counts["partial"] = counts["year"] == partial
    return counts


def domain_counts() -> pd.Series:
    """MIT domain distribution (aggregate counts only)."""
    inc = clean.load_processed("incidents_clean")
    return inc["mit_risk_domain"].value_counts()


def network() -> dict[str, Any]:
    """The dashboard network export with a fixed 2-D layout added to every node."""
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


def node_profile(node: dict[str, Any], hide: set[int] | None = None) -> dict[str, Any]:
    """What the side panel shows for a node; the domain mix is withheld while the blind audit is active."""
    hide = blind_ids() if hide is None else hide
    recent = [r for r in node.get("recent_incidents", [])]
    return {
        "label": node["label"],
        "type": node["type"],
        "community": node["community"],
        "incidents": node["n_incidents"],
        "years": f"{node['first_year']}-{node['last_year']}",
        "pagerank": node["pagerank"],
        "domains": None if hide else node.get("domains", {}),
        "recent": recent,
    }


def snapshot_info() -> dict[str, Any]:
    cfg = load_config()
    return {
        "snapshot": str(cfg["snapshot_date"]),
        "seed": int(cfg["random_seed"]),
        "name": cfg["dashboard_name"],
        "tagline": cfg["tagline"],
        "title": cfg["full_title"],
    }
