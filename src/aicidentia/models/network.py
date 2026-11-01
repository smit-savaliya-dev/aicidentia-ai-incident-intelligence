"""RQ2 - network of incidents, organisations, harmed groups and risk domains."""

from __future__ import annotations

import networkx as nx
import pandas as pd


def build_graph(incidents: pd.DataFrame, entities: pd.DataFrame) -> nx.Graph:
    """Build the multi-type graph (incidents, organisations, harmed groups, risk domains)."""
    raise NotImplementedError


def detect_communities(graph: nx.Graph, method: str = "louvain", seed: int | None = None) -> dict:
    """Detect communities with 'louvain', 'leiden' or 'label_propagation'."""
    raise NotImplementedError


def compare_community_methods(graph: nx.Graph, seed: int | None = None) -> pd.DataFrame:
    """Compare the three algorithms (modularity, number/size of communities, stability)."""
    raise NotImplementedError


def centrality_table(graph: nx.Graph) -> pd.DataFrame:
    """Degree, betweenness, closeness and eigenvector centrality per node."""
    raise NotImplementedError


def export_pyvis(graph: nx.Graph, path: str) -> None:
    """Write an interactive pyvis HTML visualisation of the graph."""
    raise NotImplementedError
