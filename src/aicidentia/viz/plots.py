"""Reusable plotting helpers for notebooks and the Streamlit app (Plotly/Matplotlib)."""

from __future__ import annotations

import pandas as pd


def plot_incidents_per_year(df: pd.DataFrame):
    """Line/bar chart of reported incidents per year."""
    raise NotImplementedError


def plot_topic_trends(topic_year: pd.DataFrame):
    """Topic share over time, one line per topic."""
    raise NotImplementedError


def plot_topic_words(topic_words: dict[int, list[str]]):
    """Bar charts of the top words per topic."""
    raise NotImplementedError


def plot_network(graph, communities: dict | None = None):
    """Interactive network figure coloured by community."""
    raise NotImplementedError


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """Confusion-matrix heatmap."""
    raise NotImplementedError


def plot_model_comparison(results: pd.DataFrame, metric: str):
    """Grouped bar chart comparing models on a metric."""
    raise NotImplementedError
