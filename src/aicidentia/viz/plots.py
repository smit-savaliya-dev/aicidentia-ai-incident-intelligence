"""Reusable plotting helpers for notebooks and the Streamlit app.

All figures use the IEEE-sized, colour-blind-safe style from `aicidentia.viz.style`.
Functions return the matplotlib Figure; call `style.save_figure` to write PNG + SVG.
"""

from __future__ import annotations

import textwrap
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aicidentia.viz.style import DOUBLE_COLUMN_IN, PALETTE, SINGLE_COLUMN_IN, apply_style

# --------------------------------------------------------------------------------------
# Phase 1: data-audit figures
# --------------------------------------------------------------------------------------


def plot_incidents_per_year(
    per_year: pd.DataFrame,
    trend_window: tuple[int, int] | None = None,
    first_year: int = 2010,
) -> plt.Figure:
    """Bar chart of incidents per year; the partial snapshot year is hatched.

    Years before `first_year` are summarised in the title (they hold a handful of incidents).
    `trend_window` shades the recommended range for trend tests.
    """
    apply_style()
    shown = per_year[per_year["year"] >= first_year]
    earlier = int(per_year.loc[per_year["year"] < first_year, "incidents"].sum())
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 2.5))
    if trend_window:
        ax.axvspan(
            trend_window[0] - 0.5,
            trend_window[1] + 0.5,
            color=PALETTE["sky"],
            alpha=0.15,
            lw=0,
            label=f"Trend-test window {trend_window[0]}-{trend_window[1]}",
        )
    complete = shown["complete"].to_numpy()
    bars = ax.bar(
        shown["year"], shown["incidents"], color=np.where(complete, PALETTE["blue"], PALETTE["sky"])
    )
    for bar, is_complete in zip(bars, complete, strict=True):
        if not is_complete:
            bar.set_hatch("//")
            bar.set_edgecolor(PALETTE["blue"])
    for x, y in zip(shown["year"], shown["incidents"], strict=True):
        ax.text(x, y + 6, str(y), ha="center", va="bottom", fontsize=6)
    ax.set_xlabel("Year the harm occurred")
    ax.set_ylabel("Incidents (count)")
    ax.set_xticks(shown["year"])
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(
        f"Reported incidents per year ({earlier} incidents dated before {first_year} not shown; "
        "hatched = partial year)"
    )
    if trend_window:
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_taxonomy_coverage(
    counts: dict[str, int], total: int, mit_by_year: pd.DataFrame, first_year: int = 2015
) -> plt.Figure:
    """(a) Share of incidents carrying each taxonomy; (b) MIT label coverage by year."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(DOUBLE_COLUMN_IN, 2.5), gridspec_kw={"width_ratios": [1, 1.4]}
    )
    names = list(counts)
    shares = [counts[n] / total * 100 for n in names]
    colours = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["grey"]][: len(names)]
    ax1.barh(names[::-1], shares[::-1], color=colours[::-1])
    for i, (n, share) in enumerate(zip(names[::-1], shares[::-1], strict=True)):
        ax1.text(share + 1, i, f"{share:.1f}% (n={counts[n]})", va="center", fontsize=6)
    ax1.set_xlim(0, 115)
    ax1.set_xlabel("Incidents covered (%)")
    ax1.set_title("(a) Taxonomy coverage")

    sub = mit_by_year[mit_by_year["year"] >= first_year]
    ax2.plot(sub["year"], sub["coverage"] * 100, marker="o", color=PALETTE["blue"], ms=3)
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("Year the harm occurred")
    ax2.set_ylabel("MIT-labelled incidents (%)")
    ax2.set_title("(b) MIT label coverage by year")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_class_balance(distribution: pd.DataFrame, title: str) -> plt.Figure:
    """Horizontal bar chart of class counts with shares (class, n, share columns)."""
    apply_style()
    dist = distribution.sort_values("n")
    labels = ["\n".join(textwrap.wrap(c, 30)) for c in dist["class"]]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 2.9))
    ax.barh(labels, dist["n"], color=PALETTE["blue"])
    for i, (n, share) in enumerate(zip(dist["n"], dist["share"], strict=True)):
        ax.text(n + dist["n"].max() * 0.01, i, f"{n} ({share:.1%})", va="center", fontsize=6)
    ax.set_xlim(0, dist["n"].max() * 1.35)
    ax.set_xlabel("Incidents (count)")
    ax.set_title(title)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_text_lengths(incidents: pd.DataFrame) -> plt.Figure:
    """Histograms of title length and description length in words."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 2.3))
    for ax, column, colour, label in (
        (axes[0], "title", PALETTE["blue"], "(a) Title length"),
        (axes[1], "description", PALETTE["orange"], "(b) Description length"),
    ):
        words = incidents[column].fillna("").str.split().str.len()
        ax.hist(words, bins=30, color=colour)
        ax.axvline(words.median(), color="black", ls="--", lw=0.8)
        ax.text(
            words.median(), ax.get_ylim()[1] * 0.92, f" median {words.median():.0f}", fontsize=6
        )
        ax.set_xlabel("Length (words)")
        ax.set_ylabel("Incidents (count)")
        ax.set_title(label)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_reports_per_incident(table: pd.DataFrame) -> plt.Figure:
    """Distribution of linked reports per incident (log-scaled counts)."""
    apply_style()
    counts = table["report_count"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 2.5))
    ax.bar(counts.index, counts.values, color=PALETTE["green"], width=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("Reports linked to the incident (count)")
    ax.set_ylabel("Incidents (count, log scale)")
    median = table["report_count"].median()
    ax.axvline(median, color="black", ls="--", lw=0.8)
    ax.text(median + 0.5, counts.max() * 0.6, f"median {median:.0f}", fontsize=6)
    ax.set_title("Reports per incident")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# --------------------------------------------------------------------------------------
# Later phases (stubs)
# --------------------------------------------------------------------------------------


def plot_topic_trends(topic_year: pd.DataFrame) -> Any:
    """Topic share over time, one line per topic."""
    raise NotImplementedError


def plot_topic_words(topic_words: dict[int, list[str]]) -> Any:
    """Bar charts of the top words per topic."""
    raise NotImplementedError


def plot_network(graph: Any, communities: dict | None = None) -> Any:
    """Interactive network figure coloured by community."""
    raise NotImplementedError


def plot_confusion_matrix(y_true: Any, y_pred: Any, labels: Any = None) -> Any:
    """Confusion-matrix heatmap."""
    raise NotImplementedError


def plot_model_comparison(results: pd.DataFrame, metric: str) -> Any:
    """Grouped bar chart comparing models on a metric."""
    raise NotImplementedError
