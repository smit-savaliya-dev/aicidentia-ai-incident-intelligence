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

from aicidentia.viz.style import (
    CATEGORICAL,
    DOUBLE_COLUMN_IN,
    PALETTE,
    SINGLE_COLUMN_IN,
    apply_style,
)

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
# Phase 2: exploratory figures on the clean tables
# --------------------------------------------------------------------------------------


def plot_incidents_per_year_by_label(
    incidents: pd.DataFrame, partial_year: int, first_year: int = 2015
) -> plt.Figure:
    """Stacked bars of incidents per year: MIT-labelled vs not labelled (partial year hatched)."""
    apply_style()
    sub = incidents[incidents["year"] >= first_year]
    table = (
        sub.groupby(["year", "has_mit"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[True, False], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 2.5))
    labelled = ax.bar(table.index, table[True], color=PALETTE["blue"], label="MIT-labelled")
    unlabelled = ax.bar(
        table.index, table[False], bottom=table[True], color=PALETTE["orange"], label="No MIT label"
    )
    for bars in (labelled, unlabelled):
        for bar, year in zip(bars, table.index, strict=True):
            if year == partial_year:
                bar.set_hatch("//")
                bar.set_edgecolor("white")
    for year, total in table.sum(axis=1).items():
        ax.text(year, total + 5, str(total), ha="center", va="bottom", fontsize=6)
    ax.set_xticks(table.index)
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Year the harm occurred")
    ax.set_ylabel("Incidents (count)")
    ax.set_title(
        f"Incidents per year by MIT-label availability ({partial_year} is partial, hatched)"
    )
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_top_entities(
    table: pd.DataFrame, title: str, xlabel: str, colour: str = PALETTE["blue"], n: int = 15
) -> plt.Figure:
    """Horizontal bar chart of the `n` most frequent entities (columns: entity, n_incidents)."""
    apply_style()
    top = table.head(n).iloc[::-1]
    labels = ["\n".join(textwrap.wrap(str(e), 34)) for e in top["entity"]]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 3.4))
    ax.barh(labels, top["n_incidents"], color=colour)
    for i, count in enumerate(top["n_incidents"]):
        ax.text(count + top["n_incidents"].max() * 0.01, i, str(count), va="center", fontsize=6)
    ax.set_xlim(0, top["n_incidents"].max() * 1.15)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_label_distributions(panels: list[tuple[str, pd.Series]]) -> plt.Figure:
    """Up to four small bar panels of label distributions (title, counts indexed by class)."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN, 4.6))
    for ax, (title, counts), colour in zip(axes.ravel(), panels, CATEGORICAL, strict=False):
        counts = counts.sort_values()
        labels = ["\n".join(textwrap.wrap(str(c), 32)) for c in counts.index]
        ax.barh(labels, counts.values, color=colour)
        for i, value in enumerate(counts.values):
            ax.text(
                value + counts.max() * 0.01,
                i,
                f"{value} ({value / counts.sum():.0%})",
                va="center",
                fontsize=6,
            )
        ax.set_xlim(0, counts.max() * 1.35)
        ax.set_xlabel("Incidents (count)")
        ax.set_title(title)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# --------------------------------------------------------------------------------------
# Phase 3: topic-model comparison
# --------------------------------------------------------------------------------------

_FAMILY_COLOUR = {"LDA": PALETTE["blue"], "NMF": PALETTE["orange"]}


def _contender_colour(contender: str, bertopic_index: dict[str, int]) -> str:
    if contender in _FAMILY_COLOUR:
        return _FAMILY_COLOUR[contender]
    palette = [PALETTE["green"], PALETTE["vermillion"], PALETTE["purple"], PALETTE["sky"]]
    return palette[bertopic_index.setdefault(contender, len(bertopic_index)) % len(palette)]


def plot_coherence_by_config(summary: pd.DataFrame, eligible: tuple[int, int]) -> plt.Figure:
    """C_v and NPMI against the number of topics for every configuration (mean +/- sd over seeds)."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 2.9), sharex=True)
    seen: dict[str, int] = {}
    for ax, (metric, title) in zip(
        axes, (("c_v", "(a) C_v coherence"), ("c_npmi", "(b) NPMI coherence")), strict=True
    ):
        ax.axvspan(eligible[0], eligible[1], color=PALETTE["sky"], alpha=0.12, lw=0)
        for contender, group in summary.groupby("contender", sort=False):
            group = group.sort_values("n_topics_mean")
            colour = _contender_colour(contender, seen)
            ax.errorbar(
                group["n_topics_mean"],
                group[f"{metric}_mean"],
                yerr=group[f"{metric}_sd"],
                marker="o",
                ms=3,
                lw=1,
                capsize=2,
                color=colour,
                label=contender,
            )
        ax.set_xlabel("Number of topics (mean over seeds)")
        ax.set_ylabel(title[4:])
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, 0.03),
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


def plot_best_model_comparison(best: pd.DataFrame, winner: str) -> plt.Figure:
    """Four panels (C_v, NPMI, diversity, stability) comparing each contender's best configuration."""
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COLUMN_IN, 2.5))
    seen: dict[str, int] = {}
    names = list(best["contender"])
    colours = [_contender_colour(c, seen) for c in names]
    spec = (
        ("c_v_mean", "c_v_sd", "C_v"),
        ("c_npmi_mean", "c_npmi_sd", "NPMI"),
        ("diversity_mean", "diversity_sd", "Topic diversity"),
        ("stability", None, "Seed stability"),
    )
    for ax, (mean, sd, title) in zip(axes, spec, strict=True):
        err = best[sd] if sd else None
        bars = ax.bar(range(len(best)), best[mean], yerr=err, color=colours, capsize=2)
        for bar, name in zip(bars, names, strict=True):
            if name == winner:
                bar.set_edgecolor("black")
                bar.set_linewidth(1.4)
        ax.set_xticks(range(len(best)))
        ax.set_xticklabels(
            [n.replace("BERTopic", "BERT") for n in names], rotation=60, ha="right", fontsize=6
        )
        ax.set_title(title)
    fig.suptitle(
        "Best eligible configuration per contender (outlined = selected)", fontsize=8, y=1.0
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def plot_topic_sizes(table: pd.DataFrame, title: str) -> plt.Figure:
    """Incidents per topic of the chosen model, labelled with the first three terms."""
    apply_style()
    data = table.sort_values("n_incidents")
    labels = [
        f"T{t}: {lab}" for t, lab in zip(data["topic_id"], data["suggested_label"], strict=True)
    ]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, max(2.4, 0.19 * len(data) + 0.8)))
    ax.barh(labels, data["n_incidents"], color=PALETTE["blue"])
    for i, count in enumerate(data["n_incidents"]):
        ax.text(count + data["n_incidents"].max() * 0.01, i, str(count), va="center", fontsize=6)
    ax.set_xlim(0, data["n_incidents"].max() * 1.15)
    ax.set_xlabel("Incidents (count)")
    ax.set_title(title)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
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
