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
# Phase 4: trends and forecasting
# --------------------------------------------------------------------------------------

_ARROW = {"increasing": "\u25b2", "decreasing": "\u25bc", "no trend": "\u25cf"}
_DIRECTION_COLOUR = {
    "increasing": PALETTE["vermillion"],
    "decreasing": PALETTE["blue"],
    "no trend": "#BBBBBB",
}


def _short(label: str, width: int = 34) -> str:
    return label if len(label) <= width else label[: width - 1] + "\u2026"


def plot_theme_heatmap(
    share: pd.DataFrame,
    trend: pd.DataFrame,
    partial: pd.Series | None = None,
    provisional: bool = False,
) -> plt.Figure:
    """Themes x years heatmap of yearly shares (%), rows ordered by Sen slope; arrows mark BH-significant trends.

    `partial` (optional) is the share series of the partial year, drawn as a separated, hatched column.
    """
    apply_style()
    order = list(trend["group"])
    direction = dict(zip(trend["group"], trend["direction"], strict=True))
    data = share[order].T * 100
    years = list(data.columns)
    ncols = len(years) + (1 if partial is not None else 0)
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 0.34 * len(order) + 1.3))
    image = ax.imshow(data.to_numpy(), aspect="auto", cmap="cividis", vmin=0)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, fontsize=6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{_ARROW[direction[g]]} {_short(g)}" for g in order], fontsize=6)
    for i in range(len(order)):
        for j in range(len(years)):
            value = data.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if value < data.to_numpy().max() * 0.55 else "black",
            )
    if partial is not None:
        ax.set_xlim(-0.5, ncols - 0.5)
        extra = (partial.reindex(order) * 100).to_numpy()[:, None]
        ax.imshow(
            extra,
            aspect="auto",
            cmap="cividis",
            vmin=0,
            vmax=data.to_numpy().max(),
            extent=(len(years) + 0.1, len(years) + 1.1, len(order) - 0.5, -0.5),
            alpha=0.55,
        )
        ax.set_xlim(-0.5, len(years) + 1.1)
        ax.set_xticks([*range(len(years)), len(years) + 0.6])
        ax.set_xticklabels([*years, f"{partial.name}*"], rotation=45, fontsize=6)
    cbar = fig.colorbar(image, ax=ax, pad=0.01, fraction=0.03)
    cbar.set_label("Share of the year's incidents (%)", fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    ax.set_title(
        ("PROVISIONAL theme names. " if provisional else "")
        + "Theme share by year (\u25b2 increasing, \u25bc decreasing, \u25cf no trend; Mann-Kendall, BH-adjusted)",
        fontsize=7,
    )
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_emerging_small_multiples(
    share: pd.DataFrame,
    trend: pd.DataFrame,
    groups: list[str],
    partial: pd.Series | None = None,
    title: str = "",
) -> plt.Figure:
    """Small multiples of yearly share for `groups`, with the Sen's-slope line over the tested window."""
    apply_style()
    n = max(len(groups), 1)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(DOUBLE_COLUMN_IN, 1.75 * nrows + 0.5), sharex=True, squeeze=False
    )
    years = np.array(share.index)
    info = trend.set_index("group")
    for ax, group in zip(axes.ravel(), groups, strict=False):
        y = share[group].to_numpy() * 100
        slope = info.loc[group, "sen_slope"] * 100
        mid = np.median(np.arange(len(y)))
        intercept = np.median(y - slope * np.arange(len(y)))
        ax.plot(years, y, marker="o", ms=3, color=PALETTE["blue"], lw=1)
        ax.plot(
            years, intercept + slope * np.arange(len(y)), color=PALETTE["vermillion"], ls="--", lw=1
        )
        if partial is not None and group in partial.index:
            ax.plot(
                [years[-1], partial.name],
                [y[-1], partial[group] * 100],
                color=PALETTE["grey"],
                ls=":",
                lw=0.8,
            )
            ax.plot(
                [partial.name],
                [partial[group] * 100],
                marker="o",
                ms=3,
                mfc="white",
                color=PALETTE["grey"],
            )
        ax.set_title(
            f"{_short(group, 40)}\nSen {slope:+.2f} pp/yr, p_BH={info.loc[group, 'p_bh']:.3f}",
            fontsize=6.5,
        )
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=6)
        _ = mid
    for ax in axes.ravel()[len(groups) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Year the harm occurred")
    for ax in axes[:, 0]:
        ax.set_ylabel("Share (%)")
    fig.suptitle(title, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig


def plot_classification_matrix(
    matrix: pd.DataFrame,
    bases: list[str],
    separator_after: int | None = None,
    provisional: bool = False,
) -> plt.Figure:
    """Direction of every theme/domain under each basis (counts, shares, weighted shares, later window)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 0.24 * len(matrix) + 1.0))
    for i, (_name, row) in enumerate(matrix.iterrows()):
        for j, basis in enumerate(bases):
            ax.scatter(
                j, i, marker="s", s=170, color=_DIRECTION_COLOUR[row[basis]], edgecolor="white"
            )
            ax.text(
                j,
                i,
                _ARROW[row[basis]],
                ha="center",
                va="center",
                fontsize=6,
                color="white" if row[basis] != "no trend" else "black",
            )
        if not row["agree"]:
            ax.text(
                len(bases) - 0.35,
                i,
                "measures disagree",
                va="center",
                fontsize=5.5,
                color=PALETTE["vermillion"],
            )
        elif row.get("window_sensitive", False):
            ax.text(
                len(bases) - 0.35,
                i,
                "later window differs",
                va="center",
                fontsize=5.5,
                color=PALETTE["grey"],
            )
    ax.set_xticks(range(len(bases)))
    ax.set_xticklabels(bases, fontsize=6.5)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels([_short(str(n), 44) for n in matrix.index], fontsize=6)
    ax.set_xlim(-0.5, len(bases) + 0.3)
    ax.set_ylim(len(matrix) - 0.5, -0.5)
    if separator_after is not None:
        ax.axhline(separator_after - 0.5, color="black", lw=0.6)
    ax.xaxis.tick_top()
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_title(
        ("PROVISIONAL theme names. " if provisional else "")
        + "Trend direction by measure (\u25b2 increasing, \u25bc decreasing, \u25cf none; BH q = 0.05)",
        fontsize=7,
        pad=24,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def plot_forecast_mase(summary: pd.DataFrame, h_max: int) -> plt.Figure:
    """Mean MASE of naive / ETS / ARIMA by horizon; the dashed line at 1 marks the in-sample naive benchmark."""
    apply_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 2.5))
    cols = [
        ("mean_mase_h1", "1 quarter"),
        (f"mean_mase_h{h_max}", f"{h_max} quarters"),
        ("mean_mase_all_horizons", f"1-{h_max} quarters"),
    ]
    width = 0.8 / len(cols)
    colours = [PALETTE["sky"], PALETTE["orange"], PALETTE["blue"]]
    for k, (col, label) in enumerate(cols):
        ax.bar(
            np.arange(len(summary)) + (k - 1) * width,
            summary[col],
            width=width,
            label=label,
            color=colours[k],
        )
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary["model"])
    ax.set_ylabel("Mean MASE (lower is better)")
    ax.set_title("Rolling-origin forecast error, quarterly counts")
    ax.legend(frameon=False, fontsize=6, title="Horizon", title_fontsize=6)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# --------------------------------------------------------------------------------------
# Phase 5: network
# --------------------------------------------------------------------------------------

_TYPE_MARKER = {
    "organisation": "o",
    "ai_system": "s",
    "harmed_group": "^",
    "generic_actor": "D",
    "harm_value": "P",
    "other": "X",
    "person": "*",
}
_TYPE_NAME = {
    "organisation": "Organisation",
    "ai_system": "AI system",
    "harmed_group": "Harmed group",
    "generic_actor": "Generic actor",
    "harm_value": "Harm / value",
    "other": "Other",
    "person": "Person-like",
}
_COMMUNITY_COLOURS = [
    PALETTE["blue"],
    PALETTE["orange"],
    PALETTE["green"],
    PALETTE["vermillion"],
    PALETTE["sky"],
    PALETTE["purple"],
    PALETTE["yellow"],
    "#000000",
]


def plot_entity_network(
    graph,
    assignment: dict,
    top_n: int = 110,
    label_n: int = 14,
    seed: int = 42,
    provisional_note: str = "",
) -> plt.Figure:
    """Spring layout of the top `top_n` entities by weighted degree; colour = community, marker = node type."""
    import networkx as nx

    apply_style()
    degree = dict(graph.degree(weight="weight"))
    keep = sorted(degree, key=degree.get, reverse=True)[:top_n]
    sub = graph.subgraph(keep)
    pos = nx.spring_layout(
        sub, weight="weight", seed=seed, k=1.4 / np.sqrt(len(keep)), iterations=200
    )
    sizes = pd.Series({n: assignment[n] for n in keep}).value_counts()
    main = list(sizes.index[:8])
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 5.4))
    strongest = sorted(sub.edges(data=True), key=lambda e: -e[2]["weight"])[: 6 * len(keep)]
    for a, b, d in strongest:
        ax.plot(
            [pos[a][0], pos[b][0]],
            [pos[a][1], pos[b][1]],
            color="#999999",
            lw=0.15 + 0.12 * np.log1p(d["weight"]),
            alpha=0.35,
            zorder=1,
        )
    for node_type, marker in _TYPE_MARKER.items():
        nodes = [n for n in keep if graph.nodes[n]["type"] == node_type]
        if not nodes:
            continue
        colours = [
            _COMMUNITY_COLOURS[main.index(assignment[n])] if assignment[n] in main else "#BBBBBB"
            for n in nodes
        ]
        ax.scatter(
            [pos[n][0] for n in nodes],
            [pos[n][1] for n in nodes],
            s=[8 + 3.2 * np.sqrt(graph.nodes[n]["n_incidents"]) ** 1.6 / 4 for n in nodes],
            c=colours,
            marker=marker,
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
            label=_TYPE_NAME[node_type],
        )
    # Label the most connected entities, skipping any whose label would sit on top of one already placed.
    placed: list[np.ndarray] = []
    extent = np.ptp(np.array(list(pos.values())), axis=0)
    min_gap = 0.075 * float(extent.max())
    for n in keep:
        if len(placed) >= label_n:
            break
        point = np.asarray(pos[n])
        if any(np.linalg.norm(point - q) < min_gap for q in placed):
            continue
        placed.append(point)
        ax.annotate(
            textwrap.shorten(graph.nodes[n]["label"], 26, placeholder="..."),
            pos[n],
            fontsize=5.5,
            ha="center",
            va="bottom",
            zorder=3,
            xytext=(0, 3),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.75},
        )
    ax.set_axis_off()
    leg = ax.legend(
        frameon=False,
        fontsize=6,
        loc="lower left",
        title="Node type",
        title_fontsize=6,
        markerscale=0.7,
    )
    for handle in leg.legend_handles:
        handle.set_color("#777777")
    ax.set_title(
        f"Entity co-occurrence network, top {len(keep)} entities by weighted degree (colour = community, size = incidents)"
        + provisional_note,
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_top_central(tables: dict[str, pd.DataFrame], measure: str = "pagerank") -> plt.Figure:
    """Top-10 bars per node type (e.g. organisations, AI systems, harmed groups) for one centrality measure."""
    apply_style()
    fig, axes = plt.subplots(1, len(tables), figsize=(DOUBLE_COLUMN_IN, 3.0))
    axes = np.atleast_1d(axes)
    colours = [PALETTE["blue"], PALETTE["green"], PALETTE["vermillion"], PALETTE["purple"]]
    for ax, ((title, table), colour) in zip(
        axes, zip(tables.items(), colours, strict=False), strict=False
    ):
        data = table.iloc[::-1]
        ax.barh(
            [textwrap.shorten(str(x), 30, placeholder="...") for x in data["label"]],
            data[measure],
            color=colour,
        )
        ax.set_title(title, fontsize=7)
        ax.set_xlabel(measure.replace("_", " "))
        ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_tau_heatmap(matrix: pd.DataFrame) -> plt.Figure:
    """Kendall's tau between centrality measures."""
    apply_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 3.0))
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="cividis")
    labels = [c.replace("_", " ") for c in matrix.columns]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = matrix.iloc[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if v < 0.6 else "black",
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Kendall tau", fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    ax.set_title("Agreement between centrality measures", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_community_methods(runs: pd.DataFrame, summary: pd.DataFrame, chosen: str) -> plt.Figure:
    """(a) modularity per run, (b) stability (mean pairwise ARI over seeds), (c) share of nodes in the largest community."""
    apply_style()
    methods = list(summary["method"])
    colours = {
        "louvain": PALETTE["blue"],
        "leiden": PALETTE["orange"],
        "label_propagation": PALETTE["green"],
    }
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.5))
    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        sub = runs[runs["method"] == m]
        axes[0].scatter(
            i + rng.uniform(-0.12, 0.12, len(sub)),
            sub["modularity"],
            color=colours[m],
            s=10,
            alpha=0.8,
        )
        axes[0].hlines(sub["modularity"].mean(), i - 0.25, i + 0.25, color="black", lw=1)
    axes[0].set_title("(a) Modularity per seed", fontsize=7)
    axes[1].bar(
        range(len(methods)), summary["stability_ari_mean"], color=[colours[m] for m in methods]
    )
    axes[1].scatter(
        range(len(methods)),
        summary["stability_ari_min"],
        color="black",
        marker="_",
        s=120,
        label="minimum pair",
    )
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False, fontsize=6)
    axes[1].set_title("(b) Stability (mean pairwise ARI)", fontsize=7)
    axes[2].bar(
        range(len(methods)), summary["largest_share_mean"], color=[colours[m] for m in methods]
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_title("(c) Share of nodes in largest community", fontsize=7)
    for ax in axes:
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(
            [m.replace("_", " ") + ("*" if m == chosen else "") for m in methods],
            rotation=30,
            ha="right",
            fontsize=6,
        )
    fig.suptitle(
        "Community detection on the entity network (* = chosen by the pre-specified rule)",
        fontsize=7,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


def plot_network_growth(snapshots: pd.DataFrame, partial_year: int | None = None) -> plt.Figure:
    """(a) cumulative node counts by type; (b) organisations first appearing each year."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 2.6))
    series = [
        ("incident", "Incidents", PALETTE["black"] if "black" in PALETTE else "#000000"),
        ("harmed_group", "Harmed groups", PALETTE["vermillion"]),
        ("ai_system", "AI systems", PALETTE["green"]),
        ("organisation", "Organisations", PALETTE["blue"]),
        ("generic_actor", "Generic actors", PALETTE["orange"]),
    ]
    for key, label, colour in series:
        col = f"nodes_{key}"
        if col in snapshots:
            axes[0].plot(
                snapshots["year"],
                snapshots[col],
                marker="o",
                ms=2.5,
                lw=1,
                label=label,
                color=colour,
            )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Year (cumulative to end of year)")
    axes[0].set_ylabel("Nodes (count, log scale)")
    axes[0].legend(frameon=False, fontsize=6)
    axes[0].set_title("(a) Cumulative nodes by type", fontsize=7)
    bars = axes[1].bar(snapshots["year"], snapshots["new_organisations"], color=PALETTE["blue"])
    if partial_year is not None:
        for bar, year in zip(bars, snapshots["year"], strict=True):
            if year == partial_year:
                bar.set_hatch("//")
                bar.set_edgecolor("white")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("New organisations (count)")
    axes[1].set_title(
        "(b) Organisations first appearing in the year"
        + (" (hatched = partial)" if partial_year else ""),
        fontsize=7,
    )
    for ax in axes:
        ax.tick_params(axis="x", rotation=45, labelsize=6)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_rank_bump(top: pd.DataFrame, min_years: int = 4, top_show: int = 8) -> plt.Figure:
    """Rank of organisations in the yearly top 10 by PageRank (lines for those present in >= `min_years` years)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 3.2))
    present = top.groupby("organisation")["year"].nunique()
    final = top[top["year"] == top["year"].max()].set_index("organisation")["rank"]
    chosen = list(present[present >= min_years].index)
    chosen = sorted(chosen, key=lambda o: (final.get(o, 99), -present[o]))[:top_show]
    colours = [
        PALETTE["blue"],
        PALETTE["orange"],
        PALETTE["green"],
        PALETTE["vermillion"],
        PALETTE["sky"],
        PALETTE["purple"],
        "#000000",
        PALETTE["grey"],
    ]
    for org, colour in zip(chosen, colours, strict=False):
        sub = top[top["organisation"] == org].sort_values("year")
        ax.plot(sub["year"], sub["rank"], marker="o", ms=3, lw=1.2, color=colour)
        ax.annotate(
            org,
            (sub["year"].iloc[-1], sub["rank"].iloc[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=6,
            color=colour,
            va="center",
        )
    ax.set_ylim(10.5, 0.5)
    ax.set_yticks(range(1, 11))
    ax.set_xlim(top["year"].min() - 0.3, top["year"].max() + 2.2)
    ax.set_xlabel("Year (cumulative network to end of year)")
    ax.set_ylabel("Rank by PageRank among organisations")
    ax.set_title(
        "Organisations in the yearly top 10 by PageRank (alleged involvement, not blame)",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# --------------------------------------------------------------------------------------
# Phase 6: classification
# --------------------------------------------------------------------------------------


def plot_classifier_comparison(
    table: pd.DataFrame, baseline_name: str = "majority class"
) -> plt.Figure:
    """Macro-F1 with bootstrap 95% CI per model for cross-validation and the time split (table columns:
    model, setting, macro_f1, ci_low, ci_high)."""
    apply_style()
    models = list(dict.fromkeys(table["model"]))
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 0.42 * len(models) + 1.2))
    offsets = {"cv": -0.17, "time": 0.17}
    colours = {"cv": PALETTE["blue"], "time": PALETTE["orange"]}
    labels = {"cv": "5-fold cross-validation", "time": "time split (train <= 2023, test >= 2024)"}
    for setting in ("cv", "time"):
        sub = table[table["setting"] == setting].set_index("model").reindex(models)
        y = np.arange(len(models)) + offsets[setting]
        ax.errorbar(
            sub["macro_f1"],
            y,
            xerr=[sub["macro_f1"] - sub["ci_low"], sub["ci_high"] - sub["macro_f1"]],
            fmt="o",
            ms=4,
            color=colours[setting],
            capsize=2,
            label=labels[setting],
        )
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1 (point and bootstrap 95% CI)")
    ax.set_xlim(0, max(0.8, float(table["ci_high"].max()) + 0.03))
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax.set_title(
        "Risk-domain classification: model comparison (MIT labels are machine-generated)",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _ = baseline_name
    return fig


def plot_confusion_heatmap(
    cm: pd.DataFrame, title: str, xlabel: str = "Predicted (out-of-fold)", ylabel: str = "MIT label"
) -> plt.Figure:
    """Row-normalised confusion matrix (share of each true class predicted as each class)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN + 1.6, 3.6))
    image = ax.imshow(cm.to_numpy(), vmin=0, vmax=1, cmap="cividis")
    short = [textwrap.shorten(c, 24, placeholder="...") for c in cm.columns]
    ax.set_xticks(range(len(short)))
    ax.set_xticklabels(short, rotation=50, ha="right", fontsize=6)
    ax.set_yticks(range(len(short)))
    ax.set_yticklabels(short, fontsize=6)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.iloc[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if v < 0.55 else "black",
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_per_class_f1(per_class: pd.DataFrame, support: dict[str, int]) -> plt.Figure:
    """Grouped bars of per-class F1 (columns: model, class, f1), classes ordered by size."""
    apply_style()
    classes = sorted(support, key=support.get, reverse=True)
    models = list(dict.fromkeys(per_class["model"]))
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 3.0))
    width = 0.8 / len(models)
    for k, model in enumerate(models):
        sub = per_class[per_class["model"] == model].set_index("class").reindex(classes)
        ax.bar(
            np.arange(len(classes)) + (k - (len(models) - 1) / 2) * width,
            sub["f1"],
            width=width,
            label=model,
            color=CATEGORICAL[k % len(CATEGORICAL)],
        )
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f"{textwrap.fill(c, 18)}\n(n={support[c]})" for c in classes], fontsize=6)
    ax.set_ylabel("F1 (cross-validation, out-of-fold)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper right")
    ax.set_title("Per-domain F1: the small domains are the hard ones", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_reliability(panels: list[tuple[str, pd.DataFrame, float]], chosen: str) -> plt.Figure:
    """Reliability diagrams (top-label confidence vs accuracy) for several calibration options.

    `panels` = [(label, reliability table, ECE)]; `chosen` is the method name highlighted in the title.
    """
    apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(DOUBLE_COLUMN_IN, 2.7), sharey=True)
    colours = [PALETTE["vermillion"], PALETTE["orange"], PALETTE["blue"]]
    for ax, (label, table, ece), colour in zip(axes, panels, colours, strict=False):
        valid = table[table["n"] > 0]
        ax.plot([0, 1], [0, 1], color="black", ls="--", lw=0.8)
        ax.plot(valid["confidence"], valid["accuracy"], marker="o", color=colour, lw=1)
        for _, r in valid.iterrows():
            ax.annotate(
                str(int(r["n"])),
                (r["confidence"], r["accuracy"]),
                fontsize=5,
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence of the predicted domain")
        ax.set_title(f"{label}: ECE = {ece:.3f}", fontsize=7)
    axes[0].set_ylabel("Accuracy in the bin (labels = bin size)")
    fig.suptitle(
        f"Reliability of the deployable model's confidence (selected: {chosen.replace('raw', 'uncalibrated')})",
        fontsize=8,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig


# --------------------------------------------------------------------------------------
# Phase 7: retrieval
# --------------------------------------------------------------------------------------


def plot_retrieval_comparison(
    summaries: dict[str, pd.DataFrame], labels: dict[str, str], p5_ceiling: float
) -> plt.Figure:
    """MRR, nDCG@10 and P@5 with bootstrap 95% CIs per method, one marker colour per setting."""
    apply_style()
    metrics = [("rr", "MRR"), ("ndcg@10", "nDCG@10"), ("p@5", "Precision@5")]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.9), sharey=True)
    colours = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"]]
    first = next(iter(summaries.values()))
    methods = list(first["method"])
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for k, (setting, table) in enumerate(summaries.items()):
            t = table.set_index("method").reindex(methods)
            y = np.arange(len(methods)) + (k - (len(summaries) - 1) / 2) * 0.25
            ax.errorbar(
                t[metric],
                y,
                xerr=[t[metric] - t[f"{metric}_low"], t[f"{metric}_high"] - t[metric]],
                fmt="o",
                ms=3.5,
                capsize=2,
                color=colours[k],
                label=setting,
            )
        if metric == "p@5":
            ax.axvline(p5_ceiling, color="black", ls="--", lw=0.8)
            title = f"{title} (dashed = best possible, {p5_ceiling:.2f})"
        ax.set_title(title, fontsize=7)
        ax.set_xlim(left=0)
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels([labels[m] for m in methods], fontsize=6.5)
    axes[0].invert_yaxis()
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        frameon=False,
        fontsize=6,
        loc="lower center",
        ncol=len(names),
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "Similar-incident retrieval against editor-marked similar incidents (mean and bootstrap 95% CI over queries)",
        fontsize=7.5,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return fig


def plot_recall_at_k(
    summaries: dict[str, pd.DataFrame], ks: tuple[int, ...], labels: dict[str, str]
) -> plt.Figure:
    """Recall@k curves per method, one panel per setting."""
    apply_style()
    fig, axes = plt.subplots(1, len(summaries), figsize=(DOUBLE_COLUMN_IN, 2.7), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (setting, table) in zip(axes, summaries.items(), strict=True):
        for k, row in enumerate(table.itertuples(index=False)):
            r = row._asdict()
            ax.plot(
                ks,
                [table.loc[table["method"] == r["method"], f"recall@{x}"].iloc[0] for x in ks],
                marker="o",
                ms=2.5,
                lw=1,
                color=CATEGORICAL[k % len(CATEGORICAL)],
                label=labels[r["method"]],
            )
        ax.set_xscale("log")
        ax.set_xticks(list(ks))
        ax.set_xticklabels([str(x) for x in ks])
        ax.set_ylim(0, 1)
        ax.set_xlabel("k (results inspected)")
        ax.set_title(setting, fontsize=7)
    axes[0].set_ylabel("Recall@k (share of editor-similar found)")
    axes[-1].legend(frameon=False, fontsize=5.5, loc="lower right")
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
