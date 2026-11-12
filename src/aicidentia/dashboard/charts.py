"""Plotly figures for the dashboard (pure functions of data frames; colour-blind-safe Okabe-Ito palette)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from aicidentia.viz.style import CATEGORICAL, PALETTE

TYPE_SYMBOL = {
    "organisation": "circle",
    "ai_system": "square",
    "harmed_group": "triangle-up",
    "generic_actor": "diamond",
    "harm_value": "cross",
    "other": "x",
    "person": "star",
}
TYPE_NAME = {
    "organisation": "Organisation (alleged)",
    "ai_system": "AI system",
    "harmed_group": "Harmed group",
    "generic_actor": "Generic actor",
    "harm_value": "Harm / value",
    "other": "Other",
    "person": "Person-like",
}


def _layout(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        margin={"l": 10, "r": 10, "t": 50 if title else 20, "b": 10},
        font={"size": 12},
        legend={"orientation": "h", "y": -0.15},
    )
    return fig


def incidents_per_year(counts: pd.DataFrame) -> go.Figure:
    """Bars per year; the partial year is hatched and labelled."""
    colours = [PALETTE["grey"] if p else PALETTE["blue"] for p in counts["partial"]]
    fig = go.Figure(
        go.Bar(
            x=counts["year"],
            y=counts["incidents"],
            marker_color=colours,
            marker_pattern_shape=["/" if p else "" for p in counts["partial"]],
            hovertemplate="%{x}: %{y} incidents<extra></extra>",
        )
    )
    fig.update_xaxes(title="Year the harm occurred")
    fig.update_yaxes(title="Incidents")
    return _layout(fig, "Incidents per year (hatched = year not yet complete)")


def domain_bars(counts: pd.Series) -> go.Figure:
    fig = go.Figure(
        go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=PALETTE["blue"])
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Incidents (MIT labels are machine-generated)")
    return _layout(fig, "MIT risk domains", 340)


def theme_lines(yearly: pd.DataFrame, themes: list[str], measure: str = "share") -> go.Figure:
    """Line per selected theme over years; the partial year is drawn as a hollow marker."""
    fig = go.Figure()
    for k, theme in enumerate(themes):
        sub = yearly[yearly["group"] == theme].sort_values("year")
        full, part = sub[~sub["partial_year"]], sub[sub["partial_year"]]
        colour = CATEGORICAL[k % len(CATEGORICAL)]
        scale = 100 if measure == "share" else 1
        fig.add_trace(
            go.Scatter(
                x=full["year"],
                y=full[measure] * scale,
                mode="lines+markers",
                name=theme,
                line={"color": colour},
            )
        )
        if len(part):
            fig.add_trace(
                go.Scatter(
                    x=part["year"],
                    y=part[measure] * scale,
                    mode="markers",
                    showlegend=False,
                    marker={"color": "white", "line": {"color": colour, "width": 2}, "size": 9},
                    hovertemplate="partial year: %{y:.1f}<extra></extra>",
                )
            )
    fig.update_yaxes(
        title="Share of the year's incidents (%)" if measure == "share" else "Incidents",
        rangemode="tozero",
    )
    fig.update_xaxes(title="Year the harm occurred")
    return _layout(fig, "", 420)


def theme_heatmap(yearly: pd.DataFrame, order: list[str]) -> go.Figure:
    """Themes x complete years, share in %."""
    data = (
        yearly[~yearly["partial_year"]]
        .pivot(index="group", columns="year", values="share")
        .reindex(order)
        * 100
    )
    fig = go.Figure(
        go.Heatmap(
            z=data.values,
            x=data.columns,
            y=[g[:45] for g in data.index],
            colorscale="Cividis",
            colorbar={"title": "%"},
            hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Share of incidents per theme and year", 30 * len(order) + 120)


def network_figure(
    data: dict[str, Any], year: int, types: list[str], highlight: str | None = None
) -> go.Figure:
    """Cumulative network up to `year`: nodes first seen by then; edges with incidents by then (weight = count)."""
    nodes = {n["id"]: n for n in data["nodes"] if n["first_year"] <= year and n["type"] in types}
    fig = go.Figure()
    xs, ys = [], []
    for e in data["edges"]:
        if e["source"] in nodes and e["target"] in nodes:
            w = sum(c for y, c in e["by_year"].items() if int(y) <= year)
            if w > 0:
                a, b = nodes[e["source"]], nodes[e["target"]]
                xs += [a["x"], b["x"], None]
                ys += [a["y"], b["y"], None]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": "#BBBBBB", "width": 0.5},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    communities = sorted({n["community"] for n in nodes.values()})
    colour = {c: CATEGORICAL[i % len(CATEGORICAL)] for i, c in enumerate(communities)}
    for node_type in types:
        group = [n for n in nodes.values() if n["type"] == node_type]
        if not group:
            continue
        fig.add_trace(
            go.Scatter(
                x=[n["x"] for n in group],
                y=[n["y"] for n in group],
                mode="markers",
                name=TYPE_NAME.get(node_type, node_type),
                marker={
                    "symbol": TYPE_SYMBOL.get(node_type, "circle"),
                    "size": [6 + min(n["n_incidents"], 200) ** 0.5 * 1.6 for n in group],
                    "color": [colour[n["community"]] for n in group],
                    "line": {
                        "width": [3 if n["id"] == highlight else 0.5 for n in group],
                        "color": ["black" if n["id"] == highlight else "white" for n in group],
                    },
                },
                text=[n["label"] for n in group],
                customdata=[[n["id"], n["n_incidents"], n["community"]] for n in group],
                hovertemplate="%{text}<br>%{customdata[1]} incidents, community %{customdata[2]}<extra></extra>",
            )
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(
        fig,
        f"Entities and their co-occurrence up to {year} (colour = community, symbol = type)",
        620,
    )


def probability_bars(prediction: pd.DataFrame) -> go.Figure:
    probs = prediction.filter(like="p_").iloc[0].sort_values()
    fig = go.Figure(
        go.Bar(
            x=probs.values * 100,
            y=[c[2:] for c in probs.index],
            orientation="h",
            marker_color=[
                PALETTE["vermillion"] if c == probs.index[-1] else PALETTE["blue"]
                for c in probs.index
            ],
        )
    )
    fig.update_xaxes(title="Calibrated probability (%)", range=[0, 100])
    return _layout(fig, "", 300)


def model_comparison(metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for k, setting in enumerate(["cv", "time"]):
        sub = metrics[(metrics["setting"] == setting) & ~metrics["model"].str.contains("ablation")]
        fig.add_trace(
            go.Scatter(
                x=sub["macro_f1"],
                y=sub["model"],
                mode="markers",
                name={"cv": "5-fold CV", "time": "time split"}[setting],
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": sub["ci_high"] - sub["macro_f1"],
                    "arrayminus": sub["macro_f1"] - sub["ci_low"],
                },
                marker={"color": [PALETTE["blue"], PALETTE["orange"]][k], "size": 9},
            )
        )
    fig.update_xaxes(title="Macro-F1 (bootstrap 95% CI)", range=[0, 0.8])
    return _layout(fig, "", 360)
