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


FONT = {"family": "Inter, Segoe UI, Helvetica, Arial, sans-serif", "size": 14, "color": "#1B2430"}
GRID = "#E6EBF1"
SHORT_DOMAIN = {
    "AI system safety, failures, and limitations": "AI system safety",
    "Discrimination and Toxicity": "Discrimination & toxicity",
    "Human-Computer Interaction": "Human-computer interaction",
    "Malicious Actors & Misuse": "Malicious actors & misuse",
    "Misinformation": "Misinformation",
    "Privacy & Security": "Privacy & security",
    "Socioeconomic & Environmental Harms": "Socioeconomic & environmental",
}


def short(name: str) -> str:
    return SHORT_DOMAIN.get(name, name)


def _layout(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """One calm house style: white background, light grid, readable fonts, legend below."""
    fig.update_layout(
        title={"text": title, "font": {"size": 16}, "x": 0, "xanchor": "left"},
        height=height,
        margin={"l": 10, "r": 10, "t": 50 if title else 16, "b": 10},
        font=FONT,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={"orientation": "h", "y": -0.18, "x": 0},
        hoverlabel={"font_size": 13},
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
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
    ).fillna(0.0)
    fig = go.Figure(
        go.Heatmap(
            z=data.values,
            x=data.columns,
            y=[g[:45] for g in data.index],
            colorscale="Blues",
            zmin=0,
            colorbar={"title": "% of year"},
            xgap=1,
            ygap=1,
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
    fig.update_yaxes(ticklabelstandoff=6)
    return _layout(fig, "", 300)


def model_comparison(metrics: pd.DataFrame) -> go.Figure:
    """Macro-F1 with 95% CI per model, cross-validation and time split."""
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


# --------------------------------------------------------------------------------------
# Story and reliability charts
# --------------------------------------------------------------------------------------


def chatgpt_slope(table: pd.DataFrame, title: str) -> go.Figure:
    """Share before vs after 30 Nov 2022; blue = significantly more common after, vermillion = less."""
    fig = go.Figure()
    for r in table.itertuples():
        colour = (
            PALETTE["blue"]
            if r.adj_residual_after > 1.96
            else PALETTE["vermillion"]
            if r.adj_residual_after < -1.96
            else PALETTE["grey"]
        )
        fig.add_trace(
            go.Scatter(
                x=["Before 30 Nov 2022", "After 30 Nov 2022"],
                y=[100 * r.share_before, 100 * r.share_after],
                mode="lines+markers",
                line={"color": colour, "width": 3},
                marker={"size": 9},
                name=short(r.group),
                showlegend=False,
                hovertemplate=f"{r.group}<br>%{{x}}: %{{y:.1f}}%<br>adjusted residual {r.adj_residual_after:+.1f}<extra></extra>",
            )
        )
    # labels at the right end, nudged apart so they never overlap
    values = (100 * table["share_after"]).to_numpy()
    gap = max(values.max(), (100 * table["share_before"]).max()) * 0.045
    order = values.argsort()
    placed: list[float] = []
    for i in order:
        y = values[i] if not placed or values[i] - placed[-1] >= gap else placed[-1] + gap
        placed.append(y)
        r = table.iloc[i]
        colour = (
            PALETTE["blue"]
            if r["adj_residual_after"] > 1.96
            else PALETTE["vermillion"]
            if r["adj_residual_after"] < -1.96
            else PALETTE["grey"]
        )
        fig.add_annotation(
            x=1,
            y=y,
            xref="x",
            text=f"{short(r['group'])} {values[i]:.0f}%",
            showarrow=False,
            xanchor="left",
            xshift=10,
            font={"size": 12, "color": colour},
        )
    fig.update_yaxes(title="Share of incidents (%)", rangemode="tozero")
    fig.update_xaxes(range=[-0.15, 2.1])
    return _layout(fig, title, 540)


def _wrap(text: str, width: int = 22) -> str:
    import textwrap

    return "<br>".join(textwrap.wrap(text, width))


def risk_matrix(matrix: pd.DataFrame, threshold: float) -> go.Figure:
    """Bubble chart: trend (pp/yr) vs CSET severity; unsupported themes grey with their CSET n."""
    fig = go.Figure()
    order = matrix.sort_values("trend_pp_per_year")["theme"].tolist()
    position = {t: ("top center" if i % 2 == 0 else "bottom center") for i, t in enumerate(order)}
    for supported in (False, True):
        sub = matrix[matrix["supported"] == supported]
        fig.add_trace(
            go.Scatter(
                x=sub["trend_pp_per_year"],
                y=100 * sub["severity"].fillna(0),
                mode="markers+text",
                text=[
                    _wrap(t if ok else f"{t} (CSET n={n})")
                    for t, ok, n in zip(sub["theme"], sub["supported"], sub["cset_n"], strict=True)
                ],
                textposition=[position[t] for t in sub["theme"]],
                textfont={"size": 12, "color": "#1B2430" if supported else "#8A94A3"},
                customdata=sub[["theme", "incidents", "cset_n", "q_value", "quadrant"]].to_numpy(),
                marker={
                    "size": sub["incidents"],
                    "sizemode": "area",
                    "sizeref": 2.0 * matrix["incidents"].max() / 70**2,
                    "color": PALETTE["blue"] if supported else "#C9D1DB",
                    "line": {"color": "#1B2430" if supported else "#9AA4B2", "width": 1},
                    "opacity": 0.85 if supported else 0.55,
                },
                name="enough CSET data (n >= 30)"
                if supported
                else "too few CSET annotations (grey)",
                hovertemplate="<b>%{customdata[0]}</b><br>trend %{x:+.2f} pp/yr (q=%{customdata[3]:.3f})"
                "<br>severity %{y:.0f}% of %{customdata[2]} CSET-annotated<br>%{customdata[1]} incidents"
                "<br>%{customdata[4]}<extra></extra>",
            )
        )
    fig.add_vline(x=0, line={"color": "#8A94A3", "dash": "dash", "width": 1})
    fig.add_hline(y=100 * threshold, line={"color": "#8A94A3", "dash": "dash", "width": 1})
    for text, x, y, anchor in (
        ("EMERGING & SEVERE", 0.99, 0.99, "right"),
        ("ESTABLISHED & SEVERE", 0.01, 0.99, "left"),
        ("EMERGING", 0.99, 0.02, "right"),
        ("ESTABLISHED", 0.01, 0.02, "left"),
    ):
        fig.add_annotation(
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            text=f"<b>{text}</b>",
            showarrow=False,
            xanchor=anchor,
            font={"size": 13, "color": PALETTE["purple"]},
        )
    fig.update_xaxes(
        title="Trend: Sen's slope of the theme's yearly share (percentage points per year)"
    )
    fig.update_yaxes(
        title="Severity: CSET-annotated incidents with tangible harm, injuries or deaths (%)",
        range=[-30, 115],
    )
    return _layout(fig, "", 620)


def ego_network(ego: dict[str, Any], colour_by: str = "community") -> go.Figure:
    """A node and its neighbours; edge width = shared incidents; colour = community or node type."""
    nodes = {n["id"]: n for n in ego["nodes"]}
    centre = next((n["id"] for n in ego["nodes"] if n["centre"]), None)
    fig = go.Figure()
    top = max((e["weight"] for e in ego["edges"]), default=1)
    for e in ego["edges"]:
        a, b = nodes[e["source"]], nodes[e["target"]]
        spoke = centre in (e["source"], e["target"])
        if not spoke and e["weight"] < 0.25 * top:
            continue  # neighbour-neighbour links only when strong, to keep the picture readable
        fig.add_trace(
            go.Scatter(
                x=[a["x"], b["x"]],
                y=[a["y"], b["y"]],
                mode="lines",
                line={
                    "color": "#9FB3C8" if spoke else "#E3E8EE",
                    "width": (1 + 6 * e["weight"] / top) if spoke else 0.8,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )
    key = (
        (lambda n: "no community" if n["community"] is None else f"Community {n['community']}")
        if colour_by == "community"
        else (lambda n: n["type_name"])
    )
    groups = sorted({key(n) for n in nodes.values()})
    palette = {g: CATEGORICAL[i % len(CATEGORICAL)] for i, g in enumerate(groups)}
    biggest = max((n["shared"] for n in nodes.values() if not n["centre"]), default=1)
    for g in groups:
        members = [n for n in nodes.values() if key(n) == g]
        fig.add_trace(
            go.Scatter(
                x=[n["x"] for n in members],
                y=[n["y"] for n in members],
                mode="markers+text",
                name=g,
                text=[n["label"] for n in members],
                textposition=[
                    "bottom center"
                    if n["centre"]
                    else ("middle right" if n["x"] >= 0 else "middle left")
                    for n in members
                ],
                textfont={"size": [16 if n["centre"] else 12 for n in members]},
                marker={
                    "size": [
                        38 if n["centre"] else 10 + 22 * (n["shared"] / biggest) ** 0.5
                        for n in members
                    ],
                    "color": palette[g],
                    "line": {
                        "width": [3 if n["centre"] else 1 for n in members],
                        "color": "#1B2430",
                    },
                    "symbol": [TYPE_SYMBOL_ENTITY.get(n["type"], "circle") for n in members],
                },
                customdata=[[n["type_name"], n["shared"]] for n in members],
                hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>%{customdata[1]} shared incidents<extra></extra>",
            )
        )
    fig.update_xaxes(visible=False, range=[-2.5, 2.5])
    fig.update_yaxes(visible=False, range=[-1.2, 1.2])
    fig.update_layout(
        legend={
            "title": {
                "text": "Colour = "
                + ("network community" if colour_by == "community" else "node type")
            }
        }
    )
    return _layout(fig, "", 640)


TYPE_SYMBOL_ENTITY = {
    "organisation": "circle",
    "technology_system": "square",
    "population_group": "triangle-up",
    "generic_actor_group": "diamond",
    "abstract_value": "cross",
}


def horizontal_bars(
    values: pd.Series, title: str, xlabel: str, colour: str = PALETTE["blue"], pct: bool = False
) -> go.Figure:
    v = values.sort_values()
    fig = go.Figure(
        go.Bar(
            x=v.values * (100 if pct else 1),
            y=[short(str(i)) for i in v.index],
            orientation="h",
            marker_color=colour,
            text=[f"{x * 100:.0f}%" if pct else f"{x:,.0f}" for x in v.values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_xaxes(title=xlabel)
    return _layout(fig, title, 60 + 34 * len(v))


def kappa_forest(mappings: pd.DataFrame) -> go.Figure:
    labels = {
        "discrimination_vs_protected_characteristic": "Discrimination vs CSET protected characteristic",
        "discrimination_vs_harm_distribution_basis": "Discrimination vs CSET harm distribution",
        "safety_vs_tangible_harm": "AI safety vs CSET tangible harm",
        "safety_vs_gmf_technical_failure": "AI safety vs GMF technical failure",
        "misinformation_or_misuse_vs_detrimental_content": "Misinformation/misuse vs CSET detrimental content",
    }
    m = mappings.iloc[::-1]
    fig = go.Figure(
        go.Scatter(
            x=m["kappa"],
            y=[f"{labels.get(k, k)} (n={n})" for k, n in zip(m["mapping"], m["n"], strict=True)],
            mode="markers",
            marker={"size": 13, "color": PALETTE["blue"]},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": m["kappa_high"] - m["kappa"],
                "arrayminus": m["kappa"] - m["kappa_low"],
                "thickness": 2,
            },
            hovertemplate="kappa %{x:.2f}<extra></extra>",
        )
    )
    for x in (0.2, 0.4, 0.6, 0.8):
        fig.add_vline(x=x, line={"color": GRID, "width": 1})
    for x, text in (
        (0.1, "slight"),
        (0.3, "fair"),
        (0.5, "moderate"),
        (0.7, "substantial"),
        (0.9, "almost<br>perfect"),
    ):
        fig.add_annotation(
            x=x,
            y=1.08,
            yref="paper",
            text=text,
            showarrow=False,
            font={"size": 11, "color": "#8A94A3"},
        )
    fig.add_vline(x=0, line={"color": "#8A94A3", "dash": "dash"})
    fig.update_xaxes(
        title="Cohen's kappa with 95% CI (0 = chance, 1 = perfect)", range=[-0.35, 1.0]
    )
    return _layout(fig, "", 360)


def pair_bars(similar: dict[str, Any], dissimilar: dict[str, Any]) -> go.Figure:
    names = [
        "Editor-marked similar pairs",
        "Random pairs (same years)",
        "Editor-marked dissimilar pairs",
    ]
    vals = [
        similar["same_domain_rate"],
        similar["baseline_rate"],
        dissimilar.get("same_domain_rate", float("nan")),
    ]
    lo = [similar["ci_low"], similar["baseline_low"], dissimilar.get("ci_low", float("nan"))]
    hi = [similar["ci_high"], similar["baseline_high"], dissimilar.get("ci_high", float("nan"))]
    fig = go.Figure(
        go.Bar(
            x=[100 * v for v in vals],
            y=names,
            orientation="h",
            marker_color=[PALETTE["blue"], PALETTE["grey"], PALETTE["orange"]],
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [100 * (h - v) for h, v in zip(hi, vals, strict=True)],
                "arrayminus": [100 * (v - low) for v, low in zip(vals, lo, strict=True)],
            },
            text=[f"{100 * v:.0f}%" for v in vals],
            textposition="inside",
            insidetextanchor="start",
            insidetextfont={"color": "white", "size": 15},
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Pairs whose two incidents have the same MIT domain (%)", range=[0, 100])
    return _layout(fig, "", 300)


def label_confusion(joint: pd.DataFrame) -> go.Figure:
    """Estimated incidents whose given MIT label is likely another domain (off-diagonal only)."""
    z = joint.to_numpy(dtype=float).copy()
    for i in range(len(z)):
        z[i, i] = float("nan")
    names = [short(c) for c in joint.columns]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=names,
            y=[short(c) for c in joint.index],
            colorscale="Oranges",
            text=[["" if v != v else f"{v:.0f}" for v in row] for row in z],
            texttemplate="%{text}",
            hovertemplate="given %{y}<br>likely %{x}<br>%{z:.0f} incidents<extra></extra>",
            colorbar={"title": "incidents"},
        )
    )
    fig.update_yaxes(autorange="reversed", title="Given MIT label")
    fig.update_xaxes(title="Likely correct domain (confident learning)")
    return _layout(fig, "", 460)


def retrieval_comparison(summary: pd.DataFrame, labels: dict[str, str]) -> go.Figure:
    s = summary.iloc[::-1]
    fig = go.Figure(
        go.Scatter(
            x=s["rr"],
            y=[labels.get(m, m) for m in s["method"]],
            mode="markers",
            marker={"size": 12, "color": PALETTE["blue"]},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": s["rr_high"] - s["rr"],
                "arrayminus": s["rr"] - s["rr_low"],
            },
        )
    )
    fig.update_xaxes(title="Mean reciprocal rank (bootstrap 95% CI)", range=[0.3, 0.6])
    return _layout(fig, "", 300)
