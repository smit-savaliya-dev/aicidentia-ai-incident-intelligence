"""Cross-question analyses that make the story (docs/decisions.md D9-D12).

1. What changed after ChatGPT: theme and risk-domain shares before vs after 30 Nov 2022 (chi-square test of
   independence, adjusted standardised residuals, Cramer's V).
2. Risk priority matrix: trend (Sen's slope of each theme's yearly share) vs severity (share of the theme's
   CSETv1-annotated incidents with tangible harm, injuries or deaths); themes with fewer than 30 annotated
   incidents are marked as not supported.
3. Repeated failures: how many recent incidents have an editor-linked earlier precedent, and how often retrieval
   finds it (computed in Phase 7, `retrieval_outputs`).
4. Synthesis: which emerging themes concentrate in which network communities, and which (alleged) organisations
   sit at their centre.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

CHATGPT_DATE = pd.Timestamp("2022-11-30")
MIN_CSET_PER_THEME = 30


# --------------------------------------------------------------------------------------
# 1. Before / after ChatGPT
# --------------------------------------------------------------------------------------


def period_of(dates: pd.Series, cutoff: pd.Timestamp = CHATGPT_DATE) -> pd.Series:
    return pd.Series(np.where(dates < cutoff, "before", "after"), index=dates.index, name="period")


def chi_square_shift(groups: pd.Series, period: pd.Series) -> dict[str, Any]:
    """Chi-square test of independence (period x group), adjusted standardised residuals and Cramer's V.

    Adjusted residual r_ij = (O - E) / sqrt(E (1 - row_i/N)(1 - col_j/N)); |r| > 1.96 marks a cell that
    deviates from independence at about the 5% level (Agresti, 2007)."""
    table = pd.crosstab(groups, period).reindex(columns=["before", "after"], fill_value=0)
    table = table[table.sum(axis=1) > 0]
    chi2, p, dof, expected = stats.chi2_contingency(table.to_numpy(), correction=False)
    n = table.to_numpy().sum()
    row = table.sum(axis=1).to_numpy()[:, None] / n
    col = table.sum(axis=0).to_numpy()[None, :] / n
    resid = (table.to_numpy() - expected) / np.sqrt(expected * (1 - row) * (1 - col))
    v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    shares = table / table.sum(axis=0)
    out = pd.DataFrame(
        {
            "group": table.index,
            "n_before": table["before"].to_numpy(),
            "n_after": table["after"].to_numpy(),
            "share_before": shares["before"].to_numpy(),
            "share_after": shares["after"].to_numpy(),
            "change_pp": (shares["after"] - shares["before"]).to_numpy() * 100,
            "adj_residual_after": resid[:, 1],
            "min_expected": expected.min(axis=1),
        }
    ).sort_values("adj_residual_after", ascending=False)
    out["driver"] = np.where(
        out["adj_residual_after"] > 1.96,
        "over-represented after",
        np.where(out["adj_residual_after"] < -1.96, "under-represented after", ""),
    )
    return {
        "table": out.reset_index(drop=True),
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "cramers_v": v,
        "n": int(n),
        "n_before": int(table["before"].sum()),
        "n_after": int(table["after"].sum()),
        "min_expected": float(expected.min()),
    }


# --------------------------------------------------------------------------------------
# 2. Risk priority matrix
# --------------------------------------------------------------------------------------


def severe(incidents: pd.DataFrame) -> pd.Series:
    """CSETv1 severity flag: tangible harm definitively occurred, or injuries > 0, or lives lost > 0.
    NaN for incidents without CSETv1 annotation."""
    tangible = (
        incidents["cset_tangible_harm"]
        .astype("string")
        .str.startswith("tangible harm definitively")
    )
    injuries = pd.to_numeric(incidents["cset_injuries"], errors="coerce").fillna(0) > 0
    deaths = pd.to_numeric(incidents["cset_lives_lost"], errors="coerce").fillna(0) > 0
    flag = (tangible.fillna(False) | injuries | deaths).astype(float)
    return flag.where(incidents["has_cset"])


def priority_matrix(
    incidents: pd.DataFrame,
    themes: pd.DataFrame,
    trends: pd.DataFrame,
    min_cset: int = MIN_CSET_PER_THEME,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One row per theme: trend (Sen's slope of yearly share, percentage points per year), q-value, emerging
    flag, severity (share severe among CSET-annotated incidents), its n, total incidents, quadrant."""
    df = incidents.merge(themes[["incident_id", "theme"]], on="incident_id")
    df["severe"] = severe(df)
    per = df.groupby("theme").agg(
        incidents=("incident_id", "size"),
        cset_n=("has_cset", "sum"),
        severe_n=("severe", "sum"),
    )
    per["severity"] = per["severe_n"] / per["cset_n"].where(per["cset_n"] > 0)
    tr = trends[trends["basis"] == "share"].set_index("group")
    per["trend_pp_per_year"] = tr["sen_slope"].reindex(per.index) * 100
    per["q_value"] = tr["p_bh"].reindex(per.index)
    per["emerging"] = tr["emerging"].reindex(per.index).fillna(False).astype(bool)
    per["supported"] = per["cset_n"] >= min_cset
    pool = df[df["has_cset"]]
    overall = float(pool["severe"].mean())
    per["quadrant"] = [
        _quadrant(t, s, overall) if ok else "not enough CSET data"
        for t, s, ok in zip(
            per["trend_pp_per_year"], per["severity"], per["supported"], strict=True
        )
    ]
    facts = {
        "severity_overall": overall,
        "cset_incidents": int(len(pool)),
        "cset_year_max": int(pool["year"].max()),
        "themes_supported": int(per["supported"].sum()),
        "themes_total": int(len(per)),
        "min_cset_per_theme": min_cset,
    }
    return per.reset_index().sort_values("trend_pp_per_year", ascending=False), facts


def _quadrant(trend: float, severity: float, threshold: float) -> str:
    emerging = trend > 0
    serious = severity >= threshold
    if emerging and serious:
        return "Emerging & severe"
    if emerging:
        return "Emerging"
    if serious:
        return "Established & severe"
    return "Established"


# --------------------------------------------------------------------------------------
# 4. Synthesis: themes x communities x domains
# --------------------------------------------------------------------------------------


def theme_community_concentration(
    themes: pd.DataFrame,
    incident_entities: pd.DataFrame,
    communities: pd.DataFrame,
    min_incidents: int = 10,
) -> pd.DataFrame:
    """Share of a theme's incidents that involve at least one entity of each network community, and the lift
    over the share of all incidents involving that community."""
    ent = incident_entities.merge(
        communities[["node", "community"]], left_on="entity_id", right_on="node"
    )
    touch = ent[["incident_id", "community"]].drop_duplicates()
    all_n = themes["incident_id"].nunique()
    base = touch.groupby("community")["incident_id"].nunique() / all_n
    tc = touch.merge(themes[["incident_id", "theme"]], on="incident_id")
    counts = tc.groupby(["theme", "community"])["incident_id"].nunique().rename("incidents")
    sizes = themes.groupby("theme")["incident_id"].nunique()
    out = counts.reset_index()
    out["theme_incidents"] = out["theme"].map(sizes)
    out["share_of_theme"] = out["incidents"] / out["theme_incidents"]
    out["share_of_all"] = out["community"].map(base)
    out["lift"] = out["share_of_theme"] / out["share_of_all"]
    out = out[out["incidents"] >= min_incidents]
    return out.sort_values(["theme", "lift"], ascending=[True, False]).reset_index(drop=True)


def central_organisations(
    theme: str,
    community: int,
    themes: pd.DataFrame,
    incident_entities: pd.DataFrame,
    centrality: pd.DataFrame,
    top: int = 3,
) -> list[str]:
    """Organisations of `community` that appear in the theme's incidents, ranked by PageRank (alleged roles)."""
    ids = set(themes.loc[themes["theme"] == theme, "incident_id"])
    present = set(incident_entities.loc[incident_entities["incident_id"].isin(ids), "entity_id"])
    orgs = centrality[
        (centrality["type"] == "organisation")
        & (centrality["community"] == community)
        & centrality["node"].isin(present)
    ].sort_values(["pagerank", "label"], ascending=[False, True])
    return orgs["label"].head(top).tolist()


def synthesis(
    themes: pd.DataFrame,
    incidents: pd.DataFrame,
    incident_entities: pd.DataFrame,
    communities: pd.DataFrame,
    centrality: pd.DataFrame,
    emerging: list[str],
) -> pd.DataFrame:
    """One row per emerging theme: most concentrated community (lift), its share of the theme, the alleged
    central organisations, and the theme's dominant MIT domain with its share and coverage."""
    conc = theme_community_concentration(themes, incident_entities, communities)
    rows = []
    merged = themes.merge(incidents[["incident_id", "mit_risk_domain"]], on="incident_id")
    for theme in emerging:
        sub = conc[conc["theme"] == theme]
        labelled = merged[(merged["theme"] == theme) & merged["mit_risk_domain"].notna()]
        dom = labelled["mit_risk_domain"].value_counts()
        n_theme = int((themes["theme"] == theme).sum())
        if sub.empty:
            continue
        top = sub.iloc[0]
        rows.append(
            {
                "theme": theme,
                "incidents": n_theme,
                "community": int(top["community"]),
                "share_in_community": float(top["share_of_theme"]),
                "lift": float(top["lift"]),
                "alleged_central_organisations": "; ".join(
                    central_organisations(
                        theme, int(top["community"]), themes, incident_entities, centrality
                    )
                ),
                "dominant_mit_domain": dom.index[0] if len(dom) else "",
                "dominant_domain_share": float(dom.iloc[0] / dom.sum()) if len(dom) else np.nan,
                "mit_coverage": float(len(labelled) / n_theme) if n_theme else np.nan,
            }
        )
    return pd.DataFrame(rows)
