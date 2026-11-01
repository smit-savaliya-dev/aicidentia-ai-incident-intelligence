"""Phase 4 orchestration: series, tests, reporting-bias analysis, forecasting evaluation, artefacts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from aicidentia import registry
from aicidentia.config import PROJECT_ROOT, load_config
from aicidentia.models import forecast, themes, trends
from aicidentia.viz import plots, style

logger = logging.getLogger(__name__)
PRODUCER = "aicidentia.models.trend_outputs.run_trends"
SOURCE = "data/processed/incident_topics.parquet"
BASES = {
    "counts": "counts",
    "share": "share",
    "wshare": "report-weighted share",
    "share_late": "share, later window",
}
JUSTIFIED_IMPROVEMENT = (
    0.9  # a model must cut mean MASE to <= 90% of the naive forecast to justify forecasting
)


@dataclass
class TrendResults:
    """Everything Phase 4 produces."""

    info: dict[str, Any]
    years: list[int]
    late_years: list[int]
    partial_year: int
    yearly: dict[str, trends.YearlySeries] = field(default_factory=dict)
    yearly_late: dict[str, trends.YearlySeries] = field(default_factory=dict)
    partial_share: dict[str, pd.Series] = field(default_factory=dict)
    tests: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    classification: dict[str, pd.DataFrame] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    forecast_long: pd.DataFrame | None = None
    forecast_summary: pd.DataFrame | None = None
    forecast_skipped: pd.DataFrame | None = None


def oecd_files() -> list[str]:
    """Files in data/external/ other than placeholders (an OECD AIM export would go here)."""
    folder = PROJECT_ROOT / load_config()["paths"]["external"]
    return sorted(p.name for p in folder.glob("*") if p.is_file() and p.name != ".gitkeep")


def _tests(
    ys: trends.YearlySeries, late: trends.YearlySeries, cfg: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    kw = {"alpha": cfg["alpha"], "recent_n": cfg["recent_years"]}
    return {
        "counts": trends.trend_table(ys.counts.astype(float), **kw),
        "share": trends.trend_table(ys.share, **kw),
        "wshare": trends.trend_table(ys.weighted_share, **kw),
        "share_late": trends.trend_table(late.share, **kw),
    }


def _long_yearly(
    scope: str,
    ys: trends.YearlySeries,
    partial_year: int,
    partial: pd.Series,
    partial_counts: pd.Series,
) -> pd.DataFrame:
    frames = []
    for group in ys.counts.columns:
        frame = pd.DataFrame(
            {
                "scope": scope,
                "group": group,
                "year": ys.counts.index,
                "count": ys.counts[group].to_numpy(),
                "total": ys.totals.to_numpy(),
                "share": ys.share[group].to_numpy(),
                "weighted_share": ys.weighted_share[group].to_numpy(),
                "tested": True,
                "partial_year": False,
            }
        )
        extra = pd.DataFrame(
            {
                "scope": scope,
                "group": group,
                "year": [partial_year],
                "count": [int(partial_counts.get(group, 0))],
                "total": [int(partial_counts.sum())],
                "share": [partial.get(group, np.nan)],
                "weighted_share": [np.nan],
                "tested": False,
                "partial_year": True,
            }
        )
        frames += [frame, extra]
    return pd.concat(frames, ignore_index=True)


def run_trends() -> TrendResults:
    """Compute every Phase 4 analysis from the saved topic assignments and your labels."""
    cfg = load_config()["trends"]
    window = registry.get("audit.year_range_recommendation")
    partial_year = int(registry.get("audit.partial_year"))
    years = list(range(window["start"], window["end"] + 1))
    late_years = list(range(cfg["sensitivity_start"], window["end"] + 1))
    tf = themes.load_theme_frame()
    data = tf.data
    res = TrendResults(tf.info, years, late_years, partial_year)

    partial_rows = data[data["year"] == partial_year]
    labelled = data[data["has_mit"]]
    for scope, frame, col in (("themes", data, "theme"), ("domains", labelled, "mit_risk_domain")):
        res.yearly[scope] = trends.build_yearly(frame, col, years)
        res.yearly_late[scope] = trends.build_yearly(frame, col, late_years)
        part = frame[frame["year"] == partial_year]
        counts = part.groupby(col).size()
        res.partial_share[scope] = (counts / counts.sum()).rename(partial_year)
        res.tests[scope] = _tests(res.yearly[scope], res.yearly_late[scope], cfg)
        res.classification[scope] = trends.classification_matrix(
            {BASES[k]: v for k, v in res.tests[scope].items()},
            core=[BASES["counts"], BASES["share"], BASES["wshare"]],
            late=BASES["share_late"],
            reference=BASES["share"],
        )
        res.tables[f"yearly_{scope}"] = _long_yearly(
            scope, res.yearly[scope], partial_year, res.partial_share[scope], counts
        )
        long_tests = pd.concat(
            [t.assign(basis=BASES[k]) for k, t in res.tests[scope].items()], ignore_index=True
        )
        res.tables[f"trends_{scope}"] = long_tests

    res.tables["reports_by_year"] = trends.reports_by_year(data, [*years, partial_year])
    coverage = (
        labelled.groupby("year")
        .size()
        .div(data.groupby("year").size())
        .reindex([*years, partial_year])
    )
    res.tables["mit_coverage"] = (
        coverage.rename("mit_labelled_share").rename_axis("year").reset_index()
    )

    # ---- forecasting (quarterly counts, complete years only)
    fc = cfg["forecast"]
    wide = pd.concat(
        [
            forecast.quarterly_counts(data, None, years[0], years[-1]),
            forecast.quarterly_counts(data, "theme", years[0], years[-1]),
        ],
        axis=1,
    )
    long, skipped = forecast.evaluate_series(
        wide, fc["initial_quarters"], fc["horizon"], fc["arima_orders"], fc["min_mean_count"]
    )
    summary = forecast.summarise_mase(long)
    res.forecast_long, res.forecast_summary, res.forecast_skipped = long, summary, skipped
    naive = float(summary.set_index("model").loc["naive", "mean_mase_all_horizons"])
    best = summary[summary["model"] != "naive"].iloc[0]
    res.tables["quarterly_counts"] = wide.reset_index(names="quarter")

    # ---- facts
    sh, ct, ws = (
        res.tests["themes"]["share"],
        res.tests["themes"]["counts"],
        res.tests["themes"]["wshare"],
    )
    cls = res.classification["themes"]
    res.facts = {
        "window": [years[0], years[-1]],
        "late_window": [late_years[0], late_years[-1]],
        "partial_year": partial_year,
        "provisional": bool(tf.info["provisional"]),
        "n_themes": int(tf.info["n_themes"]),
        "n_themes_tested": int(len(sh)),
        "emerging_themes": sh.loc[sh["emerging"], "group"].tolist(),
        "increasing_share": int((sh["direction"] == "increasing").sum()),
        "decreasing_share": int((sh["direction"] == "decreasing").sum()),
        "increasing_counts": int((ct["direction"] == "increasing").sum()),
        "decreasing_counts": int((ct["direction"] == "decreasing").sum()),
        "increasing_wshare": int((ws["direction"] == "increasing").sum()),
        "decreasing_wshare": int((ws["direction"] == "decreasing").sum()),
        "themes_with_disagreeing_bases": cls.index[~cls["agree"]].tolist(),
        "n_themes_disagree": int((~cls["agree"]).sum()),
        "themes_window_sensitive": cls.index[cls["window_sensitive"]].tolist(),
        "n_themes_window_sensitive": int(cls["window_sensitive"].sum()),
        "n_domains_disagree": int((~res.classification["domains"]["agree"]).sum()),
        "domains_emerging": res.tests["domains"]["share"].query("emerging")["group"].tolist(),
        "domains_increasing_share": int(
            (res.tests["domains"]["share"]["direction"] == "increasing").sum()
        ),
        "mean_reports_first_window_year": float(
            res.tables["reports_by_year"].iloc[0]["mean_reports"]
        ),
        "mean_reports_last_window_year": float(
            res.tables["reports_by_year"].iloc[len(years) - 1]["mean_reports"]
        ),
        "forecast_series_scored": int(long["series"].nunique()),
        "forecast_series_skipped": skipped["series"].tolist(),
        "forecast_naive_mase": naive,
        "forecast_best_model": str(best["model"]),
        "forecast_best_mase": float(best["mean_mase_all_horizons"]),
        "forecast_justified": bool(best["mean_mase_all_horizons"] <= JUSTIFIED_IMPROVEMENT * naive),
        "oecd_aim_files": oecd_files(),
        "partial_year_incidents": int(len(partial_rows)),
    }
    if res.facts["oecd_aim_files"]:
        logger.warning(
            "Files found in data/external/: OECD replication is not implemented for their format; ask before relying on it."
        )
    return res


# --------------------------------------------------------------------------------------
# Artefacts
# --------------------------------------------------------------------------------------


def register(res: TrendResults) -> None:
    """Record Phase 4 facts and compact tables under `rq1b.*`."""
    for key, value in res.facts.items():
        registry.record(f"rq1b.{key}", value, SOURCE, PRODUCER)
    registry.record(
        "rq1b.themes_share_trends",
        res.tests["themes"]["share"][["group", "sen_slope", "p", "p_bh", "direction", "emerging"]],
        SOURCE,
        PRODUCER,
    )
    registry.record(
        "rq1b.domains_share_trends",
        res.tests["domains"]["share"][["group", "sen_slope", "p", "p_bh", "direction", "emerging"]],
        SOURCE,
        PRODUCER,
    )
    registry.record(
        "rq1b.theme_classification",
        res.classification["themes"].reset_index(names="theme"),
        SOURCE,
        PRODUCER,
    )
    registry.record("rq1b.forecast_summary", res.forecast_summary, SOURCE, PRODUCER)
    registry.record(
        "rq1b.labels",
        {
            k: res.info[k]
            for k in (
                "n_topics",
                "n_answered",
                "n_themes",
                "provisional",
                "merged_themes",
                "model_key",
            )
        },
        SOURCE,
        PRODUCER,
    )


def save_tables(res: TrendResults) -> None:
    """reports/tables/rq1b_*.{csv,md}."""
    for name, df in res.tables.items():
        out = df.copy()
        for col in out.select_dtypes("float").columns:
            out[col] = out[col].round(4)
        style.save_table(out, f"rq1b_{name}")
    cls = res.classification["themes"].reset_index(names="theme")
    style.save_table(cls, "rq1b_classification_themes")
    style.save_table(
        res.classification["domains"].reset_index(names="domain"), "rq1b_classification_domains"
    )
    style.save_table(res.forecast_long.round(4), "rq1b_forecast_mase_long")
    style.save_table(res.forecast_summary.round(4), "rq1b_forecast_summary")
    style.save_table(res.forecast_skipped, "rq1b_forecast_skipped")


def make_figures(res: TrendResults) -> dict[str, Any]:
    """Four figures: heatmap, small multiples, classification matrix, forecast error."""
    prov = bool(res.facts["provisional"])
    sh = res.tests["themes"]["share"]
    emerging = res.facts["emerging_themes"]
    shown = emerging or sh.head(4)["group"].tolist()
    title = ("PROVISIONAL theme names. " if prov else "") + (
        "Emerging themes (share of incidents per year; dashed = Sen's slope; hollow = partial year)"
        if emerging
        else "No theme met the 'emerging' definition; four largest positive slopes shown"
    )
    cls_t, cls_d = res.classification["themes"], res.classification["domains"]
    cls = pd.concat([cls_t, cls_d])
    figs = {
        "fig_rq1b_1_theme_heatmap": plots.plot_theme_heatmap(
            res.yearly["themes"].share, sh, res.partial_share["themes"], prov
        ),
        "fig_rq1b_2_emerging_themes": plots.plot_emerging_small_multiples(
            res.yearly["themes"].share, sh, shown, res.partial_share["themes"], title
        ),
        "fig_rq1b_3_trend_by_measure": plots.plot_classification_matrix(
            cls, list(BASES.values()), separator_after=len(cls_t), provisional=prov
        ),
        "fig_rq1b_4_forecast_mase": plots.plot_forecast_mase(
            res.forecast_summary, int(load_config()["trends"]["forecast"]["horizon"])
        ),
    }
    for name, fig in figs.items():
        style.save_figure(fig, name, str(load_config()["snapshot_date"]))
    return figs


def write_all(res: TrendResults) -> None:
    """Registry, tables, figures and docs/rq1_trends.md."""
    from aicidentia.models import trend_docs

    register(res)
    save_tables(res)
    make_figures(res)
    trend_docs.write_trend_doc(res, PROJECT_ROOT / "docs" / "rq1_trends.md")
