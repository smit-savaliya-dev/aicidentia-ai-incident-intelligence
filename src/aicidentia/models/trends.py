"""RQ1b - temporal trends of risk themes: yearly series, Mann-Kendall tests, Sen's slope, BH correction.

Conventions
-----------
* Only *complete* years are tested (the window comes from the Phase 1 audit).
* `share` = incidents of the group in the year / all incidents in the year (for MIT domains, the
  denominator is MIT-labelled incidents only). `weighted_share` weights each incident by its number
  of non-stub linked reports (a proxy for media attention).
* A series is *increasing* / *decreasing* if its Benjamini-Hochberg-adjusted Mann-Kendall p-value
  is below alpha (adjusted within the family of series tested on the same basis).
* A group is *emerging* if it is increasing, its Sen slope is positive, and the mean of its last
  `recent_n` complete years exceeds the mean over the whole window.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy.stats import theilslopes
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


@dataclass
class YearlySeries:
    """Yearly series for every group on every basis; each table is years x groups."""

    counts: pd.DataFrame
    share: pd.DataFrame
    weighted_share: pd.DataFrame
    totals: pd.Series
    weighted_totals: pd.Series


def build_yearly(
    df: pd.DataFrame, group_col: str, years: list[int], weight_col: str = "report_count_clean"
) -> YearlySeries:
    """Counts, shares and report-weighted shares per group and year (rows of `df` define the denominator)."""
    sub = df[df["year"].isin(years)]
    counts = (
        sub.groupby(["year", group_col]).size().unstack(fill_value=0).reindex(years, fill_value=0)
    )
    weighted = (
        sub.groupby(["year", group_col])[weight_col]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=years, columns=counts.columns, fill_value=0)
    )
    totals = counts.sum(axis=1)
    wtotals = weighted.sum(axis=1)
    share = counts.div(totals.replace(0, np.nan), axis=0)
    wshare = weighted.div(wtotals.replace(0, np.nan), axis=0)
    return YearlySeries(counts, share, wshare, totals, wtotals)


def topic_counts_by_year(doc_topics: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    """Incidents per topic and year (years x topics)."""
    return doc_topics.groupby([year_col, "topic"]).size().unstack(fill_value=0)


def mann_kendall_test(series: pd.Series | np.ndarray, alpha: float = 0.05) -> dict:
    """Mann-Kendall trend test with Sen's slope.

    Returns the pymannkendall result (trend, p, z, Kendall's tau, Sen's slope) plus a Theil-Sen 95%
    confidence interval for the slope (scipy) and the Hamed-Rao autocorrelation-corrected p-value as a
    robustness check. A constant series gives p = 1 and slope 0.
    """
    y = np.asarray(series, dtype=float)
    if len(y) < 4 or np.allclose(y, y[0]):
        return {
            "n": len(y),
            "trend": "no trend",
            "p": 1.0,
            "z": 0.0,
            "tau": 0.0,
            "sen_slope": 0.0,
            "slope_low": 0.0,
            "slope_high": 0.0,
            "p_hamed_rao": 1.0,
        }
    result = mk.original_test(y, alpha=alpha)
    low, high = theilslopes(y, np.arange(len(y)), 0.95)[2:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            p_hr = float(mk.hamed_rao_modification_test(y, alpha=alpha).p)
        except Exception:  # robustness column only; short or degenerate series may fail
            p_hr = float("nan")
    return {
        "n": len(y),
        "trend": str(result.trend),
        "p": float(result.p),
        "z": float(result.z),
        "tau": float(result.Tau),
        "sen_slope": float(result.slope),
        "slope_low": float(low),
        "slope_high": float(high),
        "p_hamed_rao": p_hr,
    }


def trend_table(
    wide: pd.DataFrame, alpha: float = 0.05, recent_n: int = 2, min_years: int = 4
) -> pd.DataFrame:
    """Test every column of `wide` (years x groups) and correct for multiple testing (Benjamini-Hochberg).

    Columns: n_years, tau, z, p, p_bh, sen_slope (per year) with 95% CI, p_hamed_rao, long-run mean,
    recent mean, `direction` (increasing / decreasing / no trend after BH) and `emerging`.
    """
    rows = []
    for group in wide.columns:
        y = wide[group].fillna(0.0)
        res = mann_kendall_test(y, alpha)
        rows.append(
            {
                "group": group,
                **res,
                "longrun_mean": float(y.mean()),
                "recent_mean": float(y.iloc[-recent_n:].mean()),
            }
        )
    out = pd.DataFrame(rows)
    if len(out) and out["n"].min() >= min_years:
        out["p_bh"] = multipletests(out["p"], alpha=alpha, method="fdr_bh")[1]
    else:
        out["p_bh"] = out["p"]
    significant = out["p_bh"] < alpha
    out["direction"] = np.where(
        significant & (out["sen_slope"] > 0),
        "increasing",
        np.where(significant & (out["sen_slope"] < 0), "decreasing", "no trend"),
    )
    out["recent_above_longrun"] = out["recent_mean"] > out["longrun_mean"]
    out["emerging"] = (out["direction"] == "increasing") & out["recent_above_longrun"]
    return out.sort_values("sen_slope", ascending=False).reset_index(drop=True)


def classification_matrix(
    tables: dict[str, pd.DataFrame], core: list[str], late: str, reference: str
) -> pd.DataFrame:
    """Direction of each group under each basis.

    `agree` is True when the `core` measures (e.g. counts, share, report-weighted share) give the same
    direction: this is the reporting-bias question. `window_sensitive` is True when the `late` window gives a
    different direction from `reference`; that mostly reflects lower power with fewer years, so it is kept
    separate from `agree`.
    """
    frames = [t.set_index("group")["direction"].rename(basis) for basis, t in tables.items()]
    out = pd.concat(frames, axis=1)
    out["agree"] = out[core].nunique(axis=1) == 1
    out["window_sensitive"] = out[late] != out[reference]
    return out


def reports_by_year(
    df: pd.DataFrame, years: list[int], weight_col: str = "report_count_clean"
) -> pd.DataFrame:
    """Mean and median reports per incident by incident year (recent incidents have had less time to be reported on)."""
    sub = df[df["year"].isin(years)]
    out = sub.groupby("year")[weight_col].agg(["mean", "median", "size"]).reindex(years)
    return out.rename(
        columns={"mean": "mean_reports", "median": "median_reports", "size": "incidents"}
    ).reset_index()
