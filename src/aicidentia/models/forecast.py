"""Rolling-origin evaluation of quarterly count forecasts: naive vs exponential smoothing vs ARIMA (MASE).

For every series and every forecast origin t (from `initial` quarters up to n - h), each model is fitted
on the first t quarters only and forecasts 1..h quarters ahead. The error is scaled (MASE) by the mean
absolute one-step naive error *in the training window*, so MASE < 1 means better than the naive
one-step-ahead forecast in-sample. No forecast beyond the last complete quarter is produced here:
this is an evaluation of whether forecasting is justified, not a forecast.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
MODELS = ("naive", "ets", "arima")


def quarterly_counts(
    df: pd.DataFrame, group_col: str | None, first_year: int, last_year: int
) -> pd.DataFrame:
    """Quarterly incident counts, one column per group (or 'total'); index = '2015Q1' ... '2025Q4'."""
    index = [f"{y}Q{q}" for y in range(first_year, last_year + 1) for q in range(1, 5)]
    sub = df[df["year"].between(first_year, last_year)]
    if group_col is None:
        wide = sub.groupby("year_quarter").size().rename("total").to_frame()
    else:
        wide = sub.groupby(["year_quarter", group_col]).size().unstack(fill_value=0)
    return wide.reindex(index, fill_value=0)


def forecast_naive(train: np.ndarray, h: int) -> np.ndarray:
    """Repeat the last observed value."""
    return np.repeat(train[-1], h).astype(float)


def forecast_ets(train: np.ndarray, h: int) -> np.ndarray:
    """Exponential smoothing: the better (by AICc) of simple (A,N,N) and damped-trend (A,Ad,N) ETS."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    best, best_aicc = None, np.inf
    for trend, damped in ((None, False), ("add", True)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ETSModel(train, error="add", trend=trend, damped_trend=damped).fit(disp=False)
            if np.isfinite(fit.aicc) and fit.aicc < best_aicc:
                best, best_aicc = fit, fit.aicc
        except Exception as exc:  # a failing candidate is skipped; naive is the fallback
            logger.debug("ETS candidate failed: %s", exc)
    if best is None:
        return forecast_naive(train, h)
    return np.clip(np.asarray(best.forecast(h)), 0, None)


def forecast_arima(train: np.ndarray, h: int, orders: list[list[int]]) -> np.ndarray:
    """ARIMA with the order (from a small grid) that minimises AICc on the training window."""
    from statsmodels.tsa.arima.model import ARIMA

    best, best_aicc = None, np.inf
    for p, d, q in orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(train, order=(p, d, q), trend="c" if d == 0 else "n").fit()
            if np.isfinite(fit.aicc) and fit.aicc < best_aicc:
                best, best_aicc = fit, fit.aicc
        except Exception as exc:
            logger.debug("ARIMA(%d,%d,%d) failed: %s", p, d, q, exc)
    if best is None:
        return forecast_naive(train, h)
    return np.clip(np.asarray(best.forecast(h)), 0, None)


def rolling_origin(
    series: np.ndarray, initial: int, horizon: int, orders: list[list[int]]
) -> pd.DataFrame:
    """Forecast errors for every origin and horizon: columns origin, h, model, actual, forecast, scale."""
    fns: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
        "naive": forecast_naive,
        "ets": forecast_ets,
        "arima": lambda t, h: forecast_arima(t, h, orders),
    }
    rows = []
    for origin in range(initial, len(series) - horizon + 1):
        train = series[:origin]
        scale = float(np.mean(np.abs(np.diff(train))))
        actual = series[origin : origin + horizon]
        for name, fn in fns.items():
            pred = fn(train, horizon)
            rows += [
                {
                    "origin": origin,
                    "h": k + 1,
                    "model": name,
                    "actual": float(actual[k]),
                    "forecast": float(pred[k]),
                    "scale": scale,
                }
                for k in range(horizon)
            ]
    return pd.DataFrame(rows)


def mase(errors: pd.DataFrame) -> float:
    """Mean absolute scaled error over the rows of a `rolling_origin` frame (rows with zero scale are ignored)."""
    valid = errors[errors["scale"] > 0]
    if valid.empty:
        return float("nan")
    return float((np.abs(valid["actual"] - valid["forecast"]) / valid["scale"]).mean())


def evaluate_series(
    wide: pd.DataFrame, initial: int, horizon: int, orders: list[list[int]], min_mean: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MASE per series x model x horizon for every column of `wide` with mean count >= `min_mean`.

    Returns (long table of MASE by series/model/horizon, table of skipped series with the reason).
    """
    rows, skipped = [], []
    for name in wide.columns:
        y = wide[name].to_numpy(dtype=float)
        if y.mean() < min_mean:
            skipped.append(
                {"series": name, "reason": f"mean quarterly count {y.mean():.2f} < {min_mean}"}
            )
            continue
        errors = rolling_origin(y, initial, horizon, orders)
        if (errors["scale"] > 0).sum() == 0:
            skipped.append({"series": name, "reason": "constant training series"})
            continue
        for (model, h), group in errors.groupby(["model", "h"]):
            rows.append(
                {
                    "series": name,
                    "model": model,
                    "h": int(h),
                    "mase": mase(group),
                    "origins": int(group["origin"].nunique()),
                }
            )
        for model, group in errors.groupby("model"):
            rows.append(
                {
                    "series": name,
                    "model": model,
                    "h": 0,
                    "mase": mase(group),
                    "origins": int(group["origin"].nunique()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(skipped, columns=["series", "reason"])


def summarise_mase(long: pd.DataFrame) -> pd.DataFrame:
    """Per model: mean MASE over series (all horizons, h=1, h=4), and the number of series where it is best."""
    allh = long[long["h"] == 0].pivot(index="series", columns="model", values="mase")
    best = allh.idxmin(axis=1).value_counts()
    rows = []
    h_max = int(long["h"].max())
    for model in allh.columns:
        rows.append(
            {
                "model": model,
                "mean_mase_all_horizons": float(allh[model].mean()),
                "median_mase_all_horizons": float(allh[model].median()),
                "mean_mase_h1": float(
                    long[(long["model"] == model) & (long["h"] == 1)]["mase"].mean()
                ),
                f"mean_mase_h{h_max}": float(
                    long[(long["model"] == model) & (long["h"] == h_max)]["mase"].mean()
                ),
                "series_where_best": int(best.get(model, 0)),
                "series_total": int(len(allh)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_mase_all_horizons").reset_index(drop=True)
