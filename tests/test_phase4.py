"""Phase 4 tests: theme labels, yearly series, Mann-Kendall/BH/emerging logic, classification, forecasting.

Function-level tests use tiny hand-written series with known answers (test fixtures, not analysis data).
"""

import numpy as np
import pandas as pd
import pytest

from aicidentia.data import clean
from aicidentia.models import forecast, themes, trends

YEARS = list(range(2015, 2026))


# ------------------------------------------------------------------ themes and labels
def _labels(final=None):
    final = final or [""] * 3
    return pd.DataFrame(
        {
            "topic_id": ["0", "1", "2"],
            "suggested_label": ["a / b", "c / d", "e / f"],
            "final_label": final,
        }
    )


def test_theme_names_placeholder_vs_confirmed_and_merge():
    unconfirmed = themes.theme_names(_labels())
    assert unconfirmed["theme"].tolist() == ["T0: a / b", "T1: c / d", "T2: e / f"]
    assert not unconfirmed["answered"].any()
    confirmed = themes.theme_names(_labels(["Crashes", "Crashes", "Scams"]))
    assert confirmed["theme"].tolist() == [
        "Crashes",
        "Crashes",
        "Scams",
    ]  # same label = merged theme
    assert confirmed["answered"].all()
    mixed = themes.theme_names(_labels(["Crashes", "  ", ""]))
    assert mixed["answered"].tolist() == [True, False, False]


def test_load_theme_frame_refuses_stale_labels_and_flags_provisional(tmp_path, monkeypatch):
    assignments = pd.DataFrame({"incident_id": [1, 2, 3], "topic_id": [0, 1, 1], "model_key": "K1"})
    incidents = pd.DataFrame(
        {
            "incident_id": [1, 2, 3],
            "date": pd.to_datetime(["2020-01-01"] * 3),
            "year": 2020,
            "quarter": 1,
            "year_quarter": "2020Q1",
            "report_count": 1,
            "report_count_clean": 1,
            "has_mit": True,
            "mit_risk_domain": "x",
        }
    )
    monkeypatch.setattr(
        clean,
        "load_processed",
        lambda name: assignments if name == "incident_topics" else incidents,
    )
    path = tmp_path / "labels.csv"
    _labels().iloc[:2].assign(model_key="K1").to_csv(path, index=False)
    frame = themes.load_theme_frame(path)
    assert (
        frame.info["provisional"] and frame.info["n_answered"] == 0 and frame.info["n_themes"] == 2
    )
    assert set(frame.data["theme"]) == {"T0: a / b", "T1: c / d"}
    # answer one topic: still provisional; answer both with the same label: merged, not provisional
    _labels(["Same", "Same", ""]).iloc[:2].assign(model_key="K1").to_csv(path, index=False)
    merged = themes.load_theme_frame(path)
    assert not merged.info["provisional"] and merged.info["n_themes"] == 1
    assert merged.info["merged_themes"] == {"Same": [0, 1]}
    _labels().iloc[:2].assign(model_key="OTHER").to_csv(path, index=False)
    with pytest.raises(themes.StaleLabelsError):
        themes.load_theme_frame(path)


# ------------------------------------------------------------------ yearly series
def _incidents():
    rows = []
    for year, theme, reports in [
        (2020, "A", 2),
        (2020, "A", 0),
        (2020, "B", 3),
        (2021, "B", 1),
        (2021, "B", 1),
        (2022, "A", 4),
    ]:
        rows.append({"year": year, "theme": theme, "report_count_clean": reports})
    return pd.DataFrame(rows)


def test_build_yearly_counts_shares_and_weighted_shares():
    ys = trends.build_yearly(_incidents(), "theme", [2020, 2021, 2022, 2023])
    assert ys.counts.loc[2020].to_dict() == {"A": 2, "B": 1}
    assert ys.share.loc[2020, "A"] == pytest.approx(2 / 3)
    assert ys.weighted_share.loc[2020, "A"] == pytest.approx(2 / 5)  # reports 2+0 of 5
    assert ys.counts.loc[2023].sum() == 0 and np.isnan(
        ys.share.loc[2023, "A"]
    )  # empty year: shares undefined, not zero
    assert ys.totals.loc[2021] == 2 and ys.weighted_totals.loc[2022] == 4
    assert (ys.counts.index == [2020, 2021, 2022, 2023]).all()


def test_build_yearly_ignores_years_outside_window():
    ys = trends.build_yearly(_incidents(), "theme", [2021])
    assert ys.counts.shape[0] == 1 and ys.counts.loc[2021, "B"] == 2


# ------------------------------------------------------------------ Mann-Kendall and tables
def test_mann_kendall_detects_monotone_trend_and_matches_known_slope():
    out = trends.mann_kendall_test(np.arange(1.0, 12.0) * 2)
    assert out["trend"] == "increasing" and out["p"] < 0.001
    assert out["sen_slope"] == pytest.approx(2.0) and out["tau"] == pytest.approx(1.0)
    assert out["slope_low"] <= out["sen_slope"] <= out["slope_high"]
    down = trends.mann_kendall_test(np.arange(11.0, 0.0, -1.0))
    assert down["trend"] == "decreasing" and down["sen_slope"] == pytest.approx(-1.0)


def test_mann_kendall_constant_and_short_series_are_safe():
    for y in (np.zeros(11), np.array([3.0, 3.0, 3.0, 3.0]), np.array([1.0, 2.0])):
        out = trends.mann_kendall_test(y)
        assert out["p"] == 1.0 and out["sen_slope"] == 0.0 and out["trend"] == "no trend"


def _wide():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, len(YEARS))
    return pd.DataFrame(
        {
            "rising": np.linspace(0.02, 0.20, len(YEARS)) + noise,
            "falling": np.linspace(0.20, 0.02, len(YEARS)) + noise,
            "flat": 0.1 + noise * 0.1 * np.tile([1, -1], 6)[: len(YEARS)],
            "rising_but_recent_low": np.r_[np.linspace(0.02, 0.2, 9), 0.01, 0.01],
        },
        index=YEARS,
    )


def test_trend_table_directions_bh_and_emerging():
    out = trends.trend_table(_wide(), alpha=0.05, recent_n=2).set_index("group")
    assert out.loc["rising", "direction"] == "increasing" and bool(out.loc["rising", "emerging"])
    assert out.loc["falling", "direction"] == "decreasing" and not bool(
        out.loc["falling", "emerging"]
    )
    assert out.loc["flat", "direction"] == "no trend"
    # an increasing series whose last two years drop below its long-run mean is not 'emerging'
    row = out.loc["rising_but_recent_low"]
    assert row["recent_mean"] < row["longrun_mean"] and not bool(row["emerging"])
    assert (out["p_bh"] >= out["p"] - 1e-12).all()  # BH never lowers a p-value
    assert out["sen_slope"].is_monotonic_decreasing  # sorted by slope


def test_bh_correction_changes_borderline_results():
    # Many noisy flat series plus one weak trend: unadjusted p < .05 can become non-significant after BH.
    rng = np.random.default_rng(0)
    wide = pd.DataFrame(rng.normal(0, 1, (11, 30)), index=YEARS).abs()
    wide["weak"] = np.linspace(0, 1.4, 11) + rng.normal(0, 0.9, 11)
    out = trends.trend_table(wide).set_index("group")
    assert (out["p"] < 0.05).sum() > (
        out["p_bh"] < 0.05
    ).sum()  # correction removes false positives


def test_classification_matrix_separates_measure_disagreement_from_window_sensitivity():
    tables = {
        "counts": pd.DataFrame(
            {"group": ["x", "y", "z"], "direction": ["increasing", "increasing", "no trend"]}
        ),
        "share": pd.DataFrame(
            {"group": ["x", "y", "z"], "direction": ["increasing", "no trend", "no trend"]}
        ),
        "wshare": pd.DataFrame(
            {"group": ["x", "y", "z"], "direction": ["increasing", "no trend", "no trend"]}
        ),
        "late": pd.DataFrame(
            {"group": ["x", "y", "z"], "direction": ["no trend", "no trend", "no trend"]}
        ),
    }
    out = trends.classification_matrix(
        tables, core=["counts", "share", "wshare"], late="late", reference="share"
    )
    assert out["agree"].to_dict() == {"x": True, "y": False, "z": True}
    assert out["window_sensitive"].to_dict() == {"x": True, "y": False, "z": False}


def test_reports_by_year():
    out = trends.reports_by_year(_incidents(), [2020, 2021, 2022]).set_index("year")
    assert out.loc[2020, "mean_reports"] == pytest.approx(5 / 3) and out.loc[2022, "incidents"] == 1


# ------------------------------------------------------------------ forecasting
def test_quarterly_counts_fills_missing_quarters():
    df = pd.DataFrame(
        {
            "year": [2020, 2020, 2021],
            "year_quarter": ["2020Q1", "2020Q1", "2021Q3"],
            "theme": ["A", "B", "A"],
        }
    )
    total = forecast.quarterly_counts(df, None, 2020, 2021)
    assert list(total.index) == [
        "2020Q1",
        "2020Q2",
        "2020Q3",
        "2020Q4",
        "2021Q1",
        "2021Q2",
        "2021Q3",
        "2021Q4",
    ]
    assert total["total"].tolist() == [2, 0, 0, 0, 0, 0, 1, 0]
    by_theme = forecast.quarterly_counts(df, "theme", 2020, 2021)
    assert (
        by_theme.loc["2020Q1", "A"] == 1
        and by_theme.loc["2021Q3", "A"] == 1
        and by_theme["B"].sum() == 1
    )


def test_naive_forecast_and_mase_arithmetic():
    assert forecast.forecast_naive(np.array([1.0, 5.0]), 3).tolist() == [5.0, 5.0, 5.0]
    errors = pd.DataFrame({"actual": [10.0, 10.0], "forecast": [8.0, 14.0], "scale": [2.0, 2.0]})
    assert forecast.mase(errors) == pytest.approx((2 / 2 + 4 / 2) / 2)
    assert np.isnan(forecast.mase(errors.assign(scale=0.0)))


def test_rolling_origin_shape_and_no_leakage():
    y = np.arange(1.0, 31.0)  # perfectly linear
    errors = forecast.rolling_origin(y, initial=20, horizon=2, orders=[[0, 1, 0]])
    origins = sorted(errors["origin"].unique())
    assert origins == list(range(20, 29))  # last origin (28) still leaves a full 2-step horizon
    assert set(errors["model"]) == set(forecast.MODELS) and set(errors["h"]) == {1, 2}
    naive = errors[
        (errors["model"] == "naive") & (errors["origin"] == 20) & (errors["h"] == 1)
    ].iloc[0]
    assert naive["forecast"] == 20.0 and naive["actual"] == 21.0  # uses only data before the origin
    assert naive["scale"] == pytest.approx(1.0)


def test_evaluate_series_skips_sparse_series_and_summarises():
    rng = np.random.default_rng(3)
    wide = pd.DataFrame(
        {"busy": 20 + np.cumsum(rng.normal(0, 1, 36)), "sparse": rng.poisson(0.2, 36)}
    )
    long, skipped = forecast.evaluate_series(
        wide, initial=24, horizon=2, orders=[[1, 1, 0], [0, 1, 1]], min_mean=1.0
    )
    assert skipped["series"].tolist() == ["sparse"]
    assert set(long["series"]) == {"busy"} and set(long["model"]) == set(forecast.MODELS)
    summary = forecast.summarise_mase(long)
    assert set(summary["model"]) == set(forecast.MODELS) and summary["series_where_best"].sum() == 1
    assert summary["mean_mase_all_horizons"].is_monotonic_increasing  # sorted best first
