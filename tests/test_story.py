"""Story analyses: ChatGPT shift statistics, severity flag, risk-matrix quadrants, community concentration."""

import numpy as np
import pandas as pd
import pytest

from aicidentia.models import story as S


def test_period_split_at_chatgpt_release():
    dates = pd.Series(pd.to_datetime(["2022-11-29", "2022-11-30", "2023-01-01"]))
    assert S.period_of(dates).tolist() == ["before", "after", "after"]


def test_chi_square_shift_residuals_and_cramers_v():
    groups = pd.Series(["a"] * 50 + ["b"] * 50 + ["a"] * 10 + ["b"] * 90)
    period = pd.Series(["before"] * 100 + ["after"] * 100)
    r = S.chi_square_shift(groups, period)
    t = r["table"].set_index("group")
    assert t.loc["b", "adj_residual_after"] > 1.96 > -1.96 > t.loc["a", "adj_residual_after"]
    assert t.loc["b", "driver"] == "over-represented after"
    assert t.loc["a", "share_before"] == 0.5 and t.loc["a", "share_after"] == 0.1
    # 2x2: Cramer's V equals |phi|
    a, b, c, d = 50, 10, 50, 90
    phi = (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    assert r["cramers_v"] == pytest.approx(abs(phi))
    assert r["n_before"] == 100 and r["p_value"] < 1e-6


def test_severity_flag_uses_cset_only():
    inc = pd.DataFrame(
        {
            "has_cset": [True, True, True, True, False],
            "cset_tangible_harm": [
                "tangible harm definitively occurred",
                "imminent risk of tangible harm (near miss) did occur",
                "no tangible harm, near-miss, or issue",
                "no tangible harm, near-miss, or issue",
                None,
            ],
            "cset_injuries": [0, 0, 2, 0, 5],
            "cset_lives_lost": [0, 0, 0, 0, 1],
        }
    )
    flag = S.severe(inc)
    assert flag.tolist()[:4] == [1.0, 0.0, 1.0, 0.0] and np.isnan(flag.iloc[4])


def test_quadrants():
    assert S._quadrant(0.5, 0.6, 0.35, 0.01) == "Emerging & severe"
    assert S._quadrant(-0.5, 0.1, 0.35, 0.01) == "Established"
    assert S._quadrant(0.2, 0.2, 0.35, 0.7) == "Emerging (trend not significant)"


def test_priority_matrix_marks_themes_without_enough_cset_data():
    n = 60
    inc = pd.DataFrame(
        {
            "incident_id": range(n),
            "year": 2020,
            "has_cset": [True] * 40 + [False] * 20,
            "cset_tangible_harm": ["tangible harm definitively occurred"] * 20 + ["no"] * 40,
            "cset_injuries": 0,
            "cset_lives_lost": 0,
        }
    )
    themes = pd.DataFrame({"incident_id": range(n), "theme": ["x"] * 35 + ["y"] * 25})
    trends = pd.DataFrame(
        {
            "group": ["x", "y"],
            "basis": "share",
            "sen_slope": [0.01, -0.01],
            "p_bh": [0.01, 0.5],
            "emerging": [True, False],
        }
    )
    m, facts = S.priority_matrix(inc, themes, trends, min_cset=30)
    m = m.set_index("theme")
    assert m.loc["x", "supported"] and not m.loc["y", "supported"]
    assert m.loc["x", "cset_n"] == 35 and m.loc["x", "severity"] == pytest.approx(20 / 35)
    assert (
        m.loc["x", "quadrant"] == "Emerging & severe"
        and m.loc["y", "quadrant"] == "not enough CSET data"
    )
    assert m.loc["x", "trend_pp_per_year"] == pytest.approx(1.0)
    assert facts["themes_supported"] == 1 and facts["severity_overall"] == 0.5


def test_theme_community_concentration_lift():
    themes = pd.DataFrame({"incident_id": [1, 2, 3, 4], "theme": ["t", "t", "u", "u"]})
    ie = pd.DataFrame({"incident_id": [1, 2, 3, 4], "entity_id": ["a", "a", "b", "a"]})
    comm = pd.DataFrame({"node": ["a", "b"], "community": [0, 1]})
    out = S.theme_community_concentration(themes, ie, comm, min_incidents=1)
    row = out[(out["theme"] == "t") & (out["community"] == 0)].iloc[0]
    assert row["share_of_theme"] == 1.0 and row["lift"] == pytest.approx(1 / 0.75)
