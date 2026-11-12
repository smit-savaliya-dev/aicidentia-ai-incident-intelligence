"""RQ3 label-reliability checks on small hand-made data with known answers."""

import numpy as np
import pandas as pd
import pytest

from aicidentia.models import reliability as R


def _incidents():
    rows = []
    for i in range(40):
        disc = i < 20
        rows.append(
            {
                "incident_id": i,
                "year": 2018 + i % 3,
                "has_mit": True,
                "has_cset": True,
                "has_gmf": i < 10,
                "mit_risk_domain": R.DISCRIMINATION if disc else R.SAFETY,
                "cset_protected_characteristic": ("yes" if disc else "no")
                if i not in (0, 39)
                else "maybe",
                "cset_harm_distribution_basis": ["race"] if disc else ["none"],
                "cset_tangible_harm": "tangible harm definitively occurred"
                if not disc
                else "no tangible harm, near-miss, or issue",
                "gmf_technical_failure": ["Distributional Bias"] if i % 2 else [],
                "cset_detrimental_content": "no",
            }
        )
    return pd.DataFrame(rows)


def test_mapping_agreement_perfect_proxy_and_exclusions():
    table, crosstabs = R.mapping_agreement(_incidents(), n_boot=200)
    row = table.set_index("mapping").loc["discrimination_vs_protected_characteristic"]
    assert row["n"] == 38 and row["excluded_unclear"] == 2  # 'maybe' rows are left out
    assert row["agreement"] == 1.0 and row["kappa"] == pytest.approx(1.0)
    safety = table.set_index("mapping").loc["safety_vs_tangible_harm"]
    assert safety["kappa"] == pytest.approx(1.0) and safety["mit_positive"] == 20
    assert table.set_index("mapping").loc["safety_vs_gmf_technical_failure", "n"] == 10
    assert set(crosstabs) == set(table["mapping"])


def test_parsers():
    assert R._tangible("imminent risk of tangible harm (near miss) did occur") is True
    assert R._tangible("non-imminent risk of tangible harm (an issue) occurred") is False
    assert R._tangible("unclear") is None
    assert R._basis(["none"]) is False and R._basis(["race", "none"]) is True
    assert R._basis(["unclear"]) is None and R._cset_yes_no("maybe") is None


def test_pair_consistency_against_year_matched_baseline():
    domain = {i: ("a" if i % 2 else "b") for i in range(100)}
    year = {i: 2020 + i // 25 for i in range(100)}
    same_pairs = [
        (i, i + 2) for i in range(0, 90, 3)
    ]  # i and i+2 always share parity -> same domain
    res = R.pair_consistency(same_pairs, domain, year, n_perm=300, n_boot=300)
    assert res["same_domain_rate"] == 1.0 and res["pairs_labelled"] == len(same_pairs)
    assert 0.3 < res["baseline_rate"] < 0.7 and res["p_value"] < 0.01
    mixed = R.pair_consistency([(0, 1), (2, 3)], domain, year, n_perm=100, n_boot=100)
    assert mixed["same_domain_rate"] == 0.0
    assert R.pair_consistency([(500, 501)], domain, year)["pairs_labelled"] == 0
    assert R.unique_pairs({1: {2}, 2: {1, 3}, 3: {2}}) == [(1, 2), (2, 3)]


def test_confident_learning_finds_planted_label_errors():
    rng = np.random.default_rng(0)
    n, k = 600, 3
    true = rng.integers(0, k, n)
    probs = np.full((n, k), 0.05)
    probs[np.arange(n), true] = 0.9
    probs /= probs.sum(axis=1, keepdims=True)
    given = true.copy()
    flipped = rng.choice(n, 30, replace=False)
    given[flipped] = (true[flipped] + 1) % k
    cl = R.confident_learning(given, probs, ["x", "y", "z"], np.arange(n))
    found = set(np.flatnonzero(cl.flagged))
    assert len(found & set(flipped)) >= 27 and len(found - set(flipped)) <= 3
    assert cl.noise_rate == pytest.approx(30 / n, abs=0.02)
    assert cl.joint.to_numpy().sum() == pytest.approx(1.0)
    sym = R.symmetric_confusion(cl.joint, n)
    assert {"domain_a", "domain_b", "total", "two_way_share"} <= set(sym.columns)
    assert len(R.confusion_pairs(cl.joint, n, top=4)) == 4


def test_retrain_sensitivity_reports_primary_and_subset():
    original = pd.DataFrame(
        {"incident_id": range(6), "y_true": [0, 0, 1, 1, 2, 2], "y_pred": [0, 1, 1, 1, 2, 0]}
    )
    cleaned = original.assign(y_pred=[0, 0, 1, 1, 2, 2])
    out = R.retrain_sensitivity(original, cleaned, {1, 5}, 3, n_boot=100)
    assert out["evaluated_on"].tolist() == ["all incidents (primary)", "unflagged incidents only"]
    assert out.iloc[0]["n"] == 6 and out.iloc[1]["n"] == 4
    assert out.iloc[0]["macro_f1_cleaned_training"] == 1.0 and out.iloc[0]["difference"] > 0
