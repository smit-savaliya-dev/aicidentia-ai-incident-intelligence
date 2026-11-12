"""Rules D2 (duplicates) and D3 (named persons) that answer the review files."""

import pandas as pd

from aicidentia.data import rules
from aicidentia.pipeline import steps


def _pairs():
    return pd.DataFrame(
        {
            "incident_a": ["1", "3", "5", "7", "20"],
            "incident_b": ["2", "4", "6", "8", "9"],
            "cosine": ["0.95", "0.95", "0.95", "0.80", "0.93"],
            "date_a": ["2020-01-01", "2020-01-01", "2020-01-01", "2020-01-01", "2021-05-05"],
            "date_b": ["2020-01-05", "2020-03-01", "2020-01-02", "2020-01-02", "2021-05-01"],
            "decision": "",
            "notes": "",
        }
    )


def test_duplicate_rule_needs_all_three_criteria():
    deployers = {
        1: {"acme"},
        2: {"acme", "x"},
        3: {"acme"},
        4: {"acme"},
        5: {"acme"},
        6: {"other"},
        7: {"acme"},
        8: {"acme"},
        20: {"acme"},
        9: {"acme"},
    }
    out, counts = rules.apply_duplicate_rule(_pairs(), deployers)
    assert out["decision"].tolist() == [
        "merge_into_a",  # all criteria met
        "keep_separate",  # 60 days apart
        "keep_separate",  # no shared deployer
        "keep_separate",  # cosine 0.80
        "merge_into_b",  # met; keeps the lower id (9), which AIID treats as the original
    ]
    assert counts == {
        "pairs": 5,
        "merged": 2,
        "keep_separate": 3,
        "fail_deployer": 1,
        "fail_dates": 1,
        "fail_similarity": 1,
    }
    assert out["notes"].str.startswith("rule D2").all()
    assert "acme" not in " ".join(out["notes"])  # deployer names are never written


def test_duplicate_rule_boundaries():
    assert rules.duplicate_decision({"a"}, 7, 0.90)[0] == "merge_into_a"
    assert rules.duplicate_decision({"a"}, 8, 0.99)[0] == "keep_separate"
    assert rules.duplicate_decision({"a"}, 0, 0.899)[0] == "keep_separate"


def test_entity_rule_excludes_every_named_person_and_keeps_overridden_non_persons():
    review = pd.DataFrame(
        {
            "entity_id": ["jane-smith", "pizza-hut", "joe-biden", "google-home", "black-americans"],
            "is_named_person": "",
            "notes": "",
        }
    )
    overrides = {
        "pizza-hut": "organisation",
        "google-home": "technology_system",
        "black-americans": "population_group",
        "none": "unknown",
    }
    out, counts = rules.apply_entity_rule(review, overrides)
    assert out["is_named_person"].tolist() == ["yes", "no", "yes", "no", "no"]
    assert counts["named_persons_excluded"] == 2 and counts["not_persons_kept"] == 3
    assert counts["kept_as_organisation"] == 1 and counts["kept_as_population_group"] == 1


def test_refresh_starts_with_the_rules():
    names = [s.name for s in steps()]
    assert names[0] == "rules" and names.index("rules") < names.index("phase2")
