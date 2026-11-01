"""Phase 1 tests: audit helpers, loader header detection, registry and table rendering.

Function-level tests use tiny hand-written frames to pin down behaviour (they are test
fixtures, not analysis data). Integration tests run against the real AIID files and are
skipped when data/raw is empty.
"""

import json

import numpy as np
import pandas as pd
import pytest

from aicidentia import registry
from aicidentia.data import audit, load
from aicidentia.viz.style import to_markdown

HAS_DATA = True
try:
    load.find_excel_export()
except FileNotFoundError:
    HAS_DATA = False
needs_data = pytest.mark.skipif(not HAS_DATA, reason="AIID Excel export not in data/raw")


# ---------------------------------------------------------------- pure helpers
def test_parse_percent():
    assert audit.parse_percent("84%") == pytest.approx(0.84)
    assert audit.parse_percent(" 100 % ") == 1.0
    assert audit.parse_percent("-") is None


def test_parse_multi_value():
    assert audit.parse_multi_value(None) == []
    assert audit.parse_multi_value(float("nan")) == []
    assert audit.parse_multi_value("") == []
    assert audit.parse_multi_value("yes") == ["yes"]
    assert audit.parse_multi_value('["a", "b"]') == ["a", "b"]
    assert audit.parse_multi_value("[]") == []
    assert audit.parse_multi_value("[not json") == ["[not json"]


def test_strip_domain_number():
    assert (
        audit.strip_domain_number("1. Discrimination and Toxicity") == "Discrimination and Toxicity"
    )
    assert (
        audit.strip_domain_number("1.2. Exposure to toxic content") == "Exposure to toxic content"
    )
    assert audit.strip_domain_number("No number") == "No number"


def test_normalised_entropy_bounds():
    assert audit.normalised_entropy(pd.Series([10, 10, 10, 10])) == pytest.approx(1.0)
    assert audit.normalised_entropy(pd.Series([100, 1])) < 0.2
    assert audit.normalised_entropy(pd.Series([5])) == 0.0


def test_recommend_year_range_skips_sparse_years():
    per_year = pd.DataFrame(
        {
            "year": [2012, 2013, 2014, 2015, 2016, 2017],
            "incidents": [3, 50, 8, 25, 30, 99],
            "complete": [True, True, True, True, True, False],
        }
    )
    rec = audit.recommend_year_range(per_year, min_incidents=20)
    # 2017 is partial, 2014 has only 8 incidents so the window starts at 2015.
    assert (rec["start"], rec["end"]) == (2015, 2016)
    assert rec["excluded_years"] == [2012, 2013, 2014]


def test_incidents_per_year_flags_partial_snapshot_year():
    inc = pd.DataFrame(
        {
            "year": [2024, 2025, 2025, 2026],
            "date": pd.to_datetime(["2024-05-01", "2025-01-01", "2025-06-01", "2026-03-01"]),
        }
    )
    out = audit.incidents_per_year(inc, "20261026").set_index("year")
    assert out.loc[2025, "complete"] and not out.loc[2026, "complete"]
    assert out.loc[2025, "incidents"] == 2


def test_merge_chains_detects_multi_hop():
    pairs = pd.DataFrame(
        {"duplicate_incident_number": [342, 88, 5], "true_incident_number": [88, 1057, 6]}
    )
    assert audit.merge_chains(pairs) == [[342, 88, 1057]]


def test_label_profile_single_vs_multi_and_balance():
    inc = pd.DataFrame(
        {
            "single": ["a", "a", "b", None, "a", "b"],
            "multi": ['["x", "y"]', '["x"]', None, None, '["y"]', '["x"]'],
        }
    )
    prof = audit.label_profile(inc, {"G": ["single", "multi"]}).set_index("column")
    assert prof.loc["single", "coverage_n"] == 5
    assert prof.loc["single", "value_type"] == "single-valued"
    assert prof.loc["single", "n_classes"] == 2
    assert prof.loc["single", "top_class"] == "a"
    assert prof.loc["multi", "value_type"] == "multi-valued"
    assert prof.loc["multi", "share_multi"] == pytest.approx(1 / 4)


def test_class_counts_by_period_split():
    inc = pd.DataFrame({"year": [2020, 2023, 2024, 2025, 2025], "lab": ["a", "a", "a", "b", "b"]})
    out = audit.class_counts_by_period(inc, "lab", test_from_year=2024).set_index("class")
    assert out.loc["a", "n_before_2024"] == 2 and out.loc["a", "n_from_2024"] == 1
    assert out.loc["b", "n_from_2024"] == 2 and out.loc["b", "test_share"] == 1.0


def test_label_coverage_by_year():
    inc = pd.DataFrame({"year": [2020, 2020, 2021], "lab": ["a", None, "b"]})
    out = audit.label_coverage_by_year(inc, "lab").set_index("year")
    assert out.loc[2020, "coverage"] == 0.5 and out.loc[2021, "coverage"] == 1.0


def test_text_length_stats_counts_empty():
    inc = pd.DataFrame({"title": ["one two", ""], "description": ["a b c", "d"]})
    stats = audit.text_length_stats(inc).set_index(["field", "unit"])
    assert stats.loc[("title", "words"), "empty"] == 1
    assert stats.loc[("description", "words"), "max"] == 3


def test_duplicate_checks_finds_exact_and_near_duplicates():
    inc = pd.DataFrame(
        {
            "Incident ID": [1, 2, 3],
            "title": ["Robot arrests man", "Robot arrests man", "Completely different topic"],
            "description": [
                "A police robot wrongly arrested a man.",
                "A police robot wrongly arrested a man.",
                "Fish swim.",
            ],
        }
    )
    facts, pairs = audit.duplicate_checks(inc, None, None)
    assert facts["duplicate_titles_exact"] == 1 and facts["duplicate_descriptions_exact"] == 1
    assert facts["near_duplicate_pairs_cosine_ge_90"] == 1
    assert {pairs.iloc[0]["incident_a"], pairs.iloc[0]["incident_b"]} == {1, 2}


def test_similarity_audit_symmetrises_and_bounds_precision():
    snap = pd.DataFrame(
        {
            "incident_id": [1, 2, 3],
            "editor_similar_incidents": [[2], [], []],
            "editor_dissimilar_incidents": [[3], [], []],
            "nlp_similar_incidents": [[{"incident_id": 2}], [], []],
        }
    )
    out = audit.similarity_audit(snap, {1, 2, 3})
    assert out["incidents_with_editor_similar"] == 1
    assert out["editor_similar_usable_queries_symmetrised"] == 2  # 1 -> 2 and 2 -> 1
    assert out["max_possible_precision_at_5_mean"] == pytest.approx(0.2)
    assert out["links_with_both_ends_in_excel_pct"] == 1.0


# ---------------------------------------------------------------- loader
def test_load_sheet_detects_header_under_banner_and_keeps_none(tmp_path):
    path = tmp_path / "AIID_Excel_Export-20000101.xlsx"
    banner = [
        ["AI Incident Database - Entities | 2 entities", None],
        ["Entity ID", "Name"],
        [None, None],
    ]
    rows = banner + [["none-entity", "None"], ["acme", "Acme"]]
    pd.DataFrame(rows).to_excel(path, sheet_name="Entities", header=False, index=False)
    df = load.load_sheet("Entities", path)
    assert list(df.columns) == ["Entity ID", "Name"]
    assert len(df) == 2  # banner and spacer rows removed
    assert df.loc[0, "Name"] == "None"  # literal "None" is not turned into NaN


# ---------------------------------------------------------------- registry and tables
def test_registry_roundtrip_and_numpy_conversion(tmp_path):
    path = tmp_path / "results.json"
    registry.record("a.count", np.int64(5), "data/x.xlsx", "mod.func", path=path)
    registry.record("a.table", pd.DataFrame({"k": [1]}), "data/x.xlsx", "mod.func", path=path)
    registry.record("a.nan", float("nan"), "data/x.xlsx", "mod.func", path=path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["a.count"] == {"value": 5, "source_file": "data/x.xlsx", "function": "mod.func"}
    assert stored["a.table"]["value"] == [{"k": 1}]
    assert stored["a.nan"]["value"] is None
    assert registry.get("a.count", path=path) == 5
    assert list(stored) == sorted(stored)  # deterministic key order


def test_to_markdown_escapes_pipes_and_blanks_nan():
    md = to_markdown(pd.DataFrame({"a": ["x|y", None], "b": [1.23456, float("nan")]}))
    assert "x\\|y" in md
    assert md.splitlines()[0] == "| a | b |"
    assert md.splitlines()[-1] == "|  |  |"


# ---------------------------------------------------------------- real data
@needs_data
def test_real_excel_counts_match_sheet_banners():
    shapes = audit.sheet_shapes().set_index("sheet")
    assert shapes.loc["Incidents", "rows"] == shapes.loc["Incidents", "banner_count"]
    assert shapes.loc["Reports", "rows"] == shapes.loc["Reports", "banner_count"]


@needs_data
def test_real_incidents_are_clean_enough_for_phase2():
    inc = load.load_incidents()
    assert inc["Incident ID"].is_unique and inc["Incident ID"].notna().all()
    assert inc["date"].notna().all()
    assert (inc["date"].dt.year == inc["year"]).all()
    assert inc["title"].notna().all() and inc["description"].notna().all()


@needs_data
def test_real_excel_and_snapshot_agree_on_shared_incidents():
    if load.find_snapshot_dir() is None:
        pytest.skip("snapshot not present")
    inc = load.load_incidents().set_index("Incident ID")
    snap = load.load_snapshot_incidents().set_index("incident_id")
    shared = inc.index.intersection(snap.index)
    assert len(shared) == len(inc)
    assert (inc.loc[shared, "title"] == snap.loc[shared, "title"]).all()
