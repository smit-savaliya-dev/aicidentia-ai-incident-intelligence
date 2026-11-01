"""Phase 2 tests: cleaning functions, entity layer, duplicate review, validation, privacy handling.

Function-level tests use tiny hand-written frames (test fixtures, not analysis data). The
integration tests read data/processed/ and the manual review files if they exist.
"""

import numpy as np
import pandas as pd
import pytest

from aicidentia.data import clean, duplicates, entities, validate
from aicidentia.data.prepare import apply_privacy_exclusions

RULES = entities.load_type_rules()


# ------------------------------------------------------------------ clean: text and time
def test_normalise_whitespace():
    assert clean.normalise_whitespace("  a \n b\t c ") == "a b c"
    assert clean.normalise_whitespace("   ") is None
    assert clean.normalise_whitespace(None) is None
    assert clean.normalise_whitespace(float("nan")) is None


def test_assemble_incident_text_avoids_double_full_stop():
    title = pd.Series(["Robot fails", "Robot fails.", "Why?"])
    desc = pd.Series(["It crashed.", "It crashed.", "Unclear"])
    out = clean.assemble_incident_text(title, desc).tolist()
    assert out == ["Robot fails. It crashed.", "Robot fails. It crashed.", "Why? Unclear."]


def test_assemble_incident_text_handles_missing_description():
    out = clean.assemble_incident_text(pd.Series(["Only title"]), pd.Series([None]))
    assert out.iloc[0] == "Only title."


def test_add_time_columns_and_filter_year_range():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-02-29", "2021-07-01", "2025-12-31"])})
    out = clean.add_time_columns(df)
    assert out["quarter"].tolist() == [1, 3, 4]
    assert out["year_quarter"].tolist() == ["2020Q1", "2021Q3", "2025Q4"]
    kept = clean.filter_year_range(out, "date", 2021, 2025)
    assert kept["year"].tolist() == [2021, 2025]
    assert clean.filter_year_range(out, "date", 2021, 2021)["year"].tolist() == [2021]


# ------------------------------------------------------------------ clean: labels
def test_snake_case():
    assert clean.snake_case("Technical Failure (Potential)") == "technical_failure_potential"
    assert clean.snake_case("AI Harm Level") == "ai_harm_level"


def test_tidy_label_column_dtypes():
    lists = clean.tidy_label_column(pd.Series(['["a", "b"]', None, '["c"]']))
    assert lists.tolist() == [["a", "b"], [], ["c"]]
    bools = clean.tidy_label_column(pd.Series([True, False, None]))
    assert str(bools.dtype) == "boolean" and bools.tolist()[:2] == [True, False]
    nums = clean.tidy_label_column(pd.Series(["0", "3", None]))
    assert str(nums.dtype) == "Int64" and nums.iloc[1] == 3
    text = clean.tidy_label_column(pd.Series(["  yes ", "", None]))
    assert text.iloc[0] == "yes" and pd.isna(text.iloc[1]) and pd.isna(text.iloc[2])
    assert clean.tidy_label_column(pd.Series([None, None])).isna().all()


def _dictionary():
    return pd.DataFrame(
        {
            "Column": ["Risk Domain", "AI Goal", "Harm Domain", "Lives Lost", "Entities"],
            "Group": ["MIT", "GMF", "CSETv1", "CSETv1", "CSETv1"],
        }
    )


def test_tidy_labels_prefixes_strips_numbers_and_keeps_entities_json():
    inc = pd.DataFrame(
        {
            "Risk Domain": ["1. Discrimination and Toxicity", None],
            "AI Goal": ['["Chat"]', None],
            "Harm Domain": ["yes", None],
            "Lives Lost": ["2", None],
            "Entities": ['[{"a": 1}]', None],
        }
    )
    out = clean.tidy_labels(inc, _dictionary())
    assert set(out.columns) == {
        "mit_risk_domain",
        "gmf_ai_goal",
        "cset_harm_domain",
        "cset_lives_lost",
        "cset_entities_json",
    }
    assert out.loc[0, "mit_risk_domain"] == "Discrimination and Toxicity"
    assert out.loc[0, "gmf_ai_goal"] == ["Chat"] and out.loc[1, "gmf_ai_goal"] == []
    assert out.loc[0, "cset_lives_lost"] == 2


def test_clean_incidents_builds_expected_columns():
    raw = pd.DataFrame(
        {
            "Incident ID": [2, 1],
            "date": ["2020-05-01", "2019-01-15"],
            "year": [2020, 2019],
            "title": ["  Second  incident ", "First incident."],
            "description": ["Something   happened.", "Another thing."],
            "report_count": [3, 1],
            "Data Sources": ["MIT | CSETv1", None],
            "Risk Domain": ["4. Malicious Actors & Misuse", None],
            "AI Goal": [None, None],
            "Harm Domain": ["yes", None],
            "Lives Lost": ["0", None],
            "Entities": [None, None],
        }
    )
    out = clean.clean_incidents(raw, _dictionary())
    assert out["incident_id"].tolist() == [1, 2]  # sorted
    assert out["quarter"].tolist() == [1, 2] and out["year_quarter"].tolist() == [
        "2019Q1",
        "2020Q2",
    ]
    assert out.loc[1, "title"] == "Second incident"
    assert out.loc[1, "text"] == "Second incident. Something happened."
    assert out["has_mit"].tolist() == [False, True] and out["has_cset"].tolist() == [False, True]
    assert out.loc[1, "mit_risk_domain"] == "Malicious Actors & Misuse"


# ------------------------------------------------------------------ clean: merges, entities sheet, reports
def test_resolve_merge_map_is_transitive_and_cycle_safe():
    pairs = pd.DataFrame(
        {"duplicate_incident_number": [342, 88, 5, 6], "true_incident_number": [88, 1057, 6, 5]}
    )
    resolved = clean.resolve_merge_map(pairs)
    assert resolved[342] == 1057 and resolved[88] == 1057
    assert set(resolved) == {342, 88, 5, 6}  # the 5<->6 cycle terminates


def test_apply_merge_map_and_drop_merged_incidents():
    links = pd.DataFrame({"incident_id": [1, 2, 3], "x": ["a", "a", "b"]})
    out = clean.apply_merge_map(links.assign(x="a"), "incident_id", {2: 1})
    assert out["incident_id"].tolist() == [1, 3]  # the merged row collapses into incident 1
    inc = pd.DataFrame({"incident_id": [1, 2, 3]})
    assert clean.drop_merged_incidents(inc, {2: 1})["incident_id"].tolist() == [1, 3]


def test_clean_entities_trims_and_falls_back_to_slug():
    raw = pd.DataFrame(
        {"Entity ID": [" acme ", "nameless-entity", None], "Name": [" Acme  Corp ", None, "orphan"]}
    )
    out = clean.clean_entities(raw)
    assert out["Entity ID"].tolist() == ["acme", "nameless-entity"]
    assert out["Name"].tolist() == ["Acme Corp", "Nameless entity"]


def _reports_sheet():
    return pd.DataFrame(
        {
            "Report Number": [1, 2, 3, 4],
            "Title": ["Real one", None, "Old date", "Unlinked"],
            "URL": ["http://a", None, "http://c", "http://d"],
            "Source Domain": ["a.com", None, "c.com", "d.com"],
            "Date Published": pd.to_datetime(
                ["2020-01-01", "1967-01-01", "1970-01-01", "2024-01-01"]
            ),
            "Date Downloaded": pd.to_datetime(
                ["2020-02-01", "2020-02-01", "2020-02-01", "2024-02-01"]
            ),
            "Authors": ["A. Writer", None, "B. Writer", "C"],
            "Submitters": ["S1", None, "S2", "S3"],
            "Language": ["en", "en", "en", "en"],
            "Tags": [None, None, None, None],
            "Description": ["d", None, "d", "d"],
            "Is Incident Report": ["True", "False", "True", "False"],
        }
    )


def test_stub_detection_and_removal():
    sheet = _reports_sheet()
    assert clean.is_stub_report(sheet).tolist() == [False, True, False, False]
    assert clean.drop_stub_reports(sheet)["Report Number"].tolist() == [1, 3, 4]


def test_clean_reports_links_text_dates_and_privacy():
    snap_reports = pd.DataFrame(
        {
            "report_number": [1, 2, 3, 4],
            "text": ["  full   text one ", "x", "text three", "text four"],
        }
    )
    snap_inc = pd.DataFrame({"incident_id": [10, 11], "reports": [[1, 3], [3]]})
    out = clean.clean_reports(
        _reports_sheet(), snap_reports, snap_inc, {10, 11}, {11: 10}, "20260101"
    )
    assert out["report_number"].tolist() == [1, 3, 4]
    assert "authors" not in {c.lower() for c in out.columns} and "submitters" not in {
        c.lower() for c in out.columns
    }
    assert out.loc[0, "text"] == "full text one" and out.loc[0, "text_words"] == 3
    assert pd.isna(out.loc[1, "date_published"])  # 1970 placeholder date masked
    assert out.loc[0, "incident_ids"] == [10]
    assert out.loc[1, "incident_ids"] == [10]  # incident 11 merged into 10 and de-duplicated
    assert out["linked"].tolist() == [True, True, False]
    assert out["is_incident_report"].tolist() == [True, True, False]
    counts = clean.count_reports_per_incident(out)
    assert counts.loc[10] == 2


def test_save_and_load_processed_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(clean, "processed_path", lambda name: tmp_path / f"{name}.parquet")
    df = pd.DataFrame({"a": [1, 2], "b": [["x"], []]})
    clean.save_processed(df, "demo")
    pd.testing.assert_frame_equal(clean.load_processed("demo"), df)


# ------------------------------------------------------------------ entities
def test_build_long_table_filters_and_dedupes():
    snap = pd.DataFrame(
        {
            "incident_id": [1, 2],
            "deployer": [["a", "a"], ["b"]],
            "developer": [["a"], []],
            "harmed": [["kids"], ["kids"]],
            "implicated_systems": [[], ["sys"]],
        }
    )
    long = entities.build_long_table(snap, {1})
    assert len(long) == 3 and set(long["role"]) == {"deployer", "developer", "harmed"}
    assert entities.build_long_table(snap)["incident_id"].nunique() == 2


def test_normalise_name_key():
    assert entities.normalise_name_key("Carribex LTD.") == entities.normalise_name_key("Carribex")
    assert entities.normalise_name_key("Estée Lauder") == "esteelauder"


def test_resolve_alias_map_follows_chains_and_stops_at_cycles():
    resolved = entities.resolve_alias_map({"a": "b", "b": "c", "x": "y", "y": "x"})
    assert resolved["a"] == "c" and resolved["b"] == "c"
    assert resolved["x"] == "y"  # cycle ends at the repeated node


def test_build_alias_candidates_skips_missing_canonicals_and_cycles():
    ents = pd.DataFrame(
        {
            "Entity ID": ["scammers", "unknown-scammers", "ok-dup", "ok-true", "acme", "acme-inc."],
            "Name": ["Scammers", "Scammers", "Dup", "True", "Acme", "Acme Inc."],
        }
    )
    long = pd.DataFrame(
        {
            "incident_id": range(6),
            "role": ["deployer"] * 6,
            "entity_id_raw": ["scammers", "ok-dup", "acme", "acme-inc.", "acme-inc.", "ghost-dup"],
        }
    )
    aiid = pd.DataFrame(
        {
            "duplicate_entity_id": ["scammers", "unknown-scammers", "ok-dup", "ghost-dup"],
            "true_entity_id": ["unknown-scammers", "scammers", "ok-true", "ghost-true"],
        }
    )
    cands, skipped = entities.build_alias_candidates(ents, long, aiid)
    applied = dict(zip(cands["alias"], cands["canonical"], strict=True))
    assert applied["ok-dup"] == "ok-true"
    assert applied["acme"] == "acme-inc."  # same normalised name; the more frequent id is canonical
    assert "scammers" not in applied  # cycle
    reasons = dict(zip(skipped["alias"], skipped["reason"], strict=True))
    assert (
        reasons["scammers"] == "cycle" and reasons["ghost-dup"] == "canonical_not_in_entities_sheet"
    )


def test_alias_file_roundtrip_preserves_manual_and_blocked(tmp_path):
    path = tmp_path / "aliases.yaml"
    generated = pd.DataFrame(
        {
            "alias": ["a", "c"],
            "canonical": ["b", "d"],
            "source": ["s", "s"],
            "alias_name": ["A", "C"],
            "canonical_name": ["B", "D"],
            "incidents_alias": [1, 2],
            "incidents_canonical": [3, 4],
        }
    )
    entities.write_alias_file(generated, path)
    assert entities.read_alias_file(path)["manual"] == {
        "facebook": "meta"
    }  # example from the brief
    # User edits the file: add a manual alias and block one generated alias.
    text = (
        path.read_text(encoding="utf-8")
        .replace("manual:\n  facebook: meta", "manual:\n  facebook: meta\n  x: y")
        .replace("blocked: []", "blocked:\n- c")
    )
    path.write_text(text, encoding="utf-8")
    entities.write_alias_file(generated.iloc[:1], path)  # regenerate with fewer rows
    data = entities.read_alias_file(path)
    assert data["manual"] == {"facebook": "meta", "x": "y"} and data["blocked"] == ["c"]
    assert [r["alias"] for r in data["generated"]] == ["a"]
    assert entities.active_alias_map(path) == {"facebook": "meta", "x": "y", "a": "b"}


def test_apply_aliases_merges_and_names():
    long = pd.DataFrame(
        {"incident_id": [1, 1], "role": ["harmed", "harmed"], "entity_id_raw": ["facebook", "meta"]}
    )
    out = entities.apply_aliases(long, {"facebook": "meta"}, {"meta": "Meta"})
    assert len(out) == 1 and out.iloc[0]["entity_id"] == "meta" and out.iloc[0]["entity"] == "Meta"


def test_person_like():
    assert entities.person_like("Jane Smith", RULES) == (True, "high")
    assert entities.person_like("Jane Williams", RULES) == (True, "low")
    assert entities.person_like("OpenAI", RULES)[0] is False  # one token
    assert entities.person_like("Boston University", RULES)[0] is False  # organisation word
    assert entities.person_like("Women Drivers", RULES)[0] is False  # plural group noun
    assert entities.person_like("women and girls", RULES)[0] is False


def _long(rows):
    return pd.DataFrame(rows, columns=["incident_id", "role", "entity_id", "entity"])


def test_classify_entity_types_rules():
    long = _long(
        [
            (1, "deployer", "openai", "OpenAI"),
            (1, "developer", "openai", "OpenAI"),
            (2, "harmed", "privacy", "Privacy"),
            (2, "harmed", "students", "Students"),
            (3, "deployer", "scammers", "Scammers"),
            (3, "system", "deepfake-technology", "Deepfake technology"),
            (4, "harmed", "jane-smith", "Jane Smith"),
            (5, "harmed", "chatbot-users", "Chatbot users"),
        ]
    )
    types = entities.classify_entity_types(long, RULES).set_index("entity_id")
    assert types.loc["openai", "entity_type"] == "organisation"
    assert types.loc["privacy", "entity_type"] == "abstract_value"
    assert types.loc["students", "entity_type"] == "population_group"
    assert types.loc["scammers", "entity_type"] == "generic_actor_group"
    assert types.loc["deepfake-technology", "entity_type"] == "technology_system"
    assert types.loc["jane-smith", "entity_type"] == "individual_candidate"
    assert types.loc["chatbot-users", "entity_type"] == "population_group"
    assert set(types["entity_type"]) <= set(entities.ENTITY_TYPES)


def test_classify_entity_types_override_wins():
    long = _long([(1, "deployer", "acme", "Acme")])
    rules = {**RULES, "overrides": {"acme": "technology_system"}}
    assert entities.classify_entity_types(long, rules).iloc[0]["entity_type"] == "technology_system"


def test_review_candidates_leave_decision_blank_and_file_is_append_only(tmp_path):
    long = _long(
        [
            (1, "harmed", "jane-smith", "Jane Smith"),
            (2, "harmed", "john-doe-jr", "John Doe Jr"),
            (3, "deployer", "acme", "Acme"),
        ]
    )
    long["entity_type"] = ["individual_candidate", "individual_candidate", "organisation"]
    table = pd.DataFrame(
        {
            "entity_id": ["jane-smith", "john-doe-jr", "acme"],
            "entity": ["Jane Smith", "John Doe Jr", "Acme"],
            "n_incidents": [1, 5, 9],
            "roles": ["harmed", "harmed", "deployer"],
            "entity_type": ["individual_candidate", "individual_candidate", "organisation"],
        }
    )
    inc = pd.DataFrame({"incident_id": [1, 2, 3], "title": ["t1", "t2", "t3"]})
    cands = entities.build_review_candidates(table, long, inc, RULES)
    assert set(cands["entity_id"]) == {"jane-smith", "john-doe-jr"}
    assert (cands["is_private_individual"] == "").all() and (cands["notes"] == "").all()
    assert cands.set_index("entity_id").loc["jane-smith", "review_priority"] == "A"
    assert cands.set_index("entity_id").loc["john-doe-jr", "review_priority"] == "B"

    path = tmp_path / "entity_review.csv"
    assert entities.write_review_file(cands, path) is True
    answered = pd.read_csv(path, dtype=str, keep_default_na=False)
    answered.loc[answered["entity_id"] == "jane-smith", "is_private_individual"] = "yes"
    answered.to_csv(path, index=False)
    new = cands.assign(entity_id=cands["entity_id"] + "-2")
    assert entities.write_review_file(pd.concat([cands, new]), path) is True
    after = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("entity_id")
    assert after.loc["jane-smith", "is_private_individual"] == "yes"  # your answer survives
    assert len(after) == 4
    assert entities.write_review_file(pd.concat([cands, new]), path) is False  # nothing new
    assert entities.load_confirmed_individuals(path) == {"jane-smith"}
    assert entities.review_progress(path) == {"rows": 4, "answered": 1, "confirmed_private": 1}


def test_apply_privacy_exclusions_modes(tmp_path):
    path = tmp_path / "review.csv"
    pd.DataFrame({"entity_id": ["a", "b", "c"], "is_private_individual": ["yes", "no", ""]}).to_csv(
        path, index=False
    )
    table = pd.DataFrame(
        {
            "entity_id": ["a", "b", "c", "d"],
            "entity_type": ["individual_candidate"] * 3 + ["organisation"],
        }
    )
    excluded, info = apply_privacy_exclusions(table, False, path)
    assert excluded == {"a"} and info["confirmed_private"] == 1 and info["cleared_public"] == 1
    excluded, info = apply_privacy_exclusions(table, True, path)
    assert excluded == {"a", "c"} and info["unreviewed_candidates"] == 1  # b was cleared as public
    assert apply_privacy_exclusions(table, False, tmp_path / "missing.csv")[0] == set()


# ------------------------------------------------------------------ duplicates
def _incidents_for_dups():
    return pd.DataFrame(
        {
            "incident_id": [1, 2, 3],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2021-01-01"]),
            "title": ["Police robot arrests man", "Police robot arrests a man", "Fish swim in sea"],
            "description": [
                "A police robot wrongly arrested a man.",
                "A police robot wrongly arrested a man.",
                "Fish swim.",
            ],
            "text": [
                "Police robot arrests man. A police robot wrongly arrested a man.",
                "Police robot arrests a man. A police robot wrongly arrested a man.",
                "Fish swim in sea. Fish swim.",
            ],
        }
    )


def test_find_similar_pairs_and_decisions(tmp_path):
    pairs = duplicates.find_similar_pairs(_incidents_for_dups(), threshold=0.6)
    assert len(pairs) == 1 and (pairs.iloc[0]["incident_a"], pairs.iloc[0]["incident_b"]) == (1, 2)
    assert pairs.iloc[0]["decision"] == "" and pairs.iloc[0]["notes"] == ""
    path = tmp_path / "dups.csv"
    assert duplicates.write_review_file(pairs, path) is True
    assert duplicates.load_merge_decisions(path) == {}  # blank = nothing merged
    assert duplicates.review_progress(path) == {"rows": 1, "answered": 0, "merges": 0}
    review = pd.read_csv(path, dtype=str, keep_default_na=False)
    review.loc[0, "decision"] = "merge_into_a"
    review.to_csv(path, index=False)
    assert duplicates.load_merge_decisions(path) == {2: 1}
    assert duplicates.write_review_file(pairs, path) is False  # existing decisions untouched
    review.loc[0, "decision"] = "keep_separate"
    review.to_csv(path, index=False)
    assert duplicates.load_merge_decisions(path) == {}
    review.loc[0, "decision"] = "dunno"
    review.to_csv(path, index=False)
    with pytest.raises(ValueError):
        duplicates.load_merge_decisions(path)


# ------------------------------------------------------------------ validation
def _good_incidents():
    df = pd.DataFrame(
        {
            "incident_id": [1, 2],
            "date": pd.to_datetime(["2020-01-01", "2021-06-01"]),
            "title": ["A title", "Another"],
            "description": ["desc one", "desc two"],
            "report_count": [1, 2],
        }
    )
    df = clean.add_time_columns(df)
    df["text"] = clean.assemble_incident_text(df["title"], df["description"])
    return df


def test_validate_incidents_passes_on_good_data():
    assert all(c.passed for c in validate.validate_incidents(_good_incidents(), "20260101"))


@pytest.mark.parametrize(
    "mutate, failing",
    [
        (lambda d: d.assign(incident_id=[1, 1]), "incident_id unique"),
        (
            lambda d: d.assign(date=pd.to_datetime(["2020-01-01", "2030-01-01"])),
            "dates valid (1900 .. snapshot date)",
        ),
        (lambda d: d.assign(year=[2020, 1999]), "year equals date.year"),
        (lambda d: d.assign(title=["", "Another"]), "title non-empty"),
        (lambda d: d.assign(report_count=[1, -1]), "report_count >= 0"),
    ],
)
def test_validate_incidents_detects_problems(mutate, failing):
    checks = validate.validate_incidents(mutate(_good_incidents()), "20260101")
    assert failing in {c.name for c in checks if not c.passed}
    with pytest.raises(validate.ValidationError):
        validate.assert_all_passed(checks)


def test_validate_entities_and_reports_and_aliases():
    long = pd.DataFrame(
        {
            "incident_id": [1, 1],
            "role": ["deployer", "bogus"],
            "entity_id": ["a", "a"],
            "entity": ["A", "A"],
            "entity_type": ["organisation", "nonsense"],
        }
    )
    failed = {c.name for c in validate.validate_entities(long, {1}) if not c.passed}
    assert failed == {"roles valid", "entity_type valid"}
    reports = pd.DataFrame(
        {
            "report_number": [1, 1],
            "title": [None, "t"],
            "url": [None, "u"],
            "date_published": pd.to_datetime(["1960-01-01", "2020-01-01"]),
            "text": ["x", None],
            "incident_ids": [[1], [99]],
        }
    )
    failed = {c.name for c in validate.validate_reports(reports, {1}, "20260101") if not c.passed}
    assert {
        "report_number unique",
        "no stub reports (title or url present)",
        "published dates plausible",
        "full text present",
        "linked incident ids exist",
    } == failed
    failed = {
        c.name for c in validate.validate_alias_map({"a": "b", "b": "c"}, {"b"}) if not c.passed
    }
    assert failed == {"canonical entities exist", "alias chains fully resolved"}


# ------------------------------------------------------------------ integration (skipped without data)
PROCESSED = clean.processed_path("incidents_clean")
needs_processed = pytest.mark.skipif(
    not PROCESSED.exists(), reason="run scripts/run_phase2_prepare.py first"
)


@needs_processed
def test_processed_tables_pass_validation_again():
    inc = clean.load_processed("incidents_clean")
    ie = clean.load_processed("incident_entities")
    rep = clean.load_processed("reports_clean")
    from aicidentia.config import load_config

    snap = str(load_config()["snapshot_date"])
    ids = set(inc["incident_id"])
    checks = (
        validate.validate_incidents(inc, snap)
        + validate.validate_entities(ie, ids)
        + validate.validate_reports(rep, ids, snap)
    )
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]
    assert inc["mit_risk_domain"].notna().sum() == inc["has_mit"].sum()
    assert not np.any(ie.duplicated(["incident_id", "role", "entity_id"]))
