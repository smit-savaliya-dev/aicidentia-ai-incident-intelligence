"""Phase 2 orchestration: raw AIID files -> validated parquet tables in data/processed/.

`build_processed` is deterministic given the raw files, `config/entity_aliases.yaml` and your manual
review files. It never edits data/raw or fills in your manual review columns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from aicidentia.config import load_config
from aicidentia.data import clean, duplicates, entities, load, validate

logger = logging.getLogger(__name__)


@dataclass
class PrepResults:
    """Tables, scalar facts and validation checks produced by `build_processed`."""

    snapshot_date: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    checks: list[validate.Check] = field(default_factory=list)


def _merge_map(aiid_pairs: pd.DataFrame, user_merges: dict[int, int]) -> dict[int, int]:
    """AIID merges plus your confirmed merges, resolved transitively."""
    pairs = pd.DataFrame(
        {
            "duplicate_incident_number": list(aiid_pairs["duplicate_incident_number"].astype(int))
            + list(user_merges),
            "true_incident_number": list(aiid_pairs["true_incident_number"].astype(int))
            + list(user_merges.values()),
        }
    )
    return clean.resolve_merge_map(pairs)


def apply_privacy_exclusions(
    entity_table: pd.DataFrame,
    exclude_unreviewed_candidates: bool,
    review_path: Path | None = None,
) -> tuple[set[str], dict[str, int]]:
    """Entity ids to remove from the network and dashboard.

    Always: entities marked as named persons (rule D3). Optionally (config `privacy`): every
    `individual_candidate` not explicitly cleared with is_named_person = no.
    """
    review_path = review_path or entities.REVIEW_PATH
    confirmed = entities.load_named_persons(review_path)
    excluded = set(confirmed)
    cleared: set[str] = set()
    if review_path.exists():
        review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
        answer = review["is_named_person"].str.strip().str.lower()
        cleared = set(review.loc[answer.isin({"no", "n", "false", "0"}), "entity_id"])
    candidates = set(
        entity_table.loc[entity_table["entity_type"] == "individual_candidate", "entity_id"]
    )
    unreviewed_candidates = candidates - confirmed - cleared
    if exclude_unreviewed_candidates:
        excluded |= unreviewed_candidates
    return excluded, {
        "named_persons": len(confirmed),
        "not_persons": len(cleared),
        "unreviewed_candidates": len(unreviewed_candidates),
        "excluded_total": len(excluded),
    }


def build_processed(write: bool = True) -> PrepResults:
    """Run the whole Phase 2 pipeline; raises ValidationError if any check fails."""
    cfg = load_config()
    snapshot_date = str(cfg["snapshot_date"])
    res = PrepResults(snapshot_date=snapshot_date)
    t, f = res.tables, res.facts

    excel = load.find_excel_export()
    snap_dir = load.find_snapshot_dir()
    if snap_dir is None:
        raise FileNotFoundError(
            "Phase 2 needs the snapshot (entity slugs, report links, full text)."
        )

    raw_inc = load.load_incidents(excel)
    dictionary = load.load_data_dictionary(excel)
    sheet_entities = clean.clean_entities(load.load_entities(excel))
    sheet_reports = load.load_reports(excel)
    snap_inc = load.load_snapshot_incidents(snap_dir)
    snap_reports = load.load_full_report_text(snap_dir)
    aiid_merges = load.load_snapshot_duplicates(snap_dir)
    aiid_entity_dups = pd.DataFrame(load.read_bson("entity_duplicates", snap_dir))

    # ---- incidents and merges
    incidents = clean.clean_incidents(raw_inc, dictionary)
    f["incidents_raw"] = len(incidents)
    dup_candidates = duplicates.find_similar_pairs(incidents)
    duplicates.write_review_file(dup_candidates)
    user_merges = duplicates.load_merge_decisions()
    merge_map = _merge_map(aiid_merges, user_merges)
    in_excel_merged = [i for i in merge_map if i in set(incidents["incident_id"])]
    incidents = clean.drop_merged_incidents(incidents, merge_map)
    incident_ids = set(incidents["incident_id"])
    f["aiid_merge_pairs"] = len(aiid_merges)
    f["merge_map_size"] = len(merge_map)
    f["merged_ids_present_in_excel"] = sorted(in_excel_merged)
    f["user_merges_applied"] = len(user_merges)
    f["near_duplicate_review"] = duplicates.review_progress()
    t["near_duplicate_candidates"] = dup_candidates

    # ---- entities
    names = dict(zip(sheet_entities["Entity ID"], sheet_entities["Name"], strict=True))
    long = entities.build_long_table(snap_inc, incident_ids | set(merge_map))
    long["incident_id"] = long["incident_id"].map(lambda i: merge_map.get(i, i))
    long = long[long["incident_id"].isin(incident_ids)].drop_duplicates().reset_index(drop=True)
    candidates, skipped = entities.build_alias_candidates(sheet_entities, long, aiid_entity_dups)
    entities.write_alias_file(candidates)
    alias_map = entities.active_alias_map()
    res.checks += validate.validate_alias_map(alias_map, set(names) | set(long["entity_id_raw"]))
    entities_before_alias = int(long["entity_id_raw"].nunique())
    long = entities.apply_aliases(long, alias_map, names)
    types = entities.classify_entity_types(long)
    entity_table = entities.build_entity_table(long, types)
    review_candidates = entities.build_review_candidates(entity_table, long, incidents)
    entities.write_review_file(review_candidates)
    excluded, privacy = apply_privacy_exclusions(
        entity_table,
        bool(cfg.get("privacy", {}).get("exclude_unreviewed_individual_candidates", False)),
    )
    long = long.merge(types, on="entity_id", how="left")
    long = long[~long["entity_id"].isin(excluded)].reset_index(drop=True)
    entity_table = entity_table[~entity_table["entity_id"].isin(excluded)].reset_index(drop=True)
    incident_entities = long[
        [
            "incident_id",
            "role",
            "entity_id",
            "entity",
            "entity_type",
            "entity_type_rule",
            "entity_id_raw",
        ]
    ]
    f["entity_aliases_applied"] = len(alias_map)
    f["entity_aliases_manual"] = len(entities.read_alias_file()["manual"])
    f["entity_aliases_skipped_aiid"] = len(skipped)
    f["entities_before_alias"] = entities_before_alias
    f["entities_after_alias"] = int(long["entity_id"].nunique())
    f["entity_review"] = entities.review_progress()
    f["privacy"] = privacy
    f["review_candidates"] = {
        "total": len(review_candidates),
        **{
            f"priority_{k}": int(v)
            for k, v in review_candidates["review_priority"].value_counts().items()
        },
    }
    t["alias_candidates"] = candidates
    t["alias_skipped"] = skipped
    t["entity_table"] = entity_table
    t["review_candidates"] = review_candidates

    # ---- reports
    reports = clean.clean_reports(
        sheet_reports, snap_reports, snap_inc, incident_ids, merge_map, snapshot_date
    )
    per_incident = clean.count_reports_per_incident(reports)
    incidents["report_count_clean"] = (
        incidents["incident_id"].map(per_incident).fillna(0).astype("int64")
    )
    f["stub_reports_dropped"] = len(sheet_reports) - len(reports)
    f["reports_clean"] = len(reports)
    f["reports_unlinked_kept"] = int((~reports["linked"]).sum())
    f["incidents_with_zero_clean_reports"] = int((incidents["report_count_clean"] == 0).sum())
    f["report_count_differs_from_published"] = int(
        (incidents["report_count_clean"] != incidents["report_count"]).sum()
    )

    # ---- validation
    res.checks += validate.validate_incidents(incidents, snapshot_date)
    res.checks += validate.validate_entities(incident_entities, incident_ids)
    res.checks += validate.validate_reports(reports, incident_ids, snapshot_date)
    validate.assert_all_passed(res.checks)

    t["incidents_clean"] = incidents
    t["incident_entities"] = incident_entities
    t["reports_clean"] = reports
    f["incidents_clean"] = len(incidents)
    f["incident_entities_rows"] = len(incident_entities)
    f["entities_in_table"] = len(entity_table)
    f["entity_type_counts"] = entity_table["entity_type"].value_counts().to_dict()
    f["incidents_with_deployer"] = int(
        incident_entities.query("role == 'deployer'")["incident_id"].nunique()
    )
    f["incidents_with_harmed"] = int(
        incident_entities.query("role == 'harmed'")["incident_id"].nunique()
    )
    f["validation_checks"] = len(res.checks)
    f["validation_failed"] = sum(1 for c in res.checks if not c.passed)

    if write:
        for name in ("incidents_clean", "incident_entities", "entity_table", "reports_clean"):
            clean.save_processed(t[name], name)
    return res
