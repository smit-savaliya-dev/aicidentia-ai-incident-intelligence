"""Validation checks for the clean tables. A failed check raises `ValidationError`."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from aicidentia.data.entities import ENTITY_TYPES, ROLES

logger = logging.getLogger(__name__)


class ValidationError(AssertionError):
    """Raised when one or more validation checks fail."""


@dataclass(frozen=True)
class Check:
    """One named check, whether it passed, and a short detail (counts of offending rows)."""

    table: str
    name: str
    passed: bool
    detail: str = ""


def validate_incidents(df: pd.DataFrame, snapshot_date: str) -> list[Check]:
    """Ids present and unique, valid dates, consistent year/quarter, non-empty text."""
    last = pd.to_datetime(snapshot_date, format="%Y%m%d")
    t = "incidents_clean"
    bad_date = df["date"].isna() | (df["date"] < "1900-01-01") | (df["date"] > last)
    return [
        Check(t, "no null incident_id", bool(df["incident_id"].notna().all())),
        Check(
            t,
            "incident_id unique",
            bool(df["incident_id"].is_unique),
            f"{int(df['incident_id'].duplicated().sum())} duplicates",
        ),
        Check(
            t,
            "dates valid (1900 .. snapshot date)",
            not bad_date.any(),
            f"{int(bad_date.sum())} invalid",
        ),
        Check(t, "year equals date.year", bool((df["date"].dt.year == df["year"]).all())),
        Check(t, "quarter in 1..4", bool(df["quarter"].between(1, 4).all())),
        Check(t, "title non-empty", bool(df["title"].fillna("").str.len().gt(0).all())),
        Check(t, "description non-empty", bool(df["description"].fillna("").str.len().gt(0).all())),
        Check(
            t,
            "text starts with title",
            bool(
                [
                    txt.startswith(ttl.rstrip(".!? "))
                    for txt, ttl in zip(df["text"], df["title"], strict=True)
                ].count(False)
                == 0
            ),
        ),
        Check(t, "report_count >= 0", bool((df["report_count"] >= 0).all())),
    ]


def validate_entities(long: pd.DataFrame, incident_ids: set[int]) -> list[Check]:
    """No nulls, valid roles/types, every incident known, no duplicate (incident, role, entity)."""
    t = "incident_entities"
    required = ["incident_id", "entity_id", "entity", "role", "entity_type"]
    dup = long.duplicated(["incident_id", "role", "entity_id"]).sum()
    return [
        Check(t, "no nulls in key columns", bool(long[required].notna().all().all())),
        Check(t, "roles valid", bool(long["role"].isin(ROLES).all())),
        Check(t, "entity_type valid", bool(long["entity_type"].isin(ENTITY_TYPES).all())),
        Check(
            t,
            "every incident_id exists in incidents_clean",
            bool(long["incident_id"].isin(incident_ids).all()),
        ),
        Check(t, "no duplicate (incident, role, entity)", dup == 0, f"{int(dup)} duplicates"),
    ]


def validate_reports(
    reports: pd.DataFrame, incident_ids: set[int], snapshot_date: str
) -> list[Check]:
    """Unique report numbers, no stubs, plausible dates, text present, links point at known incidents."""
    t = "reports_clean"
    last = pd.to_datetime(snapshot_date, format="%Y%m%d")
    dates = reports["date_published"].dropna()
    linked_ids = {i for ids in reports["incident_ids"] for i in ids}
    return [
        Check(t, "report_number unique", bool(reports["report_number"].is_unique)),
        Check(
            t,
            "no stub reports (title or url present)",
            bool((reports["title"].notna() | reports["url"].notna()).all()),
        ),
        Check(
            t,
            "published dates plausible",
            bool(((dates >= "1990-01-01") & (dates <= last)).all()),
            f"{int(reports['date_published'].isna().sum())} missing",
        ),
        Check(
            t,
            "full text present",
            bool(reports["text"].fillna("").str.len().gt(0).all()),
            f"{int(reports['text'].isna().sum())} missing",
        ),
        Check(t, "linked incident ids exist", linked_ids <= incident_ids),
    ]


def validate_alias_map(alias_map: dict[str, str], known_entities: set[str]) -> list[Check]:
    """Aliases resolve to existing entities (or to an entity that is itself used) and are not self-referential."""
    t = "entity_aliases"
    unknown = [c for c in alias_map.values() if c not in known_entities]
    return [
        Check(t, "no self-referential alias", all(a != c for a, c in alias_map.items())),
        Check(
            t,
            "canonical entities exist",
            not unknown,
            f"{len(unknown)} unknown: {sorted(set(unknown))[:5]}",
        ),
        Check(
            t, "alias chains fully resolved", all(c not in alias_map for c in alias_map.values())
        ),
    ]


def assert_all_passed(checks: list[Check]) -> None:
    """Log every check and raise ValidationError listing the failures."""
    for check in checks:
        logger.info(
            "%s %-45s %s %s",
            "PASS" if check.passed else "FAIL",
            f"{check.table}: {check.name}",
            "",
            check.detail,
        )
    failed = [c for c in checks if not c.passed]
    if failed:
        raise ValidationError("; ".join(f"{c.table}: {c.name} ({c.detail})" for c in failed))
