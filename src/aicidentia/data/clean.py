"""Cleaning and preparation of AIID tables (CRISP-DM: Data Preparation).

Pure functions that turn the raw Excel/snapshot tables into the clean tables of Phase 2:
`incidents_clean`, `reports_clean` (entities live in `aicidentia.data.entities`). Raw files are
never modified; outputs are written to data/processed/ by `save_processed`.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aicidentia.config import PROJECT_ROOT, load_config
from aicidentia.data.audit import parse_multi_value, strip_domain_number

logger = logging.getLogger(__name__)

TAXONOMY_PREFIX = {"MIT": "mit", "GMF": "gmf", "CSETv1": "cset"}
# Excel columns that identify people or carry no analytical value; dropped from the clean tables.
DROPPED_INCIDENT_COLUMNS = ("Editors", "Date Modified", "Editor Notes")
# Entity columns are rebuilt from snapshot slugs (Excel strings are lossy), so they are dropped here.
ENTITY_COLUMNS = ("deployer", "developer", "harmed", "Implicated Systems")
# Report columns holding personal names of journalists/submitters: dropped (data minimisation).
DROPPED_REPORT_COLUMNS = ("Authors", "Submitters")
_MIN_PLAUSIBLE_REPORT_DATE = pd.Timestamp("1990-01-01")


# --------------------------------------------------------------------------------------
# Text and time
# --------------------------------------------------------------------------------------


def normalise_whitespace(value: object) -> str | None:
    """Collapse runs of whitespace and strip; empty or missing becomes None."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def assemble_incident_text(title: pd.Series, description: pd.Series) -> pd.Series:
    """Modelling text: title + ". " + description (no doubled full stop if the title already ends in one)."""
    title = title.fillna("").astype(str)
    description = description.fillna("").astype(str)
    joiner = np.where(title.str.contains(r"[.!?]$"), " ", ". ")
    combined = pd.Series(title.to_numpy() + joiner + description.to_numpy(), index=title.index)
    return combined.str.strip(" .").map(lambda s: s + "." if s and s[-1] not in ".!?" else s)


def add_time_columns(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add `year`, `quarter` (1-4) and `year_quarter` (e.g. '2024Q3') derived from `date_col`."""
    out = df.copy()
    out["year"] = out[date_col].dt.year.astype("int64")
    out["quarter"] = out[date_col].dt.quarter.astype("int64")
    out["year_quarter"] = out["year"].astype(str) + "Q" + out["quarter"].astype(str)
    return out


def filter_year_range(df: pd.DataFrame, date_col: str, start: int, end: int) -> pd.DataFrame:
    """Keep rows whose `date_col` year lies in [start, end] (inclusive)."""
    years = df[date_col].dt.year
    return df[(years >= start) & (years <= end)].copy()


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------


def snake_case(name: str) -> str:
    """'Technical Failure (Potential)' -> 'technical_failure_potential'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _is_bool_like(values: pd.Series) -> bool:
    uniques = {str(v).strip().lower() for v in values.dropna().unique()}
    return bool(uniques) and uniques <= {"true", "false"}


def tidy_label_column(series: pd.Series) -> pd.Series:
    """Give one label column a sensible dtype.

    JSON-list cells -> list[str] (missing = empty list); numeric columns -> nullable numbers;
    True/False columns -> nullable boolean; everything else -> stripped strings (empty = missing).
    """
    non_null = series.dropna()
    if non_null.empty:
        return series.map(normalise_whitespace).astype("string")
    if non_null.astype(str).str.strip().str.startswith("[").any():
        return series.map(parse_multi_value)
    if _is_bool_like(non_null):
        mapped = series.map(lambda v: None if pd.isna(v) else str(v).strip().lower() == "true")
        return mapped.astype("boolean")
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().mean() > 0.95:
        out = pd.to_numeric(series, errors="coerce")
        return out.astype("Int64") if (numeric.dropna() % 1 == 0).all() else out
    return series.map(normalise_whitespace).astype("string")


def tidy_labels(incidents: pd.DataFrame, dictionary: pd.DataFrame) -> pd.DataFrame:
    """Tidy MIT / GMF / CSETv1 columns: prefixed snake_case names, proper dtypes, domain numbers stripped.

    Group membership comes from the Data Dictionary. The CSETv1 `Entities` column (a JSON structure)
    is kept as a JSON string named `cset_entities_json`.
    """
    out = {}
    for group, prefix in TAXONOMY_PREFIX.items():
        for column in dictionary.loc[dictionary["Group"] == group, "Column"]:
            if column not in incidents.columns:
                continue
            name = f"{prefix}_{snake_case(column)}"
            if group == "CSETv1" and column == "Entities":
                out["cset_entities_json"] = (
                    incidents[column].map(normalise_whitespace).astype("string")
                )
                continue
            tidy = tidy_label_column(incidents[column])
            if group == "MIT" and column in ("Risk Domain", "Risk Subdomain"):
                tidy = tidy.map(
                    lambda v: strip_domain_number(v) if isinstance(v, str) else v
                ).astype("string")
            out[name] = tidy
    return pd.DataFrame(out, index=incidents.index)


# --------------------------------------------------------------------------------------
# Incidents
# --------------------------------------------------------------------------------------


def clean_incidents(incidents: pd.DataFrame, dictionary: pd.DataFrame) -> pd.DataFrame:
    """Excel Incidents sheet -> `incidents_clean`.

    Columns: incident_id, date, year, quarter, year_quarter, title, description, text, report_count,
    taxonomy flags (has_mit / has_gmf / has_cset) and the tidied label columns. Entity columns,
    editor ids/notes and the modified date are dropped (see DROPPED_INCIDENT_COLUMNS).
    """
    base = pd.DataFrame(
        {
            "incident_id": incidents["Incident ID"].astype("int64"),
            "date": pd.to_datetime(incidents["date"]),
            "title": incidents["title"].map(normalise_whitespace),
            "description": incidents["description"].map(normalise_whitespace),
            "report_count": incidents["report_count"].astype("int64"),
        }
    )
    base = add_time_columns(base)
    base["text"] = assemble_incident_text(base["title"], base["description"])
    sources = incidents["Data Sources"].fillna("")
    for taxonomy, flag in (("MIT", "has_mit"), ("GMF", "has_gmf"), ("CSETv1", "has_cset")):
        base[flag] = sources.str.contains(taxonomy).to_numpy()
    labels = tidy_labels(incidents, dictionary)
    ordered = [
        "incident_id",
        "date",
        "year",
        "quarter",
        "year_quarter",
        "title",
        "description",
        "text",
        "report_count",
        "has_mit",
        "has_gmf",
        "has_cset",
    ]
    out = pd.concat([base[ordered], labels], axis=1)
    return out.sort_values("incident_id").reset_index(drop=True)


def resolve_merge_map(pairs: pd.DataFrame) -> dict[int, int]:
    """AIID merge pairs (duplicate -> true incident) resolved transitively: 342 -> 88 -> 1057 gives 342 -> 1057."""
    direct = dict(
        zip(
            pairs["duplicate_incident_number"].astype(int),
            pairs["true_incident_number"].astype(int),
            strict=True,
        )
    )
    resolved: dict[int, int] = {}
    for dup in direct:
        seen, current = {dup}, direct[dup]
        while current in direct and direct[current] not in seen:
            seen.add(current)
            current = direct[current]
        resolved[dup] = current
    return resolved


def apply_merge_map(df: pd.DataFrame, column: str, merge_map: dict[int, int]) -> pd.DataFrame:
    """Replace merged-away incident ids in `column` by the kept id and drop rows that become identical."""
    out = df.copy()
    out[column] = out[column].map(lambda i: merge_map.get(i, i))
    return out.drop_duplicates().reset_index(drop=True)


def drop_merged_incidents(incidents: pd.DataFrame, merge_map: dict[int, int]) -> pd.DataFrame:
    """Remove incidents that were merged into another one."""
    return incidents[~incidents["incident_id"].isin(merge_map)].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Entities sheet
# --------------------------------------------------------------------------------------


def clean_entities(entities: pd.DataFrame) -> pd.DataFrame:
    """Entities sheet -> (Entity ID, Name): trim, drop rows without an id, fall back to the slug for missing names."""
    out = entities[["Entity ID", "Name"]].copy()
    out["Entity ID"] = out["Entity ID"].map(normalise_whitespace)
    out["Name"] = out["Name"].map(normalise_whitespace)
    out = out[out["Entity ID"].notna()].drop_duplicates("Entity ID")
    out["Name"] = out["Name"].fillna(out["Entity ID"].str.replace("-", " ").str.capitalize())
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------------------


def is_stub_report(reports: pd.DataFrame) -> pd.Series:
    """True for placeholder reports: no title and no URL."""
    no_title = reports["Title"].isna() | (reports["Title"].astype(str).str.strip() == "")
    return no_title & reports["URL"].isna()


def drop_stub_reports(reports: pd.DataFrame) -> pd.DataFrame:
    """Remove placeholder reports (no title, no URL, about one word of text)."""
    return reports[~is_stub_report(reports)].reset_index(drop=True)


def clean_reports(
    reports_sheet: pd.DataFrame,
    snapshot_reports: pd.DataFrame,
    snapshot_incidents: pd.DataFrame,
    incident_ids: set[int],
    merge_map: dict[int, int] | None = None,
    snapshot_date: str | None = None,
) -> pd.DataFrame:
    """Excel Reports sheet + snapshot text and links -> `reports_clean`.

    Stub reports are dropped; journalist/submitter names are not carried over. Adds `text` (full
    article text from the snapshot), `text_words`, `incident_ids` (list; the Excel sheet has no
    incident column) and `linked`. Implausible publication dates become missing.
    """
    merge_map = merge_map or {}
    real = drop_stub_reports(reports_sheet).drop(
        columns=list(DROPPED_REPORT_COLUMNS), errors="ignore"
    )
    real = real.rename(
        columns={
            "Report Number": "report_number",
            "Title": "title",
            "URL": "url",
            "Source Domain": "source_domain",
            "Date Published": "date_published",
            "Date Downloaded": "date_downloaded",
            "Language": "language",
            "Tags": "tags",
            "Description": "description",
            "Is Incident Report": "is_incident_report",
        }
    )
    real["report_number"] = real["report_number"].astype("int64")
    real["is_incident_report"] = real["is_incident_report"].astype(str).str.lower().eq("true")
    upper = pd.to_datetime(snapshot_date, format="%Y%m%d") if snapshot_date else pd.Timestamp.max
    bad_date = (real["date_published"] < _MIN_PLAUSIBLE_REPORT_DATE) | (
        real["date_published"] > upper
    )
    real["date_published"] = real["date_published"].mask(bad_date)
    for column in ("title", "description"):
        real[column] = real[column].map(normalise_whitespace)

    text = snapshot_reports.set_index("report_number")["text"].map(normalise_whitespace)
    real["text"] = real["report_number"].map(text)
    real["text_words"] = real["text"].fillna("").str.split().str.len().astype("int64")

    links = (
        snapshot_incidents[["incident_id", "reports"]]
        .explode("reports")
        .dropna()
        .astype({"reports": "int64"})
    )
    links["incident_id"] = links["incident_id"].map(lambda i: merge_map.get(i, i))
    links = links[links["incident_id"].isin(incident_ids)].drop_duplicates()
    per_report = links.groupby("reports")["incident_id"].agg(lambda s: sorted(int(i) for i in s))
    real["incident_ids"] = (
        real["report_number"].map(per_report).map(lambda v: v if isinstance(v, list) else [])
    )
    real["linked"] = real["incident_ids"].map(len) > 0
    columns = [
        "report_number",
        "title",
        "url",
        "source_domain",
        "date_published",
        "date_downloaded",
        "language",
        "tags",
        "description",
        "is_incident_report",
        "text",
        "text_words",
        "incident_ids",
        "linked",
    ]
    return real[columns].sort_values("report_number").reset_index(drop=True)


def count_reports_per_incident(reports: pd.DataFrame) -> pd.Series:
    """Number of non-stub linked reports per incident id."""
    exploded = reports[["report_number", "incident_ids"]].explode("incident_ids").dropna()
    return (
        exploded.groupby("incident_ids")["report_number"]
        .nunique()
        .astype("int64")
        .rename("report_count_clean")
    )


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def processed_path(name: str) -> Path:
    """data/processed/<name>.parquet."""
    return PROJECT_ROOT / load_config()["paths"]["processed"] / f"{name}.parquet"


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Write a dataframe to data/processed/<name>.parquet and return the path."""
    path = processed_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %s (%d rows, %d columns)", path.name, len(df), df.shape[1])
    return path


def load_processed(name: str) -> pd.DataFrame:
    """Read data/processed/<name>.parquet."""
    return pd.read_parquet(processed_path(name))


def dumps_compact(value: Any) -> str:
    """Compact JSON for logging/registry convenience."""
    return json.dumps(value, ensure_ascii=False, default=str)
