"""Phase 1 data audit (CRISP-DM: data understanding).

Pure functions that profile the AIID Excel export and snapshot and return dataframes or
plain dicts. Nothing here writes files; `run_audit` collects everything into one
`AuditResults` object that the notebook, the figure code and the document builder consume.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aicidentia.config import PROJECT_ROOT, load_config
from aicidentia.data import load

logger = logging.getLogger(__name__)

TAXONOMY_GROUPS = ("MIT", "GMF", "CSETv1")
ENTITY_ROLES = ("deployer", "developer", "harmed", "system")
# Snapshot column that carries each role's slug list.
_SNAPSHOT_ROLE_COLUMN = {
    "deployer": "deployer",
    "developer": "developer",
    "harmed": "harmed",
    "system": "implicated_systems",
}
_EXCEL_ROLE_COLUMN = {
    "deployer": "deployer",
    "developer": "developer",
    "harmed": "harmed",
    "system": "Implicated Systems",
}


@dataclass
class AuditResults:
    """Everything the Phase 1 audit produces: named tables plus scalar facts."""

    snapshot_date: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    scalars: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------


def parse_percent(value: object) -> float | None:
    """'84%' -> 0.84; None if the value is not a percentage."""
    match = re.fullmatch(r"\s*([\d.]+)\s*%\s*", str(value))
    return float(match.group(1)) / 100 if match else None


def parse_multi_value(value: object) -> list[str]:
    """Split a label cell into its values.

    Cells are either plain strings or JSON lists such as '["a", "b"]'. Empty cells give [].
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        return [str(v).strip() for v in parsed if str(v).strip()]
    return [text]


def strip_domain_number(label: str) -> str:
    """'1. Discrimination and Toxicity' -> 'Discrimination and Toxicity' (also for 1.2 sub-ids)."""
    return re.sub(r"^\d+(\.\d+)*\.?\s*", "", label).strip()


def normalised_entropy(counts: pd.Series) -> float:
    """Shannon entropy of a count vector divided by log(k): 1 = perfectly balanced."""
    p = counts[counts > 0] / counts.sum()
    if len(p) < 2:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


# --------------------------------------------------------------------------------------
# Sheet-level profiling and the Data Dictionary cross-check
# --------------------------------------------------------------------------------------


def profile_table(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """One row per column: dtype, non-null count, fill rate, number of unique values."""
    rows = []
    for col in df.columns:
        series = df[col]
        non_empty = series.notna() & (series.astype("string").str.strip() != "")
        rows.append(
            {
                "table": table,
                "column": col,
                "dtype": str(series.dtype),
                "non_empty": int(non_empty.sum()),
                "fill_rate": float(non_empty.mean()),
                "n_unique": int(series[non_empty].astype("string").nunique()),
            }
        )
    return pd.DataFrame(rows)


def sheet_shapes(excel_path: Path | None = None) -> pd.DataFrame:
    """Rows/columns of each sheet after banner removal, plus the row count the banner advertises."""
    rows = []
    for sheet in ("Incidents", "Reports", "Entities", "Data Dictionary", "Coverage Map"):
        df = load.load_sheet(sheet, excel_path)
        banner = load.sheet_banner(sheet, excel_path)
        advertised = re.search(r"([\d,]+)\s+(incidents|reports|entities)", banner)
        rows.append(
            {
                "sheet": sheet,
                "rows": len(df),
                "columns": df.shape[1],
                "banner_count": int(advertised.group(1).replace(",", "")) if advertised else None,
            }
        )
    return pd.DataFrame(rows)


def cross_check_dictionary(
    incidents: pd.DataFrame, dictionary: pd.DataFrame, tolerance: float = 0.01
) -> pd.DataFrame:
    """Compare observed fill rates in Incidents with the Data Dictionary's stated fill rates.

    The dictionary rounds to whole percent, so a gap above `tolerance` (1 percentage point)
    is flagged. Columns missing on either side are flagged too.
    """
    dd = dictionary.copy()
    dd["stated_fill"] = dd["Fill Rate"].map(parse_percent)
    observed = profile_table(incidents, "Incidents").set_index("column")["fill_rate"]
    rows = []
    for rec in dd.itertuples():
        obs = observed.get(rec.Column, np.nan)
        stated = rec.stated_fill
        gap = (obs - stated) if pd.notna(obs) and stated is not None else np.nan
        rows.append(
            {
                "column": rec.Column,
                "group": rec.Group,
                "source": rec.Source,
                "stated_fill": stated,
                "observed_fill": obs,
                "gap": gap,
                "flag": "missing in sheet"
                if pd.isna(obs)
                else ("gap > 1pp" if abs(gap) > tolerance else ""),
            }
        )
    out = pd.DataFrame(rows)
    undocumented = sorted(set(incidents.columns) - set(dd["Column"]))
    if undocumented:
        extra = pd.DataFrame(
            {"column": undocumented, "group": "", "source": "", "flag": "not in dictionary"}
        )
        out = pd.concat([out, extra], ignore_index=True)
    return out


# --------------------------------------------------------------------------------------
# Incidents: time, duplicates, text length
# --------------------------------------------------------------------------------------


def incidents_per_year(incidents: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """Incidents per year, flagging the snapshot year as partial.

    `complete` is False for the snapshot's calendar year (the year is still running);
    `last_incident_date` shows how far into that year the data reaches.
    """
    snap = pd.to_datetime(str(snapshot_date), format="%Y%m%d")
    grouped = incidents.groupby("year")
    out = grouped.size().rename("incidents").to_frame()
    out["share"] = out["incidents"] / out["incidents"].sum()
    out["complete"] = out.index < snap.year
    out["last_incident_date"] = grouped["date"].max().dt.strftime("%Y-%m-%d")
    return out.reset_index()


def monthly_counts(incidents: pd.DataFrame, last_n: int = 24) -> pd.DataFrame:
    """Incidents per month for the most recent `last_n` months (reporting-lag check)."""
    monthly = incidents.set_index("date").resample("MS").size().rename("incidents")
    return monthly.tail(last_n).rename_axis("month").reset_index()


def recommend_year_range(
    per_year: pd.DataFrame, min_incidents: int = 20
) -> dict[str, int | list[int]]:
    """Longest run of complete years ending at the last complete year in which every year has
    at least `min_incidents` incidents. Returns start, end and the excluded sparse years."""
    complete = per_year[per_year["complete"]].set_index("year")["incidents"]
    end = int(complete.index.max())
    start = end
    for year in range(end, int(complete.index.min()) - 1, -1):
        if complete.get(year, 0) < min_incidents:
            break
        start = year
    sparse = [int(y) for y in complete.index if y < start]
    return {"start": start, "end": end, "min_incidents": min_incidents, "excluded_years": sparse}


def merge_chains(merged_pairs: pd.DataFrame) -> list[list[int]]:
    """Merge paths with more than one hop, e.g. 342 -> 88 -> 1057 (a 'true' incident that is itself merged)."""
    target = dict(
        zip(
            merged_pairs["duplicate_incident_number"],
            merged_pairs["true_incident_number"],
            strict=True,
        )
    )
    chains = []
    for start in target:
        path = [int(start)]
        while path[-1] in target and int(target[path[-1]]) not in path:
            path.append(int(target[path[-1]]))
        if len(path) > 2:
            chains.append(path)
    return chains


def mit_record_facts(snapshot_dir: Path) -> dict[str, Any]:
    """Size and annotation metadata of the raw MIT classification records (snapshot BSON)."""
    docs = [d for d in load.read_bson("classifications", snapshot_dir) if d["namespace"] == "MIT"]
    return {
        "mit_records": len(docs),
        "mit_records_with_notes": int(sum(1 for d in docs if str(d.get("notes", "")).strip())),
        "mit_records_published": int(sum(1 for d in docs if d.get("publish"))),
    }


def _normalise_title(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def duplicate_checks(
    incidents: pd.DataFrame,
    snapshot_incidents: pd.DataFrame | None,
    merged_pairs: pd.DataFrame | None,
    near_threshold: float = 0.8,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Exact and near-duplicate checks.

    Returns scalar findings and a table of the most similar incident pairs (TF-IDF cosine
    on title + description). Near-duplicates are *candidates*: AIID legitimately holds many
    similar-but-separate incidents, so they are reviewed in Phase 2 rather than dropped here.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    facts: dict[str, Any] = {
        "duplicate_ids": int(incidents["Incident ID"].duplicated().sum()),
        "duplicate_titles_exact": int(incidents["title"].duplicated().sum()),
        "duplicate_titles_normalised": int(
            incidents["title"].map(_normalise_title).duplicated().sum()
        ),
        "duplicate_descriptions_exact": int(incidents["description"].duplicated().sum()),
    }
    text = (incidents["title"].fillna("") + ". " + incidents["description"].fillna("")).tolist()
    tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(text)
    sim = (tfidf @ tfidf.T).tocoo()
    mask = sim.row < sim.col
    pairs = pd.DataFrame({"i": sim.row[mask], "j": sim.col[mask], "cosine": sim.data[mask]})
    ids = incidents["Incident ID"].to_numpy()
    titles = incidents["title"].to_numpy()
    pairs["incident_a"], pairs["incident_b"] = ids[pairs["i"]], ids[pairs["j"]]
    pairs["title_a"], pairs["title_b"] = titles[pairs["i"]], titles[pairs["j"]]
    pairs = pairs.sort_values("cosine", ascending=False).drop(columns=["i", "j"])
    for threshold in (0.7, 0.8, 0.9):
        facts[f"near_duplicate_pairs_cosine_ge_{int(threshold * 100)}"] = int(
            (pairs["cosine"] >= threshold).sum()
        )
    facts["near_duplicate_threshold"] = near_threshold

    if merged_pairs is not None and snapshot_incidents is not None:
        in_snapshot = set(snapshot_incidents["incident_id"])
        in_excel = set(incidents["Incident ID"])
        dup_ids = set(merged_pairs["duplicate_incident_number"])
        true_ids = set(merged_pairs["true_incident_number"])
        facts["merged_pairs_listed"] = len(merged_pairs)
        facts["merged_duplicate_ids_still_in_snapshot"] = sorted(
            int(i) for i in dup_ids & in_snapshot
        )
        facts["merged_duplicate_ids_in_excel"] = sorted(int(i) for i in dup_ids & in_excel)
        facts["merged_true_ids_missing_from_excel"] = sorted(int(i) for i in true_ids - in_excel)
        facts["merge_chains"] = merge_chains(merged_pairs)
    return facts, pairs.head(25).reset_index(drop=True)


def text_length_stats(incidents: pd.DataFrame) -> pd.DataFrame:
    """Words and characters of title and description: mean, median, 5th/95th percentile, max."""
    rows = []
    for column in ("title", "description"):
        text = incidents[column].fillna("")
        for unit, values in (("words", text.str.split().str.len()), ("characters", text.str.len())):
            rows.append(
                {
                    "field": column,
                    "unit": unit,
                    "empty": int((values == 0).sum()),
                    "mean": values.mean(),
                    "median": values.median(),
                    "p05": values.quantile(0.05),
                    "p95": values.quantile(0.95),
                    "max": values.max(),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Labels: MIT, GMF, CSETv1
# --------------------------------------------------------------------------------------


def label_columns_by_group(dictionary: pd.DataFrame) -> dict[str, list[str]]:
    """Columns of each taxonomy group, taken from the Data Dictionary 'Group' column."""
    return {
        group: dictionary.loc[dictionary["Group"] == group, "Column"].tolist()
        for group in TAXONOMY_GROUPS
    }


def _column_kind(series: pd.Series) -> str:
    """'numeric', 'free_text' or 'categorical' (heuristic used only to decide how to summarise)."""
    values = series.dropna().astype(str)
    if values.empty:
        return "empty"
    if pd.to_numeric(values, errors="coerce").notna().mean() > 0.95:
        return "numeric"
    flat = [v for cell in values for v in parse_multi_value(cell)]
    if len(flat) == 0:
        return "empty"
    unique_share = len(set(flat)) / len(flat)
    if np.median([len(v) for v in flat]) > 60 or (unique_share > 0.5 and len(set(flat)) > 50):
        return "free_text"
    return "categorical"


def label_profile(incidents: pd.DataFrame, columns_by_group: dict[str, list[str]]) -> pd.DataFrame:
    """Coverage, number of classes, class balance and single/multi-valued status per label column.

    Multi-valued cells (JSON lists) are flattened: class counts are label instances, coverage
    is incidents with a non-empty cell. `imbalance_ratio` = largest / smallest class count.
    """
    rows = []
    for group, columns in columns_by_group.items():
        for col in columns:
            if col not in incidents.columns:
                continue
            kind = _column_kind(incidents[col])
            parsed = incidents[col].map(parse_multi_value)
            n_values = parsed.map(len)
            covered = int((n_values > 0).sum())
            row: dict[str, Any] = {
                "group": group,
                "column": col,
                "kind": kind,
                "coverage_n": covered,
                "coverage_pct": covered / len(incidents),
                "value_type": ("multi-valued" if (n_values > 1).any() else "single-valued")
                if covered
                else "",
                "share_multi": float((n_values > 1).sum() / covered) if covered else np.nan,
            }
            if kind == "categorical":
                counts = pd.Series([v for cell in parsed for v in cell]).value_counts()
                row.update(
                    {
                        "n_classes": len(counts),
                        "top_class": counts.index[0],
                        "top_share": counts.iloc[0] / counts.sum(),
                        "min_class_n": int(counts.iloc[-1]),
                        "imbalance_ratio": counts.iloc[0] / counts.iloc[-1],
                        "norm_entropy": normalised_entropy(counts),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def class_distribution(
    incidents: pd.DataFrame, column: str, strip_numbers: bool = False
) -> pd.DataFrame:
    """Class counts and shares for one label column (multi-valued cells are flattened)."""
    values = [v for cell in incidents[column].map(parse_multi_value) for v in cell]
    if strip_numbers:
        values = [strip_domain_number(v) for v in values]
    counts = pd.Series(values).value_counts()
    out = counts.rename_axis("class").rename("n").reset_index()
    out["share"] = out["n"] / out["n"].sum()
    return out


def label_coverage_by_year(incidents: pd.DataFrame, column: str) -> pd.DataFrame:
    """Share of incidents per year that carry a non-empty `column` (time-split feasibility)."""
    has = incidents[column].map(lambda v: len(parse_multi_value(v)) > 0)
    out = (
        has.groupby(incidents["year"])
        .agg(["sum", "size"])
        .rename(columns={"sum": "labelled", "size": "incidents"})
    )
    out["coverage"] = out["labelled"] / out["incidents"]
    return out.reset_index()


def class_counts_by_period(
    incidents: pd.DataFrame, column: str, test_from_year: int = 2024
) -> pd.DataFrame:
    """Class counts before and from `test_from_year` (feasibility of a train <= 2023 / test >= 2024 split)."""
    frame = incidents.assign(
        values=incidents[column].map(parse_multi_value),
        period=np.where(incidents["year"] < test_from_year, "train", "test"),
    ).explode("values")
    frame = frame[frame["values"].notna()]
    out = frame.pivot_table(
        index="values", columns="period", values="year", aggfunc="size", fill_value=0
    )
    out = out.reindex(columns=["train", "test"], fill_value=0).rename_axis("class").reset_index()
    out = out.rename(
        columns={"train": f"n_before_{test_from_year}", "test": f"n_from_{test_from_year}"}
    )
    out["test_share"] = out[f"n_from_{test_from_year}"] / (
        out[f"n_before_{test_from_year}"] + out[f"n_from_{test_from_year}"]
    )
    return out.sort_values(f"n_from_{test_from_year}", ascending=False).reset_index(drop=True)


def coverage_map_check(incidents: pd.DataFrame, coverage_map: pd.DataFrame) -> pd.DataFrame:
    """Compare the Coverage Map sheet's counts with counts derived from 'Data Sources'."""
    derived = incidents["Data Sources"].fillna("(none)").value_counts().rename("derived_n")
    stated = coverage_map[coverage_map["Data Sources"].fillna("TOTAL") != "TOTAL"].copy()
    stated["Data Sources"] = stated["Data Sources"].fillna("(none)").replace("None", "(none)")
    stated["stated_n"] = pd.to_numeric(stated["Incidents"], errors="coerce")
    out = stated.set_index("Data Sources")[["stated_n"]].join(derived, how="outer")
    out["match"] = out["stated_n"] == out["derived_n"]
    return out.reset_index()


def mit_provenance(taxa: list[dict[str, Any]]) -> dict[str, Any]:
    """Whether each taxonomy's classifications were produced automatically, per the taxa records."""
    return {
        t["namespace"]: {"automated": bool(t.get("automatedClassifications", False))}
        for t in taxa
        if t["namespace"] in ("MIT", "GMF", "CSETv1", "CSETv0")
    }


# --------------------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------------------


def entity_name_map(entities_sheet: pd.DataFrame) -> dict[str, str]:
    """slug -> display name from the Entities sheet."""
    return dict(zip(entities_sheet["Entity ID"], entities_sheet["Name"], strict=False))


def entity_long_table(snapshot_incidents: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    """One row per (incident, role, entity slug) from the snapshot slug lists."""
    rows = []
    for rec in snapshot_incidents.itertuples():
        for role, col in _SNAPSHOT_ROLE_COLUMN.items():
            for slug in getattr(rec, col):
                rows.append((rec.incident_id, role, slug, name_map.get(slug)))
    return pd.DataFrame(rows, columns=["incident_id", "role", "entity_id", "name"])


def entity_audit(
    incidents: pd.DataFrame,
    snapshot_incidents: pd.DataFrame,
    entities_sheet: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame]]:
    """Format, size and top-20 of the deployer / developer / harmed / system fields.

    Returns scalar facts, a per-role summary table, and the top-20 table per role (counted as
    number of incidents). Uses the snapshot slug lists, because the Excel columns join display
    names with ', ' and some entity names contain commas themselves.
    """
    name_map = entity_name_map(entities_sheet)
    in_excel = set(incidents["Incident ID"])
    snap = snapshot_incidents[snapshot_incidents["incident_id"].isin(in_excel)]
    long = entity_long_table(snap, name_map)

    facts: dict[str, Any] = {
        "entities_sheet_rows": len(entities_sheet),
        "entity_names_containing_comma": int(entities_sheet["Name"].str.contains(",").sum()),
        "slugs_not_in_entities_sheet": int(long["name"].isna().sum()),
    }
    summary_rows, tops = [], {}
    for role in ENTITY_ROLES:
        sub = long[long["role"] == role]
        per_incident = sub.groupby("incident_id").size().reindex(snap["incident_id"], fill_value=0)
        excel_col = incidents[_EXCEL_ROLE_COLUMN[role]]
        example = excel_col.dropna().astype(str).iloc[1] if excel_col.notna().sum() > 1 else ""
        summary_rows.append(
            {
                "role": role,
                "excel_column": _EXCEL_ROLE_COLUMN[role],
                "excel_format": "names joined by ', '",
                "snapshot_format": "list of slugs",
                "incidents_with_value": int((per_incident > 0).sum()),
                "fill_rate": float((per_incident > 0).mean()),
                "unique_entities": int(sub["entity_id"].nunique()),
                "entity_mentions": len(sub),
                "mean_per_incident": float(per_incident.mean()),
                "max_per_incident": int(per_incident.max()),
                "excel_example": example,
            }
        )
        top = (
            sub.groupby(["entity_id", "name"], dropna=False)["incident_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(20)
            .rename("incidents")
            .reset_index()
        )
        top["share_of_incidents"] = top["incidents"] / len(snap)
        tops[role] = top
    used = long["entity_id"].nunique()
    facts["entities_used_in_any_role"] = int(used)
    facts["entities_never_referenced"] = int(len(entities_sheet) - used)

    # Do the Excel names agree with the snapshot slugs mapped through the Entities sheet?
    deployer_mapped = snap.set_index("incident_id")["deployer"].map(
        lambda slugs: ", ".join(name_map.get(s, s) for s in slugs).lower()
    )
    deployer_excel = incidents.set_index("Incident ID")["deployer"].fillna("").str.lower()
    shared = deployer_mapped.index.intersection(deployer_excel.index)
    facts["deployer_excel_equals_mapped_names_pct"] = float(
        (deployer_mapped.loc[shared] == deployer_excel.loc[shared]).mean()
    )
    return facts, pd.DataFrame(summary_rows), tops


# --------------------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------------------


def reports_audit(
    incidents: pd.DataFrame,
    reports_sheet: pd.DataFrame,
    snapshot_incidents: pd.DataFrame | None,
    snapshot_reports: pd.DataFrame | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Reports per incident, report metadata and (with the snapshot) full-text availability.

    Returns scalar facts and the per-incident report-count table.
    """
    counts = incidents.set_index("Incident ID")["report_count"]
    title_missing = reports_sheet["Title"].isna() | (
        reports_sheet["Title"].astype(str).str.strip() == ""
    )
    url_missing = reports_sheet["URL"].isna()
    stub = title_missing & url_missing
    real = reports_sheet[~stub]
    facts: dict[str, Any] = {
        "reports_sheet_rows": len(reports_sheet),
        "reports_sheet_has_incident_id_column": "Incident ID" in reports_sheet.columns,
        "stub_reports": int(stub.sum()),
        "stub_reports_pct": float(stub.mean()),
        "stub_reports_epoch_placeholder_dates": int(
            (reports_sheet.loc[stub, "Date Published"] < "1990-01-01").sum()
        ),
        "real_report_date_min": str(real["Date Published"].min().date()),
        "real_report_date_max": str(real["Date Published"].max().date()),
        "report_count_mean": float(counts.mean()),
        "report_count_median": float(counts.median()),
        "report_count_p90": float(counts.quantile(0.9)),
        "report_count_max": int(counts.max()),
        "incidents_with_zero_reports": int((counts == 0).sum()),
        "incidents_with_one_report": int((counts == 1).sum()),
        "languages": reports_sheet["Language"].value_counts().head(5).to_dict(),
        "is_incident_report_true_pct": float(
            reports_sheet["Is Incident Report"].astype(str).str.lower().eq("true").mean()
        ),
        "top_source_domains": real["Source Domain"].value_counts().head(5).to_dict(),
    }
    if snapshot_incidents is not None and snapshot_reports is not None:
        linked = snapshot_incidents[snapshot_incidents["incident_id"].isin(counts.index)]
        link_len = linked.set_index("incident_id")["reports"].map(len)
        facts["report_count_matches_snapshot_links_pct"] = float(
            (link_len.reindex(counts.index) == counts).mean()
        )
        report_to_incidents = (
            linked[["incident_id", "reports"]]
            .explode("reports")
            .dropna()
            .astype({"reports": "int64"})
        )
        facts["reports_in_snapshot"] = len(snapshot_reports)
        linked_ids = set(report_to_incidents["reports"].unique())
        facts["reports_linked_to_some_incident"] = len(linked_ids)
        facts["reports_linked_to_multiple_incidents"] = int(
            (report_to_incidents.groupby("reports")["incident_id"].nunique() > 1).sum()
        )
        unlinked = ~reports_sheet["Report Number"].isin(linked_ids)
        facts["reports_not_linked_to_any_incident"] = int(unlinked.sum())
        facts["unlinked_reports_flagged_not_incident_report"] = int(
            (
                unlinked & reports_sheet["Is Incident Report"].astype(str).str.lower().eq("false")
            ).sum()
        )
        facts["unlinked_reports_that_are_stubs"] = int((unlinked & stub).sum())
        text = snapshot_reports.set_index("report_number")["text"].fillna("")
        words = text.str.split().str.len()
        stub_numbers = set(reports_sheet.loc[stub, "Report Number"])
        real_words = words[~words.index.isin(stub_numbers)]
        facts["real_reports_with_nonempty_text_pct"] = float((real_words > 0).mean())
        facts["report_text_words_median"] = float(real_words.median())
        facts["report_text_words_p05"] = float(real_words.quantile(0.05))
        facts["report_text_words_p95"] = float(real_words.quantile(0.95))
        facts["real_reports_under_50_words_pct"] = float((real_words < 50).mean())
        merged = report_to_incidents.assign(words=report_to_incidents["reports"].map(words))
        per_incident_text = merged.groupby("incident_id")["words"].agg(["sum", "max"])
        facts["incidents_with_any_full_text_pct"] = float(
            (per_incident_text["max"].reindex(counts.index).fillna(0) >= 50).mean()
        )
        facts["incident_total_text_words_median"] = float(
            per_incident_text["sum"].reindex(counts.index).fillna(0).median()
        )
    table = counts.rename("report_count").reset_index()
    return facts, table


# --------------------------------------------------------------------------------------
# Snapshot
# --------------------------------------------------------------------------------------


def snapshot_inventory(snapshot_dir: Path) -> pd.DataFrame:
    """Top-level snapshot files with size, and row/column counts for the CSVs."""
    rows = []
    for path in sorted(snapshot_dir.glob("*.csv")):
        df = pd.read_csv(path, low_memory=False)
        rows.append(
            {
                "file": path.name,
                "size_mb": path.stat().st_size / 1e6,
                "rows": len(df),
                "columns": df.shape[1],
                "fields": ", ".join(map(str, df.columns[:8])) + (" ..." if df.shape[1] > 8 else ""),
            }
        )
    for path in sorted((snapshot_dir / "aiidprod").glob("*.bson")):
        rows.append(
            {
                "file": f"aiidprod/{path.name}",
                "size_mb": path.stat().st_size / 1e6,
                "rows": None,
                "columns": None,
                "fields": "BSON collection",
            }
        )
    return pd.DataFrame(rows)


def similarity_audit(snapshot_incidents: pd.DataFrame, excel_ids: set[int]) -> dict[str, Any]:
    """Editor-marked similar/dissimilar links in the snapshot (candidate RQ4 ground truth)."""
    n = len(snapshot_incidents)
    links = {
        int(rec.incident_id): [int(x) for x in rec.editor_similar_incidents]
        for rec in snapshot_incidents.itertuples()
    }
    directed = [(a, b) for a, lst in links.items() for b in lst]
    undirected = {frozenset(p) for p in directed}
    reciprocal = sum(1 for a, b in directed if a in links.get(b, []))
    with_links = {a for a, lst in links.items() if lst}
    in_scope = [a for a in with_links if a in excel_ids]
    # Query incidents that also have at least one in-scope target.
    usable_queries = [a for a in in_scope if any(b in excel_ids and b != a for b in links[a])]
    # An incident is also "relevant" for its partner, so count queries under symmetrisation.
    sym: dict[int, set[int]] = {}
    for a, b in directed:
        sym.setdefault(a, set()).add(b)
        sym.setdefault(b, set()).add(a)
    sym_queries = [a for a, s in sym.items() if a in excel_ids and any(b in excel_ids for b in s)]
    relevant_per_query = pd.Series(
        {
            a: len({b for b in s if b in excel_ids and b != a})
            for a, s in sym.items()
            if a in excel_ids
        }
    )
    relevant_per_query = relevant_per_query[relevant_per_query > 0]
    return {
        "links_with_both_ends_in_excel_pct": sum(
            1 for x, y in directed if x in excel_ids and y in excel_ids
        )
        / len(directed),
        "relevant_per_query_mean": float(relevant_per_query.mean()),
        "relevant_per_query_median": float(relevant_per_query.median()),
        "relevant_per_query_max": int(relevant_per_query.max()),
        "queries_with_exactly_one_relevant_pct": float((relevant_per_query == 1).mean()),
        "max_possible_precision_at_5_mean": float((relevant_per_query.clip(upper=5) / 5).mean()),
        "snapshot_incidents": n,
        "has_editor_similar_field": "editor_similar_incidents" in snapshot_incidents.columns,
        "incidents_with_editor_similar": len(with_links),
        "incidents_with_editor_similar_pct": len(with_links) / n,
        "editor_similar_directed_links": len(directed),
        "editor_similar_undirected_pairs": len(undirected),
        "editor_similar_reciprocal_pct": reciprocal / len(directed) if directed else 0.0,
        "editor_similar_usable_queries_directed": len(usable_queries),
        "editor_similar_usable_queries_symmetrised": len(sym_queries),
        "incidents_with_editor_dissimilar": int(
            sum(1 for lst in snapshot_incidents["editor_dissimilar_incidents"] if len(lst))
        ),
        "incidents_with_nlp_similar": int(
            sum(1 for lst in snapshot_incidents["nlp_similar_incidents"] if len(lst))
        ),
    }


def excel_vs_snapshot_consistency(
    incidents: pd.DataFrame, snapshot_incidents: pd.DataFrame, mit: pd.DataFrame
) -> dict[str, Any]:
    """Do the Excel export and the snapshot agree on shared incidents?"""
    ex = incidents.set_index("Incident ID")
    sn = snapshot_incidents.set_index("incident_id")
    shared = ex.index.intersection(sn.index)
    facts: dict[str, Any] = {
        "ids_in_excel": len(ex),
        "ids_in_snapshot": len(sn),
        "ids_only_in_snapshot": sorted(int(i) for i in sn.index.difference(ex.index)),
        "ids_only_in_excel": sorted(int(i) for i in ex.index.difference(sn.index)),
        "title_equal_pct": float((ex.loc[shared, "title"] == sn.loc[shared, "title"]).mean()),
        "description_equal_pct": float(
            (ex.loc[shared, "description"] == sn.loc[shared, "description"]).mean()
        ),
        "date_equal_pct": float((ex.loc[shared, "date"] == sn.loc[shared, "date"]).mean()),
    }
    mit_domain = mit.set_index("Incident ID")["Risk Domain"].map(strip_domain_number)
    excel_domain = ex["Risk Domain"]
    both = excel_domain.index.intersection(mit_domain.index)
    facts["mit_domain_equal_pct"] = float(
        (excel_domain.loc[both].fillna("") == mit_domain.loc[both]).mean()
    )
    facts["mit_ids_in_snapshot"] = int(mit_domain.index.nunique())
    facts["mit_ids_not_in_excel"] = sorted(int(i) for i in mit_domain.index.difference(ex.index))
    return facts


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def run_audit() -> AuditResults:
    """Run every Phase 1 check against data/raw and collect tables and scalar facts."""
    cfg = load_config()
    snapshot_date = str(cfg["snapshot_date"])
    excel = load.find_excel_export()
    logger.info("Auditing %s", excel.name)

    inc = load.load_incidents(excel)
    rep = load.load_reports(excel)
    ent = load.load_entities(excel)
    dd = load.load_data_dictionary(excel)
    cm = load.load_coverage_map(excel)

    res = AuditResults(snapshot_date=snapshot_date)
    t, s = res.tables, res.scalars

    # Sheets and dictionary
    t["sheet_shapes"] = sheet_shapes(excel)
    t["profile_incidents"] = profile_table(inc, "Incidents")
    t["profile_reports"] = profile_table(rep, "Reports")
    t["profile_entities"] = profile_table(ent, "Entities")
    t["dictionary_crosscheck"] = cross_check_dictionary(inc, dd)
    t["coverage_map_check"] = coverage_map_check(inc, cm)
    s["n_incidents"] = len(inc)
    s["n_reports"] = len(rep)
    s["n_entities_sheet"] = len(ent)

    # Time
    per_year = incidents_per_year(inc, snapshot_date)
    t["incidents_per_year"] = per_year
    t["incidents_monthly_recent"] = monthly_counts(inc)
    s["date_min"] = str(inc["date"].min().date())
    s["date_max"] = str(inc["date"].max().date())
    s["snapshot_year_is_partial"] = bool(not per_year["complete"].iloc[-1])
    s["partial_year"] = int(per_year["year"].iloc[-1])
    s["partial_year_incidents"] = int(per_year["incidents"].iloc[-1])
    s["last_complete_year"] = int(per_year.loc[per_year["complete"], "year"].max())
    s["incidents_pre_2015"] = int(inc.loc[inc["year"] < 2015].shape[0])
    s["year_range_recommendation"] = recommend_year_range(per_year)

    # Text
    t["text_length_stats"] = text_length_stats(inc)

    # Labels
    groups = label_columns_by_group(dd)
    t["label_profile"] = label_profile(inc, groups)
    t["mit_domain_distribution"] = class_distribution(inc, "Risk Domain")
    t["mit_subdomain_distribution"] = class_distribution(inc, "Risk Subdomain")
    t["mit_coverage_by_year"] = label_coverage_by_year(inc, "Risk Domain")
    t["mit_domain_by_period"] = class_counts_by_period(inc, "Risk Domain")
    s["mit_domain_n"] = int(t["mit_domain_distribution"]["n"].sum())
    s["mit_domain_classes"] = len(t["mit_domain_distribution"])
    s["csetv1_incidents"] = int(inc["Data Sources"].fillna("").str.contains("CSETv1").sum())
    s["gmf_incidents"] = int(inc["Data Sources"].fillna("").str.contains("GMF").sum())
    s["no_taxonomy_incidents"] = int(inc["Data Sources"].isna().sum())

    # Duplicates (need the snapshot for the merge list)
    snap_dir = load.find_snapshot_dir()
    s["snapshot_present"] = snap_dir is not None
    snap_inc = merged = reps_full = None
    if snap_dir is not None:
        snap_inc = load.load_snapshot_incidents(snap_dir)
        merged = load.load_snapshot_duplicates(snap_dir)
        reps_full = load.load_full_report_text(snap_dir)
    dup_facts, near_pairs = duplicate_checks(inc, snap_inc, merged)
    s["duplicates"] = dup_facts
    t["near_duplicate_pairs_top25"] = near_pairs

    # Entities and snapshot
    if snap_dir is not None and snap_inc is not None and reps_full is not None:
        ent_facts, ent_summary, tops = entity_audit(inc, snap_inc, ent)
        s["entities"] = ent_facts
        t["entity_roles_summary"] = ent_summary
        for role, top in tops.items():
            t[f"entities_top20_{role}"] = top
        t["snapshot_inventory"] = snapshot_inventory(snap_dir)
        s["similarity"] = similarity_audit(snap_inc, set(inc["Incident ID"]))
        s["excel_vs_snapshot"] = excel_vs_snapshot_consistency(
            inc, snap_inc, load.load_snapshot_classification("MIT", snap_dir)
        )
        s["taxonomy_provenance"] = mit_provenance(load.load_taxa(snap_dir))
        s["mit_records"] = mit_record_facts(snap_dir)
        inventory = t["snapshot_inventory"].set_index("file")["rows"]
        s["snapshot_rows"] = {
            "quickadd": int(inventory["quickadd.csv"]),
            "submissions": int(inventory["submissions.csv"]),
        }
        rep_facts, rep_table = reports_audit(inc, rep, snap_inc, reps_full)
    else:
        rep_facts, rep_table = reports_audit(inc, rep, None, None)
    s["reports"] = rep_facts
    t["reports_per_incident"] = rep_table
    return res


def project_relative(path: Path) -> str:
    """Repo-relative POSIX path string, for registry `source` fields."""
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()
