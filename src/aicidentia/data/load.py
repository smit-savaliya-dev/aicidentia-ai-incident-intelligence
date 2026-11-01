"""Load the raw AIID files (Excel export and snapshot) from data/raw/.

The Excel export has banner rows above the real header, so headers are detected by
looking for the expected first column name rather than by a fixed row number.
The snapshot is a MongoDB dump: BSON files give native lists/ints, CSV files give
the classification tables and the full report text. Nothing here modifies data/raw.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import bson
import pandas as pd

from aicidentia.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

# First column of each Excel sheet, used to locate the header row under the banner rows.
_SHEET_FIRST_COLUMN = {
    "Incidents": "Incident ID",
    "Reports": "Report Number",
    "Entities": "Entity ID",
    "Data Dictionary": "Column",
    "Coverage Map": "Data Sources",
}

# Large embedding/projection arrays in the BSON dump that the analysis does not need.
_BSON_DROP_FIELDS = ("embedding", "tsne")


def raw_dir(cfg: dict[str, Any] | None = None) -> Path:
    """Absolute path of data/raw/."""
    cfg = cfg or load_config()
    return PROJECT_ROOT / cfg["paths"]["raw"]


def find_excel_export(raw: Path | None = None) -> Path:
    """Return the AIID Excel export in `raw` (latest match of the configured glob)."""
    cfg = load_config()
    raw = raw or raw_dir(cfg)
    matches = sorted(raw.glob(cfg["files"]["excel_export_glob"]))
    if not matches:
        raise FileNotFoundError(f"No file matching {cfg['files']['excel_export_glob']!r} in {raw}")
    return matches[-1]


def find_snapshot_dir(raw: Path | None = None) -> Path | None:
    """Return the extracted snapshot directory (contains aiidprod/), or None if absent.

    Looks for any directory in `raw` that contains an `aiidprod` folder, which covers both
    `snapshot_<date>/<backup>/` (extracted by scripts/check_data.py) and a manually extracted
    `mongodump_full_snapshot/`.
    """
    raw = raw or raw_dir()
    candidates = sorted(p.parent for p in raw.glob("*/aiidprod") if p.is_dir())
    candidates += sorted(p.parent.parent for p in raw.glob("*/*/aiidprod") if p.is_dir())
    return candidates[-1] if candidates else None


def find_snapshot_archive(raw: Path | None = None) -> Path | None:
    """Return the optional backup .tar.bz2 snapshot, or None if absent."""
    cfg = load_config()
    raw = raw or raw_dir(cfg)
    matches = sorted(raw.glob(cfg["files"]["snapshot_glob"]))
    return matches[-1] if matches else None


def _require_snapshot() -> Path:
    snap = find_snapshot_dir()
    if snap is None:
        raise FileNotFoundError("Snapshot directory not found in data/raw/ (it is optional).")
    return snap


def load_sheet(sheet_name: str, excel_path: Path | None = None) -> pd.DataFrame:
    """Read one Excel sheet with banner rows removed and the real header applied.

    Entities has an empty spacer row under the header; fully empty rows are dropped.
    """
    excel_path = excel_path or find_excel_export()
    raw = _read_excel_cached(str(excel_path), sheet_name)
    first = _SHEET_FIRST_COLUMN[sheet_name]
    header_rows = raw.index[raw.iloc[:, 0].astype("string").str.strip() == first]
    if len(header_rows) == 0:
        raise ValueError(f"Header row starting with {first!r} not found in sheet {sheet_name!r}")
    header_row = header_rows[0]
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = [str(c).strip() for c in raw.iloc[header_row]]
    df = df.dropna(how="all").reset_index(drop=True)
    return df


@lru_cache(maxsize=8)
def _read_excel_cached(path: str, sheet_name: str) -> pd.DataFrame:
    # keep_default_na=False: pandas would otherwise turn the literal strings "None"/"NA"/"null"
    # (e.g. Data Sources = "None", entity names) into missing values. Only empty cells are NaN.
    return pd.read_excel(
        path, sheet_name=sheet_name, header=None, keep_default_na=False, na_values=[""]
    )


def sheet_banner(sheet_name: str, excel_path: Path | None = None) -> str:
    """Text of the title banner in the first cell of a sheet (states the advertised row count)."""
    excel_path = excel_path or find_excel_export()
    return str(_read_excel_cached(str(excel_path), sheet_name).iloc[0, 0])


def load_incidents(excel_path: Path | None = None) -> pd.DataFrame:
    """Read the Incidents sheet with typed id, date, year and report_count columns."""
    df = load_sheet("Incidents", excel_path)
    df["Incident ID"] = pd.to_numeric(df["Incident ID"]).astype("int64")
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = pd.to_numeric(df["year"]).astype("int64")
    df["report_count"] = pd.to_numeric(df["report_count"]).astype("int64")
    # The export writes the literal "None" in Data Sources for incidents with no taxonomy.
    df["Data Sources"] = df["Data Sources"].replace("None", pd.NA)
    return df


def load_reports(excel_path: Path | None = None) -> pd.DataFrame:
    """Read the Reports sheet (no incident id column; linkage lives in the snapshot)."""
    df = load_sheet("Reports", excel_path)
    df["Report Number"] = pd.to_numeric(df["Report Number"]).astype("int64")
    for col in ("Date Published", "Date Downloaded"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_entities(excel_path: Path | None = None) -> pd.DataFrame:
    """Read the Entities sheet (slug id -> display name)."""
    return load_sheet("Entities", excel_path)


def load_data_dictionary(excel_path: Path | None = None) -> pd.DataFrame:
    """Read the Data Dictionary sheet."""
    return load_sheet("Data Dictionary", excel_path)


def load_coverage_map(excel_path: Path | None = None) -> pd.DataFrame:
    """Read the Coverage Map sheet (a blank 'Data Sources' label means no taxonomy applied)."""
    return load_sheet("Coverage Map", excel_path)


def read_bson(collection: str, snapshot_dir: Path | None = None) -> list[dict[str, Any]]:
    """Decode one collection of the MongoDB dump (aiidprod/<collection>.bson)."""
    snapshot_dir = snapshot_dir or _require_snapshot()
    path = snapshot_dir / "aiidprod" / f"{collection}.bson"
    with path.open("rb") as fh:
        docs = list(bson.decode_file_iter(fh))
    for doc in docs:
        for field in _BSON_DROP_FIELDS:
            doc.pop(field, None)
    return docs


def load_snapshot_incidents(snapshot_dir: Path | None = None) -> pd.DataFrame:
    """Snapshot incidents with native list columns (reports, entity slugs, similarity links)."""
    df = pd.DataFrame(read_bson("incidents", snapshot_dir))
    df = df.rename(
        columns={
            "Alleged deployer of AI system": "deployer",
            "Alleged developer of AI system": "developer",
            "Alleged harmed or nearly harmed parties": "harmed",
        }
    )
    df["incident_id"] = df["incident_id"].astype("int64")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_snapshot_entities(snapshot_dir: Path | None = None) -> pd.DataFrame:
    """Snapshot entities (entity_id slug, name)."""
    return pd.DataFrame(read_bson("entities", snapshot_dir))[["entity_id", "name"]]


def load_snapshot_duplicates(snapshot_dir: Path | None = None) -> pd.DataFrame:
    """Incident numbers that were merged into another incident (duplicates.csv)."""
    snapshot_dir = snapshot_dir or _require_snapshot()
    return pd.read_csv(snapshot_dir / "duplicates.csv")


def load_snapshot_classification(namespace: str, snapshot_dir: Path | None = None) -> pd.DataFrame:
    """One classification table from the snapshot: 'MIT', 'GMF', 'CSETv1', 'CSETv0'."""
    snapshot_dir = snapshot_dir or _require_snapshot()
    return pd.read_csv(snapshot_dir / f"classifications_{namespace}.csv", low_memory=False)


def load_taxa(snapshot_dir: Path | None = None) -> list[dict[str, Any]]:
    """Taxonomy definitions (field lists, permitted values, provenance flags)."""
    return read_bson("taxa", snapshot_dir)


def load_full_report_text(snapshot_dir: Path | None = None) -> pd.DataFrame:
    """Reports with full article text from the snapshot (reports.csv)."""
    snapshot_dir = snapshot_dir or _require_snapshot()
    df = pd.read_csv(snapshot_dir / "reports.csv", low_memory=False)
    df["report_number"] = df["report_number"].astype("int64")
    return df


def load_editor_similar_incidents(snapshot_dir: Path | None = None) -> pd.DataFrame:
    """Editor-marked similar incidents as (incident_id, similar_id) rows (RQ4 ground truth)."""
    inc = load_snapshot_incidents(snapshot_dir)
    rows = [
        (int(row.incident_id), int(other))
        for row in inc.itertuples()
        for other in row.editor_similar_incidents
    ]
    return pd.DataFrame(rows, columns=["incident_id", "similar_id"])


def load_hand_labels(path: Path | None = None) -> pd.DataFrame:
    """Read the 150 hand-labelled incidents from data/manual/."""
    cfg = load_config()
    path = path or PROJECT_ROOT / cfg["paths"]["manual"] / cfg["files"]["hand_labels"]
    return pd.read_csv(path)
