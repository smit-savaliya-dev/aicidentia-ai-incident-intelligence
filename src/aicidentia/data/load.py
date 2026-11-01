"""Load the raw AIID files (Excel export and optional full snapshot) from data/raw/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def find_excel_export(raw_dir: Path | None = None) -> Path:
    """Return the path of the AIID Excel export in `raw_dir` (default: configured raw dir)."""
    raise NotImplementedError


def find_snapshot_archive(raw_dir: Path | None = None) -> Path | None:
    """Return the path of the optional backup .tar.bz2 snapshot, or None if absent."""
    raise NotImplementedError


def load_sheet(sheet_name: str, excel_path: Path | None = None) -> pd.DataFrame:
    """Read one sheet (Incidents, Reports, Entities, Data Dictionary, Coverage Map)."""
    raise NotImplementedError


def load_incidents(excel_path: Path | None = None) -> pd.DataFrame:
    """Read the Incidents sheet."""
    raise NotImplementedError


def load_reports(excel_path: Path | None = None) -> pd.DataFrame:
    """Read the Reports sheet."""
    raise NotImplementedError


def load_entities(excel_path: Path | None = None) -> pd.DataFrame:
    """Read the Entities sheet."""
    raise NotImplementedError


def load_full_report_text(snapshot_dir: Path | None = None) -> pd.DataFrame:
    """Read full article text from the extracted snapshot (CSV/JSON)."""
    raise NotImplementedError


def load_editor_similar_incidents(snapshot_dir: Path | None = None) -> pd.DataFrame:
    """Read editor-marked similar incidents from the snapshot (ground truth for RQ4)."""
    raise NotImplementedError


def load_hand_labels(path: Path | None = None) -> pd.DataFrame:
    """Read the 150 hand-labelled incidents from data/manual/."""
    raise NotImplementedError
