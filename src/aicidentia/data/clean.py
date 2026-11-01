"""Cleaning and preparation of AIID tables (CRISP-DM: Data Preparation)."""

from __future__ import annotations

import pandas as pd


def clean_incidents(incidents: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes, dates and text fields of the Incidents table."""
    raise NotImplementedError


def clean_reports(reports: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes, dates and text fields of the Reports table."""
    raise NotImplementedError


def clean_entities(entities: pd.DataFrame) -> pd.DataFrame:
    """Normalise entity names (deduplicate aliases, strip whitespace/case)."""
    raise NotImplementedError


def assemble_incident_text(incidents: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """Build one modelling text per incident from title, description and report text."""
    raise NotImplementedError


def filter_year_range(df: pd.DataFrame, date_col: str, start: int, end: int) -> pd.DataFrame:
    """Keep rows whose `date_col` year lies in [start, end]."""
    raise NotImplementedError


def save_processed(df: pd.DataFrame, name: str) -> None:
    """Write a dataframe to data/processed/<name>.parquet."""
    raise NotImplementedError
