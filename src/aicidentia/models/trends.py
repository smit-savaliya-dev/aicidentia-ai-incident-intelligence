"""RQ1 - temporal trends of risk themes (Mann-Kendall tests)."""

from __future__ import annotations

import pandas as pd


def topic_counts_by_year(doc_topics: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    """Count (or share) incidents per topic and year."""
    raise NotImplementedError


def mann_kendall_test(series: pd.Series) -> dict:
    """Run the Mann-Kendall trend test; return trend, p-value, tau and Sen's slope."""
    raise NotImplementedError


def trend_table(topic_year: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Apply the Mann-Kendall test to every topic series; correct for multiple testing."""
    raise NotImplementedError
