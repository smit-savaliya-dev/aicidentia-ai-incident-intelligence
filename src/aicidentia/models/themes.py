"""Join incidents to their themes using YOUR topic labels (data/manual/topic_labels.csv).

A theme is named by `final_label` when you have filled it in. Two topics given the *same*
`final_label` are merged into one theme. Topics without a final label fall back to an automatic
placeholder (`T<id>: <first terms>`) and the whole result is flagged provisional. A label file
whose `model_key` does not match the saved assignments is refused, so labels are never applied to a
different topic solution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from aicidentia.data import clean
from aicidentia.models.topic_outputs import LABELS_PATH

logger = logging.getLogger(__name__)


class StaleLabelsError(ValueError):
    """The label file belongs to a different topic solution than the saved assignments."""


@dataclass
class ThemeFrame:
    """Incident-level table with a `theme` column, plus facts about how the labels were applied."""

    data: pd.DataFrame
    topics: pd.DataFrame  # topic_id, theme, n_incidents, answered
    info: dict = field(default_factory=dict)


def theme_names(labels: pd.DataFrame) -> pd.DataFrame:
    """topic_id -> theme name (your final_label, else a marked placeholder), with answered flag."""
    out = labels[["topic_id", "suggested_label", "final_label"]].copy()
    out["topic_id"] = out["topic_id"].astype(int)
    answered = out["final_label"].str.strip() != ""
    out["answered"] = answered
    placeholder = "T" + out["topic_id"].astype(str) + ": " + out["suggested_label"]
    out["theme"] = out["final_label"].str.strip().where(answered, placeholder)
    return out[["topic_id", "theme", "answered"]]


def load_theme_frame(labels_path: Path = LABELS_PATH) -> ThemeFrame:
    """Incidents with year/quarter, report counts, MIT domain and their theme."""
    assignments = clean.load_processed("incident_topics")
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    keys_assign, keys_labels = set(assignments["model_key"]), set(labels["model_key"])
    if keys_assign != keys_labels:
        raise StaleLabelsError(
            f"{labels_path.name} is for {sorted(keys_labels)} but the saved assignments are {sorted(keys_assign)}; "
            "see topic_labels.proposed.csv or re-run scripts/run_phase3_topics.py."
        )
    names = theme_names(labels)
    incidents = clean.load_processed("incidents_clean")[
        [
            "incident_id",
            "date",
            "year",
            "quarter",
            "year_quarter",
            "report_count",
            "report_count_clean",
            "has_mit",
            "mit_risk_domain",
        ]
    ]
    data = (
        assignments[["incident_id", "topic_id"]]
        .merge(names, on="topic_id", how="left")
        .merge(incidents, on="incident_id", how="left")
    )
    topics = (
        data.groupby(["topic_id", "theme"])
        .size()
        .rename("n_incidents")
        .reset_index()
        .merge(names[["topic_id", "answered"]], on="topic_id")
    )
    merged = {t: sorted(g["topic_id"].tolist()) for t, g in topics.groupby("theme") if len(g) > 1}
    n_answered = int(names["answered"].sum())
    info = {
        "n_topics": len(names),
        "n_answered": n_answered,
        "n_themes": int(data["theme"].nunique()),
        "provisional": n_answered < len(names),
        "merged_themes": merged,
        "model_key": next(iter(keys_assign)),
    }
    if info["provisional"]:
        logger.warning(
            "Only %d of %d topic labels are confirmed: results are PROVISIONAL",
            n_answered,
            len(names),
        )
    return ThemeFrame(data, topics, info)
