"""Near-duplicate incidents: candidate pairs for manual review and applying your decisions.

AIID's own merge list is applied automatically (`clean.resolve_merge_map`). Near-duplicates found
by text similarity are only *candidates*: AIID legitimately holds several incidents about similar
events, so nothing is merged unless you write a decision in data/manual/duplicate_review.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from aicidentia.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

REVIEW_PATH = PROJECT_ROOT / "data" / "manual" / "duplicate_review.csv"
DECISIONS = ("merge_into_a", "merge_into_b", "keep_separate")
_COLUMNS = [
    "incident_a",
    "incident_b",
    "cosine",
    "date_a",
    "date_b",
    "title_a",
    "title_b",
    "description_a",
    "description_b",
    "decision",
    "notes",
]


def find_similar_pairs(incidents: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """Incident pairs whose TF-IDF cosine similarity (title + description) is at least `threshold`."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    matrix = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(
        incidents["text"]
    )
    sim = (matrix @ matrix.T).tocoo()
    keep = (sim.row < sim.col) & (sim.data >= threshold)
    pairs = pd.DataFrame({"i": sim.row[keep], "j": sim.col[keep], "cosine": sim.data[keep]})
    rows = incidents.reset_index(drop=True)
    out = pd.DataFrame(
        {
            "incident_a": rows.loc[pairs["i"], "incident_id"].to_numpy(),
            "incident_b": rows.loc[pairs["j"], "incident_id"].to_numpy(),
            "cosine": pairs["cosine"].round(3).to_numpy(),
            "date_a": rows.loc[pairs["i"], "date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "date_b": rows.loc[pairs["j"], "date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "title_a": rows.loc[pairs["i"], "title"].to_numpy(),
            "title_b": rows.loc[pairs["j"], "title"].to_numpy(),
            "description_a": rows.loc[pairs["i"], "description"].to_numpy(),
            "description_b": rows.loc[pairs["j"], "description"].to_numpy(),
        }
    )
    out["decision"] = ""
    out["notes"] = ""
    return out.sort_values("cosine", ascending=False).reset_index(drop=True)[_COLUMNS]


def write_review_file(candidates: pd.DataFrame, path: Path = REVIEW_PATH) -> bool:
    """Write the review template; if a file exists, only append new pairs (your decisions are kept)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        seen = set(zip(existing["incident_a"], existing["incident_b"], strict=True))
        mask = [
            (str(a), str(b)) not in seen
            for a, b in zip(candidates["incident_a"], candidates["incident_b"], strict=True)
        ]
        new = candidates[mask]
        if new.empty:
            return False
        combined = pd.concat([existing, new.astype(str)], ignore_index=True)
    else:
        combined = candidates.astype(str)
    combined.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return True


def load_merge_decisions(path: Path = REVIEW_PATH) -> dict[int, int]:
    """Your decisions as {merged-away incident id: kept incident id}. Blank / keep_separate rows are ignored."""
    if not path.exists():
        return {}
    review = pd.read_csv(path, dtype=str, keep_default_na=False)
    merges: dict[int, int] = {}
    for row in review.itertuples():
        decision = row.decision.strip().lower()
        a, b = int(row.incident_a), int(row.incident_b)
        if decision == "merge_into_a":
            merges[b] = a
        elif decision == "merge_into_b":
            merges[a] = b
        elif decision not in ("", "keep_separate"):
            raise ValueError(
                f"Unknown decision {row.decision!r} for pair {a}/{b}; use one of {DECISIONS}"
            )
    return merges


def review_progress(path: Path = REVIEW_PATH) -> dict[str, int]:
    """Rows, answered rows and merges requested."""
    if not path.exists():
        return {"rows": 0, "answered": 0, "merges": 0}
    review = pd.read_csv(path, dtype=str, keep_default_na=False)
    decision = review["decision"].str.strip().str.lower()
    return {
        "rows": len(review),
        "answered": int((decision != "").sum()),
        "merges": int(decision.isin({"merge_into_a", "merge_into_b"}).sum()),
    }
