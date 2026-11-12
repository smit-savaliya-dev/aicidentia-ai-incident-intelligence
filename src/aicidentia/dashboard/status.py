"""Status of the files you fill in by hand, used by the refresh command and the Method page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from aicidentia.config import PROJECT_ROOT
from aicidentia.data import duplicates, entities
from aicidentia.models import label_audit
from aicidentia.models.topic_outputs import LABELS_PATH

MANUAL = PROJECT_ROOT / "data" / "manual"
COLAB_PREDICTIONS = PROJECT_ROOT / "reports" / "colab" / "rq3_distilbert_predictions.csv"


def _state(done: int, total: int) -> str:
    if total == 0:
        return "missing"
    return "complete" if done >= total else ("not started" if done == 0 else "partial")


def topic_label_status(path: Path = LABELS_PATH, topics_path: Path | None = None) -> dict[str, Any]:
    """Confirmed topic labels, and whether the file matches the saved topic solution."""
    if not path.exists():
        return {"done": 0, "total": 0, "stale": False}
    labels = pd.read_csv(path, dtype=str, keep_default_na=False)
    stale = False
    try:
        from aicidentia.data import clean

        assigned = (
            set(clean.load_processed("incident_topics")["model_key"])
            if topics_path is None
            else set(pd.read_parquet(topics_path)["model_key"])
        )
        stale = assigned != set(labels["model_key"])
    except FileNotFoundError:
        pass
    return {
        "done": int((labels["final_label"].str.strip() != "").sum()),
        "total": len(labels),
        "stale": stale,
    }


def colab_status(path: Path = COLAB_PREDICTIONS) -> dict[str, Any]:
    """Whether the DistilBERT predictions from Colab are present (validity is checked in Phase 6)."""
    return {"done": int(path.exists()), "total": 1}


def manual_status() -> pd.DataFrame:
    """One row per manual input: what it is, progress, state, which phases use it, what to do next."""
    rows = []
    topic = topic_label_status()
    rows.append(
        {
            "input": "Topic labels",
            "file": "data/manual/topic_labels.csv",
            "done": topic["done"],
            "total": topic["total"],
            "state": "stale (re-label from topic_labels.proposed.csv)"
            if topic["stale"]
            else _state(topic["done"], topic["total"]),
            "used_by": "Phase 4 trends, dashboard theme names",
            "how": "fill final_label (same label on two topics merges them)",
        }
    )
    ent = entities.review_progress() if entities.REVIEW_PATH.exists() else {}
    rows.append(
        {
            "input": "Entity review",
            "file": "data/manual/entity_review.csv",
            "done": int(ent.get("answered", 0)),
            "total": int(ent.get("rows", 0)),
            "state": _state(int(ent.get("answered", 0)), int(ent.get("rows", 0))),
            "used_by": "Phase 2 privacy filter -> Phases 5, 7, dashboard",
            "how": "is_private_individual = yes / no",
        }
    )
    dup = duplicates.review_progress() if duplicates.REVIEW_PATH.exists() else {}
    rows.append(
        {
            "input": "Duplicate review",
            "file": "data/manual/duplicate_review.csv",
            "done": int(dup.get("answered", 0)),
            "total": int(dup.get("rows", 0)),
            "state": _state(int(dup.get("answered", 0)), int(dup.get("rows", 0))),
            "used_by": "Phase 2 merges -> every later phase",
            "how": "decision = merge_into_a / merge_into_b / keep_separate",
        }
    )
    audit = label_audit.annotation_progress()
    rows.append(
        {
            "input": "Blind label audit",
            "file": "data/manual/label_audit_sample.csv",
            "done": audit["labelled"],
            "total": audit["rows"],
            "state": _state(audit["labelled"], audit["rows"])
            + (f"; {audit['unrecognised']} unrecognised labels" if audit["unrecognised"] else ""),
            "used_by": "Phase 6 audit (kappa); unhides MIT labels in the dashboard when complete",
            "how": "my_label (1-7 or name), my_confidence (1-3)",
        }
    )
    colab = colab_status()
    rows.append(
        {
            "input": "DistilBERT predictions (Colab)",
            "file": "reports/colab/rq3_distilbert_predictions.csv",
            "done": colab["done"],
            "total": 1,
            "state": "present" if colab["done"] else "missing",
            "used_by": "Phase 6 model comparison",
            "how": "run notebooks/colab_distilbert_finetune.ipynb on Colab, copy the CSV here",
        }
    )
    return pd.DataFrame(rows)
