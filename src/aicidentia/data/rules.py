"""Rule-based answers for the review files in data/manual/ (see docs/decisions.md, D2 and D3).

The review files were designed for a human; these rules answer them reproducibly instead:

* Duplicate review (D2): a candidate pair is merged only when ALL hold: the same deployer (at least one shared
  deployer entity), incident dates at most 7 days apart, and text similarity >= 0.90 (the TF-IDF cosine stored
  in the file). Every other pair is `keep_separate`. AIID's own merge list is applied separately and always.
* Entity review (D3): every person-like candidate is a named person (private or public) and is excluded, unless
  its entity id has a non-person type override in `config/entity_types.yaml` (organisations, products/systems,
  groups identified by reading the candidate names). Only organisations, systems and groups remain.

Existing files are rewritten in place with the rule's answer and a note naming the rule.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from aicidentia.data import duplicates, entities

logger = logging.getLogger(__name__)

MAX_DAYS = 7
MIN_COSINE = 0.90
PERSON_TYPES = {"individual_candidate"}


def duplicate_decision(
    shared_deployers: set[str], days_apart: float, cosine: float
) -> tuple[str, str]:
    """(decision, note) for one candidate pair under rule D2."""
    checks = {
        "same deployer": bool(shared_deployers),
        f"dates within {MAX_DAYS} days": days_apart <= MAX_DAYS,
        f"similarity >= {MIN_COSINE:.2f}": cosine >= MIN_COSINE,
    }
    failed = [name for name, ok in checks.items() if not ok]
    # Deployer ids are counted, not listed: a deployer can be a named person (rule D3).
    detail = f"cosine {cosine:.3f}, {days_apart:.0f} days apart, {len(shared_deployers)} shared deployer(s)"
    if failed:
        return "keep_separate", f"rule D2: fails {'; '.join(failed)} ({detail})"
    return "merge_into_a", f"rule D2: all criteria met ({detail})"


def apply_duplicate_rule(
    review: pd.DataFrame, deployers: dict[int, set[str]]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fill `decision` and `notes` for every pair. `deployers` maps incident id -> deployer entity ids.

    `merge_into_a` keeps the earlier-numbered incident (incident_a), which AIID treats as the original."""
    out = review.copy()
    decisions, notes = [], []
    for row in out.itertuples():
        a, b = int(row.incident_a), int(row.incident_b)
        days = abs((pd.Timestamp(row.date_a) - pd.Timestamp(row.date_b)).days)
        shared = deployers.get(a, set()) & deployers.get(b, set())
        decision, note = duplicate_decision(shared, days, float(row.cosine))
        if decision == "merge_into_a" and a > b:
            decision = "merge_into_b"
        decisions.append(decision)
        notes.append(note)
    out["decision"], out["notes"] = decisions, notes
    counts = {
        "pairs": len(out),
        "merged": int(out["decision"].str.startswith("merge").sum()),
        "keep_separate": int((out["decision"] == "keep_separate").sum()),
        "fail_deployer": int(out["notes"].str.contains("fails[^(]*same deployer").sum()),
        "fail_dates": int(out["notes"].str.contains(f"fails[^(]*dates within {MAX_DAYS}").sum()),
        "fail_similarity": int(out["notes"].str.contains("fails[^(]*similarity").sum()),
    }
    return out, counts


def apply_entity_rule(
    review: pd.DataFrame, overrides: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fill `is_named_person` and `notes` under rule D3."""
    out = review.copy()
    override = out["entity_id"].map(overrides)
    not_person = override.notna() & ~override.isin(PERSON_TYPES)
    out["is_named_person"] = ["no" if x else "yes" for x in not_person]
    out["notes"] = [
        f"rule D3: not a person ({override[i]}); kept" if x else "rule D3: named person; excluded"
        for i, x in zip(out.index, not_person, strict=True)
    ]
    counts = {
        "candidates": len(out),
        "named_persons_excluded": int((~not_person).sum()),
        "not_persons_kept": int(not_person.sum()),
        **{
            f"kept_as_{t}": int((override[not_person] == t).sum())
            for t in sorted(set(override[not_person]))
        },
    }
    return out, counts


def deployers_by_incident(incident_entities: pd.DataFrame) -> dict[int, set[str]]:
    """incident id -> deployer entity ids (from the long entity table, before privacy exclusions)."""
    dep = incident_entities[incident_entities["role"] == "deployer"]
    return dep.groupby("incident_id")["entity_id"].agg(set).to_dict()


def apply_all(
    duplicate_path: Path = duplicates.REVIEW_PATH,
    entity_path: Path = entities.REVIEW_PATH,
) -> dict[str, dict[str, int]]:
    """Rewrite both review files with the rule answers; returns the counts for the registry."""
    from aicidentia.data import load

    facts: dict[str, dict[str, int]] = {}
    if duplicate_path.exists():
        alias_map = entities.active_alias_map()
        snap = load.load_snapshot_incidents(load.find_snapshot_dir())
        long = entities.build_long_table(snap, set(snap["incident_id"].astype(int)))
        long = long.assign(entity_id=long["entity_id_raw"].map(lambda s: alias_map.get(s, s)))
        review = pd.read_csv(duplicate_path, dtype=str, keep_default_na=False)
        answered, facts["duplicates"] = apply_duplicate_rule(review, deployers_by_incident(long))
        answered.to_csv(duplicate_path, index=False, encoding="utf-8", lineterminator="\n")
    if entity_path.exists():
        overrides = entities.load_type_rules().get("overrides") or {}
        review = pd.read_csv(entity_path, dtype=str, keep_default_na=False)
        answered, facts["entities"] = apply_entity_rule(review, overrides)
        answered.to_csv(entity_path, index=False, encoding="utf-8", lineterminator="\n")
    logger.info("Rule answers written: %s", facts)
    return facts
