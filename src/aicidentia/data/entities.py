"""Entity layer: long table, alias normalisation, rule-based entity types, privacy review.

AIID stores deployers, developers, harmed parties and implicated systems as entity slugs.
This module turns them into one tidy long table, merges aliases (AIID's own duplicate list,
name-normalisation clashes and manual entries in config/entity_aliases.yaml), assigns a
heuristic `entity_type` (the data has none) and prepares the manual review of candidate
named private individuals. Nothing here decides who is a private individual: that is a
human judgement recorded in data/manual/entity_review.csv.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from aicidentia.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

ROLES = ("deployer", "developer", "harmed", "system")
ENTITY_TYPES = (
    "organisation",
    "generic_actor_group",
    "technology_system",
    "population_group",
    "abstract_value",
    "individual_candidate",
    "unknown",
)
ALIASES_PATH = PROJECT_ROOT / "config" / "entity_aliases.yaml"
TYPE_RULES_PATH = PROJECT_ROOT / "config" / "entity_types.yaml"
REVIEW_PATH = PROJECT_ROOT / "data" / "manual" / "entity_review.csv"
_SNAPSHOT_ROLE_COLUMN = {
    "deployer": "deployer",
    "developer": "developer",
    "harmed": "harmed",
    "system": "implicated_systems",
}
_CORPORATE_SUFFIXES = r"\b(inc|llc|ltd|corp|corporation|co|company|plc|gmbh|the)\b\.?"


# --------------------------------------------------------------------------------------
# Long table
# --------------------------------------------------------------------------------------


def build_long_table(
    snapshot_incidents: pd.DataFrame, incident_ids: set[int] | None = None
) -> pd.DataFrame:
    """One row per (incident_id, role, entity_id_raw) from the snapshot slug lists.

    Rows are de-duplicated; if `incident_ids` is given, only those incidents are kept.
    """
    rows = []
    for rec in snapshot_incidents.itertuples():
        if incident_ids is not None and rec.incident_id not in incident_ids:
            continue
        for role, column in _SNAPSHOT_ROLE_COLUMN.items():
            rows.extend((int(rec.incident_id), role, slug) for slug in getattr(rec, column))
    long = pd.DataFrame(rows, columns=["incident_id", "role", "entity_id_raw"])
    return long.drop_duplicates().reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Aliases
# --------------------------------------------------------------------------------------


def normalise_name_key(name: str) -> str:
    """Lower-case ASCII key without punctuation or corporate suffixes ('Carribex LTD.' -> 'carribex')."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    ascii_name = re.sub(_CORPORATE_SUFFIXES, "", ascii_name)
    return re.sub(r"[^a-z0-9]+", "", ascii_name)


def resolve_alias_map(mapping: dict[str, str]) -> dict[str, str]:
    """Follow alias chains to their end (a -> b -> c gives a -> c); cycles stop at the repeat."""
    resolved = {}
    for alias in mapping:
        seen, current = {alias}, mapping[alias]
        while current in mapping and mapping[current] not in seen and mapping[current] != current:
            seen.add(current)
            current = mapping[current]
        if current != alias:
            resolved[alias] = current
    return resolved


def build_alias_candidates(
    entities: pd.DataFrame,
    long: pd.DataFrame,
    aiid_duplicates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Starter alias list built from the data: AIID's duplicate pairs plus name-normalisation clashes.

    Only aliases that matter (the alias is used by an incident) are listed. AIID pairs are skipped
    (and returned separately with a reason) when the canonical entity does not exist in the Entities
    sheet or when the pair is part of a cycle. Returns (candidates, skipped); candidate columns are
    alias, canonical, alias_name, canonical_name, source, incidents_alias, incidents_canonical.
    """
    names = dict(zip(entities["Entity ID"], entities["Name"], strict=False))
    known = set(names)
    incidents_per_entity = long.groupby("entity_id_raw")["incident_id"].nunique()
    used = set(incidents_per_entity.index)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    # 1. AIID's official duplicate list (duplicate -> true entity).
    aiid = dict(
        zip(aiid_duplicates["duplicate_entity_id"], aiid_duplicates["true_entity_id"], strict=False)
    )
    valid: dict[str, str] = {}
    for alias, canonical in aiid.items():
        if aiid.get(canonical) == alias:
            reason = "cycle"
        elif canonical not in known:
            reason = "canonical_not_in_entities_sheet"
        else:
            valid[alias] = canonical
            continue
        if alias in used:
            skipped.append(
                {
                    "alias": alias,
                    "canonical": canonical,
                    "source": "aiid_entity_duplicates",
                    "reason": reason,
                }
            )
    resolved_aiid = resolve_alias_map(valid)
    for alias, canonical in resolved_aiid.items():
        if alias in used:
            rows.append(
                {"alias": alias, "canonical": canonical, "source": "aiid_entity_duplicates"}
            )

    # 2. Entities whose normalised names are identical (case, accents, punctuation, Inc/LLC...).
    frame = entities.assign(key=entities["Name"].fillna("").map(normalise_name_key))
    frame = frame[(frame["key"] != "") & frame["Entity ID"].isin(used)]
    frame["n"] = frame["Entity ID"].map(incidents_per_entity)
    for _, group in frame.groupby("key"):
        if len(group) < 2:
            continue
        group = group.sort_values(["n", "Entity ID"], ascending=[False, True])
        canonical = group.iloc[0]["Entity ID"]
        canonical = resolved_aiid.get(canonical, canonical)
        for alias in group["Entity ID"].iloc[1:]:
            if resolved_aiid.get(alias, alias) != canonical:
                rows.append(
                    {"alias": alias, "canonical": canonical, "source": "name_normalisation"}
                )

    out = pd.DataFrame(rows, columns=["alias", "canonical", "source"]).drop_duplicates("alias")
    out["alias_name"] = out["alias"].map(names)
    out["canonical_name"] = out["canonical"].map(names)
    out["incidents_alias"] = out["alias"].map(incidents_per_entity).fillna(0).astype(int)
    out["incidents_canonical"] = out["canonical"].map(incidents_per_entity).fillna(0).astype(int)
    out = out.sort_values(["source", "incidents_alias"], ascending=[True, False]).reset_index(
        drop=True
    )
    return out, pd.DataFrame(skipped, columns=["alias", "canonical", "source", "reason"])


def read_alias_file(path: Path = ALIASES_PATH) -> dict[str, Any]:
    """Read config/entity_aliases.yaml (sections: manual, blocked, generated)."""
    if not path.exists():
        return {"manual": {}, "blocked": [], "generated": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        "manual": data.get("manual") or {},
        "blocked": data.get("blocked") or [],
        "generated": data.get("generated") or [],
    }


def write_alias_file(generated: pd.DataFrame, path: Path = ALIASES_PATH) -> None:
    """Rewrite the `generated` section; `manual` and `blocked` (your edits) are preserved."""
    current = read_alias_file(path)
    if not current["manual"] and not path.exists():
        current["manual"] = {"facebook": "meta"}  # the example from the project brief
    header = (
        "# Entity aliases (alias slug -> canonical slug).\n"
        "# manual:    YOUR edits; applied first and never overwritten when this file is regenerated.\n"
        "# blocked:   alias slugs from `generated` that you do NOT want applied.\n"
        "# generated: starter list built from the data by scripts/run_phase2_prepare.py (sources:\n"
        "#            aiid_entity_duplicates = AIID's own duplicate list; name_normalisation = same name\n"
        "#            apart from case/accents/punctuation/'Inc'). Regenerated on every run.\n"
        "# Aliases are chained (a -> b -> c resolves to c). Parent/subsidiary links (e.g. Instagram -> Meta)\n"
        "# are deliberately NOT aliases.\n"
    )
    records = generated[
        [
            "alias",
            "canonical",
            "source",
            "alias_name",
            "canonical_name",
            "incidents_alias",
            "incidents_canonical",
        ]
    ].to_dict(orient="records")
    body = yaml.safe_dump(
        {"manual": current["manual"], "blocked": current["blocked"], "generated": records},
        sort_keys=False,
        allow_unicode=True,
        width=120,
    )
    path.write_text(header + body, encoding="utf-8", newline="\n")
    logger.info("Wrote %s (%d generated aliases)", path.name, len(records))


def active_alias_map(path: Path = ALIASES_PATH) -> dict[str, str]:
    """Aliases to apply: generated (minus blocked), then manual entries on top; chains resolved."""
    data = read_alias_file(path)
    blocked = set(data["blocked"])
    mapping = {r["alias"]: r["canonical"] for r in data["generated"] if r["alias"] not in blocked}
    mapping.update({str(k): str(v) for k, v in data["manual"].items()})
    return resolve_alias_map(mapping)


def apply_aliases(
    long: pd.DataFrame, alias_map: dict[str, str], names: dict[str, str]
) -> pd.DataFrame:
    """Add canonical `entity_id` and display `entity`; drop rows made identical by merging."""
    out = long.copy()
    out["entity_id"] = out["entity_id_raw"].map(lambda s: alias_map.get(s, s))
    out["entity"] = out["entity_id"].map(lambda s: names.get(s) or s)
    return out.drop_duplicates(["incident_id", "role", "entity_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Entity types (heuristic)
# --------------------------------------------------------------------------------------


def load_type_rules(path: Path = TYPE_RULES_PATH) -> dict[str, Any]:
    """Read config/entity_types.yaml."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _term_regex(terms: list[str]) -> re.Pattern[str]:
    return re.compile(
        r"\b(" + "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True)) + r")\b",
        re.I,
    )


_PERSON_TOKEN = re.compile(r"^[A-Z][a-z]+(?:[-'][A-Za-z]+)*\.?$|^[A-Z]\.$")


def person_like(name: str, rules: dict[str, Any]) -> tuple[bool, str]:
    """Does the name look like a person's name? Returns (is_candidate, confidence).

    A candidate has 2-4 capitalised tokens, none a known organisation word, and the last token is
    not a plural group noun. Confidence is 'low' when the last token ends in 's' (could be a plural
    group, but is also a common surname). This only generates candidates for human review.
    """
    tokens = name.replace("’", "'").split()
    if not 2 <= len(tokens) <= 4 or not all(_PERSON_TOKEN.match(t) for t in tokens):
        return False, ""
    lowered = [t.lower().strip(".") for t in tokens]
    if any(t in set(rules["organisation_words"]) for t in lowered):
        return False, ""
    if lowered[-1] in set(rules["group_nouns"]):
        return False, ""
    return True, ("low" if lowered[-1].endswith("s") else "high")


def classify_entity_types(long: pd.DataFrame, rules: dict[str, Any] | None = None) -> pd.DataFrame:
    """Assign `entity_type` per canonical entity from its name and the roles it plays.

    Rule order (first match wins), recorded in `entity_type_rule`:
    override > abstract value (harmed side) > generic actor (actor side) > implicated system >
    population term > technology word > individual candidate (person-like name) >
    harmed side > actor side > plurality of roles.
    """
    rules = rules or load_type_rules()
    abstract = _term_regex(rules["abstract_value_terms"])
    tech = _term_regex(rules["technology_terms"])
    generic = _term_regex(rules["generic_actor_terms"])
    population = _term_regex(rules["population_terms"])
    overrides = rules.get("overrides") or {}
    counts = long.groupby(["entity_id", "role"])["incident_id"].nunique().unstack(fill_value=0)
    counts = counts.reindex(columns=list(ROLES), fill_value=0)
    names = long.drop_duplicates("entity_id").set_index("entity_id")["entity"]
    rows = []
    for entity_id, role_counts in counts.iterrows():
        total = role_counts.sum()
        name = str(names[entity_id])
        harmed = role_counts["harmed"] / total
        system = role_counts["system"] / total
        actor = (role_counts["deployer"] + role_counts["developer"]) / total
        is_person, _ = person_like(name, rules)
        is_person = is_person and system < 0.3
        if entity_id in overrides:
            etype, rule = overrides[entity_id], "override"
        elif harmed >= 0.8 and abstract.search(name):
            etype, rule = "abstract_value", "harmed_side+abstract_term"
        elif actor >= 0.5 and (generic.search(name) or entity_id.startswith("unknown-")):
            etype, rule = "generic_actor_group", "actor_side+generic_term"
        elif system >= 0.5:
            etype, rule = "technology_system", "implicated_system_role"
        elif population.search(name):
            etype, rule = "population_group", "population_term"
        elif tech.search(name) and harmed < 0.5:
            etype, rule = "technology_system", "technology_term"
        elif is_person:
            etype, rule = "individual_candidate", "person_like_name"
        elif harmed >= 0.5:
            etype, rule = "population_group", "harmed_side"
        elif actor >= 0.5:
            etype, rule = "organisation", "actor_side"
        elif max(actor, harmed, system) == actor:
            etype, rule = "organisation", "actor_side_plurality"
        elif harmed >= system:
            etype, rule = "population_group", "harmed_side_plurality"
        else:
            etype, rule = "technology_system", "system_side_plurality"
        rows.append({"entity_id": entity_id, "entity_type": etype, "entity_type_rule": rule})
    return pd.DataFrame(rows)


def build_entity_table(long: pd.DataFrame, types: pd.DataFrame) -> pd.DataFrame:
    """One row per canonical entity: name, type, incident count, role mix."""
    base = long.groupby("entity_id").agg(
        entity=("entity", "first"),
        n_incidents=("incident_id", "nunique"),
        n_mentions=("incident_id", "size"),
        roles=("role", lambda r: ",".join(sorted(set(r)))),
    )
    return (
        base.reset_index()
        .merge(types, on="entity_id", how="left")
        .sort_values("n_incidents", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------------------
# Privacy review of candidate named private individuals
# --------------------------------------------------------------------------------------

_REVIEW_COLUMNS = [
    "entity_id",
    "entity",
    "n_incidents",
    "roles",
    "candidate_confidence",
    "review_priority",
    "entity_type",
    "example_incident_ids",
    "example_titles",
    "is_named_person",
    "notes",
]


def build_review_candidates(
    entity_table: pd.DataFrame,
    long: pd.DataFrame,
    incidents: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Candidate named individuals for manual review, with blank decision columns.

    Candidates are entities with a person-like name. `is_named_person` and `notes` are left EMPTY here;
    `aicidentia.data.rules` answers them (rule D3: every named person, private or public, is excluded).
    `review_priority` is only a guide: A = named in 1-2 incidents (the pattern of private people), B = named in
    3+ incidents (more often public figures), C = surname ends in 's' (may be a plural group).
    """
    rules = rules or load_type_rules()
    titles = incidents.set_index("incident_id")["title"]
    rows = []
    for rec in entity_table.itertuples():
        is_person, confidence = person_like(rec.entity, rules)
        if not is_person or rec.entity_type in ("organisation", "technology_system"):
            continue
        ids = sorted(long.loc[long["entity_id"] == rec.entity_id, "incident_id"].unique())[:3]
        rows.append(
            {
                "entity_id": rec.entity_id,
                "entity": rec.entity,
                "n_incidents": rec.n_incidents,
                "roles": rec.roles,
                "candidate_confidence": confidence,
                "review_priority": "C"
                if confidence == "low"
                else ("A" if rec.n_incidents <= 2 else "B"),
                "entity_type": rec.entity_type,
                "example_incident_ids": ";".join(map(str, ids)),
                "example_titles": " || ".join(str(titles.get(i, "")) for i in ids[:2]),
                "is_named_person": "",
                "notes": "",
            }
        )
    out = pd.DataFrame(rows, columns=_REVIEW_COLUMNS)
    return out.sort_values(["review_priority", "n_incidents"], ascending=[True, False]).reset_index(
        drop=True
    )


def write_review_file(candidates: pd.DataFrame, path: Path = REVIEW_PATH) -> bool:
    """Write the review template unless a file with your decisions already exists.

    If the file exists, new candidates are appended and every existing row (including your answers)
    is left untouched. Returns True if the file changed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        new = candidates[~candidates["entity_id"].isin(existing["entity_id"])]
        if new.empty:
            return False
        combined = pd.concat([existing, new.astype(str)], ignore_index=True)
    else:
        combined = candidates.astype(str)
    combined.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return True


def load_named_persons(path: Path = REVIEW_PATH) -> set[str]:
    """Entity ids marked is_named_person = yes (y / true / 1 also accepted); they are excluded."""
    if not path.exists():
        return set()
    review = pd.read_csv(path, dtype=str, keep_default_na=False)
    answer = review["is_named_person"].str.strip().str.lower()
    return set(review.loc[answer.isin({"yes", "y", "true", "1"}), "entity_id"])


def review_progress(path: Path = REVIEW_PATH) -> dict[str, int]:
    """How many review rows are answered / confirmed (for the registry and the phase report)."""
    if not path.exists():
        return {"rows": 0, "answered": 0, "named_persons": 0}
    review = pd.read_csv(path, dtype=str, keep_default_na=False)
    answer = review["is_named_person"].str.strip().str.lower()
    return {
        "rows": len(review),
        "answered": int((answer != "").sum()),
        "named_persons": int(answer.isin({"yes", "y", "true", "1"}).sum()),
    }
