"""Build docs/data_preparation.md from the Phase 2 results."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from aicidentia.data import clean
from aicidentia.data.audit_docs import n
from aicidentia.data.prepare import PrepResults
from aicidentia.data.prepare_outputs import FIGURES, _top_by_type, schema_table
from aicidentia.viz.style import to_markdown

logger = logging.getLogger(__name__)


def _figure(name: str, caption: str, snapshot: str) -> str:
    return f"![{caption}](../reports/figures/{name}.png)\n\n*Figure: {caption}. Source: AIID snapshot {snapshot}.*\n"


def write_data_preparation(res: PrepResults, path: Path) -> None:
    """Write docs/data_preparation.md."""
    f, t, snap = res.facts, res.tables, res.snapshot_date
    entity_table = t["entity_table"]
    type_counts = entity_table["entity_type"].value_counts()
    review = f["entity_review"]
    privacy = f["privacy"]
    dupes = f["near_duplicate_review"]
    rc = f["review_candidates"]
    L: list[str] = []
    add = L.append
    add("# Data preparation (Phase 2)\n")
    add(
        f"_Auto-generated from `aicidentia.data.prepare` (AIID snapshot **{snap}**); re-run `scripts/run_phase2_prepare.py` to refresh. "
        "Numbers are registered in `reports/results.json` under `prep.*`. Raw files in `data/raw/` are never modified._\n"
    )
    add("## Outputs (`data/processed/*.parquet`, git-ignored)\n")
    add(
        f"| Table | Rows | What it is |\n|---|---|---|\n"
        f"| `incidents_clean` | {n(f['incidents_clean'])} | one row per incident: ids, dates, `text`, report counts, tidied MIT/GMF/CSETv1 labels |\n"
        f"| `incident_entities` | {n(f['incident_entities_rows'])} | long table (incident, role, canonical entity, `entity_type`) |\n"
        f"| `entity_table` | {n(f['entities_in_table'])} | one row per canonical entity: name, type, incident count, roles |\n"
        f"| `reports_clean` | {n(f['reports_clean'])} | non-stub reports with full text and linked incident ids |\n"
    )
    add("## Cleaning decisions\n")
    add(
        "- **Sources.** Excel for incidents and labels; snapshot for entity slugs, incident-to-report links and full text (Phase 1 decision). "
        "Excel entity strings are not used (lossy).\n"
        f'- **Text field.** `text = title + ". " + description` (no doubled full stop). Whitespace is collapsed; no lower-casing or stop-word removal here (that belongs to feature extraction).\n'
        "- **Time.** `year`, `quarter` (1-4) and `year_quarter` (e.g. `2024Q3`) from the harm date.\n"
        "- **Labels.** Columns are prefixed `mit_`, `gmf_`, `cset_` in snake_case. JSON-list cells become real lists (missing = empty list), True/False columns become nullable booleans, "
        "numeric columns nullable numbers, domain numbering is stripped. The CSETv1 `Entities` structure is kept as the JSON string `cset_entities_json`. "
        "Flags `has_mit`, `has_gmf`, `has_cset` come from `Data Sources`.\n"
        f"- **Dropped columns.** {', '.join(f'`{c}`' for c in clean.DROPPED_INCIDENT_COLUMNS)} (editor ids/notes, modification date) and the four Excel entity columns (rebuilt from slugs). "
        f"From reports: {', '.join(f'`{c}`' for c in clean.DROPPED_REPORT_COLUMNS)} (names of journalists and submitters, data minimisation).\n"
        f"- **Reports.** {n(f['stub_reports_dropped'])} stub reports (no title, URL or text) removed; {n(f['reports_unlinked_kept'])} real reports not linked to any incident are kept (flag `linked = False`). "
        f"Implausible dates (before 1990 or after the snapshot) become missing. `report_count_clean` (non-stub linked reports) differs from the published `report_count` for "
        f"{f['report_count_differs_from_published']} incidents (linked stubs). {f['incidents_with_zero_clean_reports']} incident has no real report.\n"
    )
    add("## Duplicate incidents\n")
    add(
        f"- AIID's merge list has {f['aiid_merge_pairs']} pairs, resolved transitively ({f['merge_map_size']} ids). The Excel export already excludes merged incidents "
        f"(merged ids present in Excel: {f['merged_ids_present_in_excel'] or 'none'}), so no incident was removed by this step.\n"
        f"- **Near-duplicate candidates for your review:** {dupes['rows']} pairs with TF-IDF cosine >= 0.6 in `data/manual/duplicate_review.csv`. "
        f"The `decision` column is blank; you answer `merge_into_a`, `merge_into_b` or `keep_separate` ({dupes['answered']} answered so far, {f['user_merges_applied']} merges applied). "
        "Re-run the build after answering.\n"
    )
    cand = t["near_duplicate_candidates"].head(8)[
        ["incident_a", "incident_b", "cosine", "title_a", "title_b"]
    ]
    add(to_markdown(cand))
    add("")
    add("## Entities\n")
    add(
        f"- **Aliases.** {f['entity_aliases_applied']} alias mappings are applied "
        f"(entities used by incidents: {n(f['entities_before_alias'])} before, {n(f['entities_after_alias'])} after). Sources: AIID's own duplicate list, entities whose names differ only in "
        "case/accents/punctuation/Inc, and your manual entries (`facebook -> meta`, the example from the brief). The full list is in `config/entity_aliases.yaml` (sections `manual`, "
        "`blocked`, `generated`; your edits to `manual`/`blocked` survive regeneration) and the table below.\n"
    )
    aliases = t["alias_candidates"].copy()
    add(
        to_markdown(
            aliases[["alias", "canonical", "source", "incidents_alias", "incidents_canonical"]]
        )
    )
    add("")
    skipped = t["alias_skipped"]
    add(
        f"AIID pairs **not** applied ({len(skipped)}): "
        + (
            "; ".join(f"`{r.alias}` -> `{r.canonical}` ({r.reason})" for r in skipped.itertuples())
            or "none"
        )
        + ". AIID's list contains cycles and canonical entities that do not exist in the Entities sheet; applying those would rename entities to slugs or merge them arbitrarily.\n"
    )
    add(
        'Parent/subsidiary relationships (Instagram, WhatsApp -> Meta) are not aliases and are not merged; AIID\'s relationship table only holds symmetric "related" links, not hierarchies.\n'
    )
    add("### Entity types (heuristic)\n")
    add(
        "The data has no entity-type field. `entity_type` is assigned by the ordered rules in `config/entity_types.yaml` (first match wins; the rule is stored in `entity_type_rule`) "
        "from the entity's name and the roles it plays. **These are heuristics for grouping, not ground truth**; correct any entity via `overrides` in that file.\n"
    )
    add(to_markdown(type_counts.rename_axis("entity_type").rename("entities").reset_index()))
    add("\nTop 10 entities per type, to eyeball the rules:\n")
    add(to_markdown(_top_by_type(entity_table, 10)))
    add(
        "\n**Known weaknesses seen when spot-checking snapshot 20261026** (do not trust the type of low-frequency entities): "
        "class-level institutions (Law firms, Universities, Hospitals) are typed `organisation`; a few company names that look like personal names "
        "(Boston Dynamics, Lockheed Martin) are typed `individual_candidate`; platforms that are both a company and an implicated system "
        "(for example X (Twitter), 55 incidents) are typed `technology_system`, while the bare entity Twitter is `organisation`. "
        "Fix any of these with `overrides` in `config/entity_types.yaml`; the network phase should group by type only after you have glanced at the top entities.\n"
    )
    add(
        "**Possible aliases not applied (your call; add them under `manual:` in `config/entity_aliases.yaml`):** "
        "Perplexity / Perplexity AI; Twitter / X (Twitter); Russian government / Government of Russia; Chinese government / Government of China; "
        "New York Police Department / New York City Police Department. These pairs were noticed in the frequency lists of this snapshot; "
        "whether they are the same entity for your analysis is a judgement.\n"
    )
    add("### Named individuals: your review\n")
    add(
        f"`data/manual/entity_review.csv` lists **{n(rc['total'])} entities with person-like names** (priority A: named in 1-2 incidents, {n(rc.get('priority_A', 0))}; "
        f"B: 3+ incidents, {n(rc.get('priority_B', 0))}; C: surname ending in -s, which may be a plural group, {n(rc.get('priority_C', 0))}). "
        "Each row shows the roles and example incident titles. **Fill `is_private_individual` with `yes` or `no`** (`no` = public figure or not a person). "
        f"Answered so far: {review['answered']} of {review['rows']}; confirmed private: {review['confirmed_private']}. "
        "Re-run the build afterwards; confirmed individuals are removed from `incident_entities`, `entity_table`, the network and the dashboard.\n"
    )
    add(
        f"Privacy switch: `privacy.exclude_unreviewed_individual_candidates` in `config/config.yaml` is currently **off**, so only confirmed individuals are excluded "
        f"(now: {privacy['excluded_total']} excluded; {n(privacy['unreviewed_candidates'])} person-like entities still unreviewed). Turning it on removes every unreviewed person-like entity until you clear it with `no`. "
        "Note that incident titles and descriptions are not edited; they come from AIID and the dashboard links back to AIID.\n"
    )
    add("## Validation\n")
    checks = pd.DataFrame(
        [
            {
                "table": c.table,
                "check": c.name,
                "result": "PASS" if c.passed else "FAIL",
                "detail": c.detail,
            }
            for c in res.checks
        ]
    )
    add(
        f"{f['validation_checks']} checks, {f['validation_failed']} failed (the build raises an error on any failure).\n"
    )
    add(to_markdown(checks))
    add("")
    add("## Schema\n")
    schema = pd.concat(
        [
            schema_table(t[x], x)
            for x in ("incidents_clean", "incident_entities", "entity_table", "reports_clean")
        ]
    )
    add(to_markdown(schema))
    add("")
    add("## Exploratory figures\n")
    for name, caption in FIGURES.items():
        add(_figure(name, caption, snap))
    add(
        "Top deployers show only entities typed `organisation` and the harmed chart only `population_group`; generic actors, technologies, abstract values and person-like entities are "
        "excluded from these charts by design. Organisations are **alleged** deployers; counts are numbers of incident records, not findings of fault.\n"
    )
    add("## Limitations\n")
    add(
        "- Entity types and aliases are heuristic; see the known misclassifications above.\n"
        "- Person-like names are candidates only; recall of the name heuristic is unknown, and individuals can also appear in titles and descriptions.\n"
        "- MIT labels are still missing for many recent incidents (Phase 1); `has_mit` marks them.\n"
        "- The near-duplicate threshold (0.6) is a review aid, not a statement that pairs are duplicates.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    logger.info("Wrote %s", path)
