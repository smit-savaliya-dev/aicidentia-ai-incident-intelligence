"""Phase 2 artefacts: registry entries, tables, EDA figures and docs/data_preparation.md."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from aicidentia import registry
from aicidentia.config import PROJECT_ROOT
from aicidentia.data import audit, load
from aicidentia.data.prepare import PrepResults
from aicidentia.viz import plots, style

logger = logging.getLogger(__name__)

PRODUCER = "aicidentia.data.prepare.build_processed"
SOURCE = "data/processed/"
_PLAIN_FACTS = (
    "incidents_raw",
    "incidents_clean",
    "aiid_merge_pairs",
    "merge_map_size",
    "user_merges_applied",
    "entity_aliases_applied",
    "entity_aliases_manual",
    "entity_aliases_skipped_aiid",
    "entities_before_alias",
    "entities_after_alias",
    "entities_in_table",
    "incident_entities_rows",
    "stub_reports_dropped",
    "reports_clean",
    "reports_unlinked_kept",
    "incidents_with_zero_clean_reports",
    "report_count_differs_from_published",
    "incidents_with_deployer",
    "incidents_with_harmed",
    "validation_checks",
    "validation_failed",
)
FIGURES = {
    "fig_eda_1_incidents_per_year": "Incidents per year by MIT-label availability",
    "fig_eda_2_top_deployers": "Top alleged deployers (organisations)",
    "fig_eda_3_top_harmed": "Top harmed population groups",
    "fig_eda_4_label_distributions": "Label distributions (MIT and CSETv1)",
}


def register(res: PrepResults) -> int:
    """Record the Phase 2 facts in reports/results.json under `prep.*`."""
    count = 0
    for key in _PLAIN_FACTS:
        registry.record(f"prep.{key}", res.facts[key], SOURCE, PRODUCER)
        count += 1
    for group in (
        "entity_type_counts",
        "privacy",
        "entity_review",
        "near_duplicate_review",
        "review_candidates",
    ):
        for key, value in res.facts[group].items():
            registry.record(f"prep.{group}.{key}", value, SOURCE, PRODUCER)
            count += 1
    registry.record(
        "prep.merged_ids_present_in_excel",
        res.facts["merged_ids_present_in_excel"],
        SOURCE,
        PRODUCER,
    )
    return count + 1


def _top_by_type(entity_table: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        entity_table.sort_values("n_incidents", ascending=False)
        .groupby("entity_type", sort=True)
        .head(n)[["entity_type", "entity", "n_incidents", "roles", "entity_type_rule"]]
        .sort_values(["entity_type", "n_incidents"], ascending=[True, False])
        .reset_index(drop=True)
    )


def schema_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Column, dtype and non-null count of a clean table."""
    return pd.DataFrame(
        {
            "table": name,
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "non_null": [
                int(df[c].map(lambda v: bool(len(v)) if isinstance(v, list) else pd.notna(v)).sum())
                for c in df.columns
            ],
        }
    )


def save_tables(res: PrepResults) -> None:
    """Write Phase 2 tables to reports/tables/prep_*.{csv,md}."""
    t = res.tables
    style.save_table(t["alias_candidates"], "prep_alias_list")
    style.save_table(t["alias_skipped"], "prep_alias_skipped")
    counts = (
        t["entity_table"]["entity_type"]
        .value_counts()
        .rename_axis("entity_type")
        .rename("entities")
        .reset_index()
    )
    style.save_table(counts, "prep_entity_type_counts")
    style.save_table(_top_by_type(t["entity_table"]), "prep_entity_top_by_type")
    checks = pd.DataFrame([c.__dict__ for c in res.checks])
    style.save_table(checks, "prep_validation_checks")
    schema = pd.concat(
        [
            schema_table(t[n], n)
            for n in ("incidents_clean", "incident_entities", "entity_table", "reports_clean")
        ]
    )
    style.save_table(schema, "prep_schema")


def make_figures(res: PrepResults) -> dict[str, Any]:
    """Draw and save the four EDA figures."""
    t = res.tables
    inc, ie = t["incidents_clean"], t["incident_entities"]
    partial_year = int(res.snapshot_date[:4])

    def top(role: str, entity_type: str) -> pd.DataFrame:
        sub = ie[(ie["role"] == role) & (ie["entity_type"] == entity_type)]
        return (
            sub.groupby("entity")["incident_id"]
            .nunique()
            .sort_values(ascending=False)
            .rename("n_incidents")
            .reset_index()
        )

    cset = inc[inc["has_cset"]]
    panels = [
        (
            f"MIT risk domain (n={int(inc['mit_risk_domain'].notna().sum())})",
            inc["mit_risk_domain"].value_counts(),
        ),
        (
            f"MIT intent (n={int(inc['mit_intent'].notna().sum())})",
            inc["mit_intent"].value_counts(),
        ),
        (
            f"CSETv1 AI harm level (n={int(cset['cset_ai_harm_level'].notna().sum())})",
            cset["cset_ai_harm_level"].value_counts(),
        ),
        (
            f"CSETv1 tangible harm (n={int(cset['cset_tangible_harm'].notna().sum())})",
            cset["cset_tangible_harm"].value_counts(),
        ),
    ]
    panels = [(title, counts.rename(lambda s: str(s)[:60])) for title, counts in panels]
    figs = {
        "fig_eda_1_incidents_per_year": plots.plot_incidents_per_year_by_label(inc, partial_year),
        "fig_eda_2_top_deployers": plots.plot_top_entities(
            top("deployer", "organisation"),
            "Top alleged deployers (organisations)",
            "Incidents naming the entity (count)",
            style.PALETTE["blue"],
        ),
        "fig_eda_3_top_harmed": plots.plot_top_entities(
            top("harmed", "population_group"),
            "Top harmed population groups",
            "Incidents naming the group (count)",
            style.PALETTE["vermillion"],
        ),
        "fig_eda_4_label_distributions": plots.plot_label_distributions(panels),
    }
    for name, fig in figs.items():
        style.save_figure(fig, name, res.snapshot_date)
    return figs


def write_all(res: PrepResults) -> None:
    """Registry + tables + figures + docs/data_preparation.md."""
    from aicidentia.data import prepare_docs

    register(res)
    save_tables(res)
    make_figures(res)
    prepare_docs.write_data_preparation(res, PROJECT_ROOT / "docs" / "data_preparation.md")
    # Keep the audit's source reference to the raw Excel for provenance in the registry.
    registry.record(
        "prep.source_excel", audit.project_relative(load.find_excel_export()), SOURCE, PRODUCER
    )
