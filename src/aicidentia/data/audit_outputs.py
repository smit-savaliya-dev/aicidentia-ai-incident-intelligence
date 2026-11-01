"""Write the Phase 1 artefacts: registry entries, tables (CSV + Markdown) and figures (PNG + SVG)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from aicidentia import registry
from aicidentia.config import PROJECT_ROOT
from aicidentia.data import load
from aicidentia.data.audit import AuditResults, project_relative
from aicidentia.viz import plots, style

logger = logging.getLogger(__name__)

PRODUCER = "aicidentia.data.audit.run_audit"
# Scalar groups derived from the snapshot (everything else comes from the Excel export).
_SNAPSHOT_GROUPS = {
    "similarity",
    "excel_vs_snapshot",
    "taxonomy_provenance",
    "entities",
    "mit_records",
    "snapshot_rows",
}
# Tables small enough to be useful in the registry (dashboard / report read them from there).
_REGISTRY_TABLES = (
    "incidents_per_year",
    "mit_domain_distribution",
    "mit_domain_by_period",
    "mit_coverage_by_year",
    "sheet_shapes",
)

FIGURES = {
    "fig_audit_1_incidents_per_year": "Incidents per year",
    "fig_audit_2_taxonomy_coverage": "Taxonomy coverage and MIT coverage by year",
    "fig_audit_3_mit_domain_balance": "MIT risk-domain class balance",
    "fig_audit_4_text_lengths": "Title and description length",
    "fig_audit_5_reports_per_incident": "Reports per incident",
}


def _flatten(prefix: str, value: Any) -> dict[str, Any]:
    """Flatten one nesting level of dict scalars into dotted keys."""
    if isinstance(value, dict) and value and all(isinstance(k, str) for k in value):
        if prefix.endswith("year_range_recommendation"):
            return {prefix: value}
        return {f"{prefix}.{k}": v for k, v in value.items()}
    return {prefix: value}


def register_results(res: AuditResults) -> int:
    """Record every scalar fact and a few compact tables in reports/results.json."""
    excel = project_relative(load.find_excel_export())
    snap = load.find_snapshot_dir()
    snapshot_src = project_relative(snap) if snap else excel
    count = 0
    for name, value in res.scalars.items():
        source = snapshot_src if name in _SNAPSHOT_GROUPS else excel
        for key, val in _flatten(f"audit.{name}", value).items():
            registry.record(key, val, source, PRODUCER)
            count += 1
    for name in _REGISTRY_TABLES:
        registry.record(f"audit.table.{name}", res.tables[name], excel, PRODUCER)
        count += 1
    registry.record("audit.snapshot_date", res.snapshot_date, excel, PRODUCER)
    logger.info("Recorded %d registry entries", count + 1)
    return count + 1


def save_tables(res: AuditResults) -> list[Path]:
    """Write every audit table to reports/tables/audit_<name>.{csv,md}."""
    written: list[Path] = []
    for name, df in res.tables.items():
        written += style.save_table(_for_display(df), f"audit_{name}")
    return written


def _for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Round floats so the Markdown tables are readable (CSV keeps the rounded values too)."""
    out = df.copy()
    for col in out.select_dtypes("float").columns:
        out[col] = out[col].round(4)
    return out


def make_figures(res: AuditResults) -> dict[str, Any]:
    """Draw and save the five exploratory figures; returns {name: Figure}."""
    inc = load.load_incidents()
    t, s = res.tables, res.scalars
    window = s["year_range_recommendation"]
    figs = {
        "fig_audit_1_incidents_per_year": plots.plot_incidents_per_year(
            t["incidents_per_year"], trend_window=(window["start"], window["end"])
        ),
        "fig_audit_2_taxonomy_coverage": plots.plot_taxonomy_coverage(
            {
                "MIT": s["mit_domain_n"],
                "CSETv1": s["csetv1_incidents"],
                "GMF": s["gmf_incidents"],
                "No taxonomy": s["no_taxonomy_incidents"],
            },
            total=s["n_incidents"],
            mit_by_year=t["mit_coverage_by_year"],
        ),
        "fig_audit_3_mit_domain_balance": plots.plot_class_balance(
            t["mit_domain_distribution"], f"MIT risk domain (n={s['mit_domain_n']})"
        ),
        "fig_audit_4_text_lengths": plots.plot_text_lengths(inc),
        "fig_audit_5_reports_per_incident": plots.plot_reports_per_incident(
            t["reports_per_incident"]
        ),
    }
    for name, fig in figs.items():
        style.save_figure(fig, name, res.snapshot_date)
    return figs


def write_all(res: AuditResults) -> None:
    """Registry + tables + figures + the two audit documents."""
    from aicidentia.data import audit_docs

    register_results(res)
    save_tables(res)
    make_figures(res)
    audit_docs.write_data_audit(res, PROJECT_ROOT / "docs" / "data_audit.md")
    audit_docs.write_datasheet(res, PROJECT_ROOT / "docs" / "datasheet.md")
