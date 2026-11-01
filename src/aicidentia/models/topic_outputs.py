"""Phase 3 artefacts: chosen-model tables, per-incident assignments, topic-label template, registry, figures, doc."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aicidentia import cache, registry
from aicidentia.config import PROJECT_ROOT, load_config
from aicidentia.data import clean
from aicidentia.features.text import Corpus, embed_sentences, topic_settings
from aicidentia.models import topic_experiment as E
from aicidentia.models import topics as M
from aicidentia.viz import plots, style

logger = logging.getLogger(__name__)
PRODUCER = "aicidentia.models.topic_experiment.run_experiment"
SOURCE = "data/processed/incidents_clean.parquet"
LABELS_PATH = PROJECT_ROOT / "data" / "manual" / "topic_labels.csv"
LABEL_COLUMNS = [
    "model_key",
    "topic_id",
    "n_incidents",
    "top_10_terms",
    "representative_incident_ids",
    "representative_titles",
    "suggested_label",
    "final_label",
    "notes",
]


def chosen_fit(result: E.ExperimentResult, selection: E.Selection) -> M.TopicFit:
    """The selected configuration fitted with the base seed (config `random_seed`)."""
    return result.fits[(selection.winner_config, int(load_config()["random_seed"]))]


def suggested_label(terms: list[str], n: int = 3) -> str:
    """Automatic placeholder label: the first `n` terms joined by ' / '. NOT a human judgement."""
    return " / ".join(terms[:n])


def model_key(config_label: str, terms: list[list[str]]) -> str:
    """Identifies the exact topic solution; used to protect your final labels from a changed model."""
    return f"{config_label}|{cache.hash_parts(terms)[:10]}"


def topic_table(
    fit: M.TopicFit, corpus: Corpus, incidents: pd.DataFrame, embeddings: np.ndarray | None
) -> pd.DataFrame:
    """One row per topic: size, terms, five representative incidents and the automatic label."""
    reps = M.representative_docs(fit, embeddings, n=5)
    titles = incidents.set_index("incident_id")["title"]
    rows = []
    for topic in range(fit.n_topics):
        members = int((fit.assignments == topic).sum())
        rep_ids = [corpus.incident_ids[i] for i in reps[topic]]
        rows.append(
            {
                "topic": topic,
                "topic_id": fit.topic_ids[topic],
                "n_incidents": members,
                "share": members / len(corpus.incident_ids),
                "terms": fit.terms[topic],
                "top_10_terms": ", ".join(fit.terms[topic]),
                "representative_incident_ids": ";".join(map(str, rep_ids)),
                "representative_titles": " || ".join(str(titles[i]) for i in rep_ids),
                "suggested_label": suggested_label(fit.terms[topic]),
            }
        )
    return pd.DataFrame(rows)


def write_topic_labels(table: pd.DataFrame, key: str, path: Path = LABELS_PATH) -> str:
    """Create data/manual/topic_labels.csv with a BLANK `final_label` column for you to fill.

    If the file already exists for the same topic solution it is left untouched (your answers are safe).
    If the solution changed and you had already answered, the new template goes to
    topic_labels.proposed.csv instead. Returns 'created', 'unchanged', 'replaced' or 'proposed'.
    """
    out = table.assign(model_key=key, final_label="", notes="")[LABEL_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        same = set(existing["model_key"]) == {key}
        answered = bool(
            (existing["final_label"].str.strip() != "").any()
            or (existing["notes"].str.strip() != "").any()
        )
        if same:
            return "unchanged"
        if answered:
            out.to_csv(
                path.with_name("topic_labels.proposed.csv"),
                index=False,
                encoding="utf-8",
                lineterminator="\n",
            )
            logger.warning(
                "Topic solution changed but %s has your answers; wrote topic_labels.proposed.csv",
                path.name,
            )
            return "proposed"
        out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        return "replaced"
    out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return "created"


def build_assignments(
    fit: M.TopicFit, corpus: Corpus, table: pd.DataFrame, key: str
) -> pd.DataFrame:
    """Per-incident dominant topic with the automatic label (saved to data/processed/incident_topics.parquet)."""
    out = M.assign_topics(fit, corpus)
    labels = table.set_index("topic_id")["suggested_label"]
    out["suggested_label"] = out["topic_id"].map(labels)
    out["model_key"] = key
    return out


def register(
    result: E.ExperimentResult,
    selection: E.Selection,
    corpus: Corpus,
    table: pd.DataFrame,
    key: str,
    status: str,
) -> int:
    """Record the Phase 3 facts under `rq1a.*`."""
    cfg = topic_settings()
    count = 0

    def rec(name: str, value: Any) -> None:
        nonlocal count
        registry.record(f"rq1a.{name}", value, SOURCE, PRODUCER)
        count += 1

    lengths = np.array([len(t) for t in corpus.tokens])
    rec("corpus.documents", len(corpus.docs))
    rec("corpus.vocabulary", len(corpus.vocabulary))
    rec("corpus.tokens_per_doc_mean", float(lengths.mean()))
    rec("corpus.tokens_per_doc_min", int(lengths.min()))
    rec("grid.n_configs", int(len(result.summary)))
    rec("grid.n_fits", int(len(result.runs)))
    rec("grid.seeds", result.seeds)
    rec("grid.eligible_topics", cfg["eligible_topics"])
    rec("selection.winner_contender", selection.winner)
    rec("selection.winner_config", selection.winner_config)
    rec("selection.sensitivity", selection.sensitivity)
    rec("selection.ranks", selection.ranks)
    rec(
        "selection.best_per_contender",
        selection.best_per_contender[
            [
                "contender",
                "label",
                "n_topics_mean",
                "c_v_mean",
                "c_v_sd",
                "c_npmi_mean",
                "c_npmi_sd",
                "diversity_mean",
                "diversity_sd",
                "stability",
            ]
        ],
    )
    best = selection.best_per_contender.set_index("label").loc[selection.winner_config]
    for metric in (
        "n_topics_mean",
        "c_v_mean",
        "c_v_sd",
        "c_npmi_mean",
        "c_npmi_sd",
        "diversity_mean",
        "diversity_sd",
        "stability",
        "outlier_share_before_mean",
    ):
        rec(f"chosen.{metric}", float(best[metric]))
    rec("chosen.model_key", key)
    rec("chosen.n_topics", len(table))
    rec("chosen.topic_sizes", table[["topic_id", "n_incidents", "suggested_label"]])
    rec("chosen.largest_topic_share", float(table["share"].max()))
    rec("chosen.smallest_topic_n", int(table["n_incidents"].min()))
    rec("manual.topic_labels_status", status)
    return count


def save_tables(result: E.ExperimentResult, selection: E.Selection, table: pd.DataFrame) -> None:
    """reports/tables/rq1a_*.{csv,md}."""
    style.save_table(result.runs.round(4), "rq1a_runs")
    summary = result.summary.copy()
    for col in summary.select_dtypes("float").columns:
        summary[col] = summary[col].round(4)
    style.save_table(summary, "rq1a_summary")
    style.save_table(selection.best_per_contender.round(4), "rq1a_best_per_contender")
    style.save_table(selection.ranks.round(2), "rq1a_selection_ranks")
    style.save_table(table.drop(columns=["terms"]).round(4), "rq1a_chosen_topics")


def make_figures(
    result: E.ExperimentResult, selection: E.Selection, table: pd.DataFrame
) -> dict[str, Any]:
    """Three figures: metrics by configuration, best-per-contender comparison, topic sizes."""
    lo, hi = topic_settings()["eligible_topics"]
    figs = {
        "fig_rq1_1_coherence_by_config": plots.plot_coherence_by_config(result.summary, (lo, hi)),
        "fig_rq1_2_model_comparison": plots.plot_best_model_comparison(
            selection.best_per_contender, selection.winner
        ),
        "fig_rq1_3_topic_sizes": plots.plot_topic_sizes(
            table, f"Topics of the chosen model ({selection.winner_config})"
        ),
    }
    for name, fig in figs.items():
        style.save_figure(fig, name, str(load_config()["snapshot_date"]))
    return figs


def write_all(corpus: Corpus, result: E.ExperimentResult, selection: E.Selection) -> dict[str, Any]:
    """Everything for the phase; returns the chosen fit, table and label status."""
    from aicidentia.models import topic_docs

    incidents = clean.load_processed("incidents_clean")
    fit = chosen_fit(result, selection)
    embeddings = None
    if fit.family == "BERTopic":
        embeddings = embed_sentences(corpus.texts, load_config()["models"][fit.params["embedding"]])
    table = topic_table(fit, corpus, incidents, embeddings)
    key = model_key(selection.winner_config, fit.terms)
    status = write_topic_labels(table, key)
    assignments = build_assignments(fit, corpus, table, key)
    clean.save_processed(assignments, "incident_topics")
    register(result, selection, corpus, table, key, status)
    registry.record(
        "rq1a.chosen.dominant_weight_median",
        float(np.nanmedian(assignments["weight"])) if fit.weights is not None else None,
        SOURCE,
        PRODUCER,
    )
    registry.record(
        "rq1a.chosen.dominant_weight_below_half_share",
        float((assignments["weight"] < 0.5).mean()) if fit.weights is not None else None,
        SOURCE,
        PRODUCER,
    )
    save_tables(result, selection, table)
    make_figures(result, selection, table)
    topic_docs.write_topic_doc(
        corpus, result, selection, table, fit, status, PROJECT_ROOT / "docs" / "rq1_topic_models.md"
    )
    return {"fit": fit, "table": table, "label_status": status, "model_key": key}
