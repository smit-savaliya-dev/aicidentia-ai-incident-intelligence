"""Phase 7 orchestration: build the index, evaluate every method in three settings, choose the deployed method."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from aicidentia import cache, registry
from aicidentia.config import PROJECT_ROOT, load_config
from aicidentia.data import clean, load
from aicidentia.features.text import embed_sentences, tokenize
from aicidentia.models import retrieval as R
from aicidentia.viz import plots, style

logger = logging.getLogger(__name__)
PRODUCER = "aicidentia.models.retrieval_outputs.run_retrieval"
SOURCE = "data/processed/incidents_clean.parquet + snapshot editor_similar_incidents"
BASELINE = "bm25"
PRIMARY = "rr"  # mean reciprocal rank (MRR)
METRICS = ["rr", "p@5", "r@10", "hit@5", "ndcg@10", "dissimilar@5", "rr@3", "p@3", "hit@3"] + [
    f"recall@{k}" for k in R.KS
]
SETTINGS = {
    "all": "all other incidents are candidates; symmetrised editor links",
    "earlier": "only incidents strictly earlier than the query are candidates (find precedents)",
    "directed": "only links in the direction the editor recorded them",
}
SETTINGS_SHORT = {
    "all": "all incidents as candidates",
    "earlier": "earlier incidents only",
    "directed": "directed links",
}


@dataclass
class RetrievalResults:
    """Everything Phase 7 produces."""

    index: R.RetrievalIndex
    gt: R.GroundTruth
    per_query: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: dict[str, pd.DataFrame] = field(default_factory=dict)
    comparisons: dict[str, pd.DataFrame] = field(default_factory=dict)
    suggestions: pd.DataFrame | None = None
    examples: pd.DataFrame | None = None
    deployed: str = ""
    deployed_reason: str = ""
    facts: dict[str, Any] = field(default_factory=dict)


def collection() -> pd.DataFrame:
    """The incidents that are indexed (incident_id, title, text, date, year)."""
    return clean.load_processed("incidents_clean")[
        ["incident_id", "title", "text", "date", "year"]
    ].reset_index(drop=True)


def cached_tokens(frame: pd.DataFrame) -> list[list[str]]:
    """BM25 tokens of every incident (the project's shared tokenisation), cached in data/interim/."""
    return cache.cached(
        "bm25_tokens",
        (R.fingerprint(frame),),
        lambda: tokenize(frame["text"].tolist()),
        version="1",
    )


def build_default_index(
    frame: pd.DataFrame | None = None, local_files_only: bool = False
) -> tuple[R.RetrievalIndex, list[str]]:
    """Index with every method whose inputs are available. Embeddings come from the disk cache; if they are
    not cached and the model cannot be loaded (offline, missing), that method is skipped and reported."""
    frame = collection() if frame is None else frame
    models = load_config()["models"]
    embeddings, names, missing = {}, {}, []
    for method, key in R.EMBEDDING_KEYS.items():
        try:
            embeddings[method] = embed_sentences(
                frame["text"].tolist(), models[key], local_files_only=local_files_only
            )
            names[method] = models[key]
        except Exception as exc:  # a missing model must not break lexical search
            logger.warning("Embedding method %s unavailable: %s", method, exc)
            missing.append(method)
    return R.build_index(frame, cached_tokens(frame), tokenize, embeddings, names), missing


def choose_deployed(summary: pd.DataFrame, comparisons: pd.DataFrame) -> tuple[str, str]:
    """Same rule as Phase 6: the method with the best mean MRR, unless its paired-bootstrap advantage over the
    lexical baseline (BM25) is not significant (95% CI includes 0); then the baseline, which needs no model."""
    best = str(summary.sort_values(PRIMARY, ascending=False).iloc[0]["method"])
    if best == BASELINE:
        return best, "best mean MRR"
    row = comparisons.set_index("method").loc[best]
    if row["ci_low"] <= 0 <= row["ci_high"]:
        return BASELINE, (
            f"{R.LABELS[best]} has the best mean MRR, but its advantage over BM25 is not significant by the primary test "
            f"(paired bootstrap difference {row['diff']:+.3f}, 95% CI [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]); "
            "BM25 is deployed because it needs no embedding model and works offline"
        )
    return (
        best,
        f"best mean MRR and significantly better than BM25 (difference {row['diff']:+.3f}, 95% CI [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}])",
    )


def pick_examples(
    per_query: pd.DataFrame,
    index: R.RetrievalIndex,
    gt: R.GroundTruth,
    method: str,
    seed: int,
) -> pd.DataFrame:
    """Three illustrative queries for `method` (two found in the top 5, one missed), chosen by seed, with titles only."""
    rng = np.random.default_rng(seed)
    sub = per_query[per_query["method"] == method]
    picks = list(rng.choice(sub.loc[sub["hit@5"] == 1, "query"].to_numpy(), 2, replace=False))
    misses = sub.loc[sub["hit@5"] == 0, "query"].to_numpy()
    if len(misses):
        picks.append(int(rng.choice(misses)))
    rows = []
    for q in picks:
        scores = R.score_rows(index, method, [index.row_of[int(q)]])[0]
        ranked = R.rank_candidates(scores, index, exclude_ids=[int(q)])
        first = next(i for i, r in enumerate(ranked, 1) if r in gt.relevant[int(q)])
        for rank, r in enumerate(ranked[:5], 1):
            rows.append(
                {
                    "query": int(q),
                    "query_title": index.titles[index.row_of[int(q)]],
                    "rank": rank,
                    "incident_id": r,
                    "title": index.titles[index.row_of[r]],
                    "editor_similar": r in gt.relevant[int(q)],
                    "first_relevant_rank": first,
                }
            )
    return pd.DataFrame(rows)


def run_retrieval() -> RetrievalResults:
    """Evaluate all retrieval methods against the editor-marked similar incidents."""
    cfg = load_config()
    seed = int(cfg["random_seed"])
    n_boot = int(cfg["retrieval"]["n_boot"])
    frame = collection()
    index, missing = build_default_index(frame)
    if missing:
        raise RuntimeError(f"embedding methods unavailable for the evaluation: {missing}")
    gt = R.build_ground_truth(load.load_snapshot_incidents(), set(frame["incident_id"].astype(int)))
    res = RetrievalResults(index, gt)
    variants = {
        "all": (gt, False),
        "earlier": (gt, True),
        "directed": (
            R.GroundTruth(gt.directed, gt.directed, gt.dissimilar, gt.aiid_suggestions),
            False,
        ),
    }
    for setting, (truth, earlier) in variants.items():
        pq = pd.concat(
            [R.evaluate_method(index, m, truth, earlier) for m in R.METHODS], ignore_index=True
        )
        res.per_query[setting] = pq
        res.summary[setting] = R.summarise(pq, METRICS, n_boot, seed)
        res.comparisons[setting] = R.paired_comparisons(pq, BASELINE, PRIMARY, n_boot, seed)
    res.suggestions = R.summarise(
        R.evaluate_suggestions(gt, gt.queries), ["rr@3", "p@3", "hit@3"], n_boot, seed
    )
    res.deployed, res.deployed_reason = choose_deployed(res.summary["all"], res.comparisons["all"])
    res.examples = pick_examples(res.per_query["all"], index, gt, res.deployed, seed)
    s = res.summary["all"].set_index("method")
    best = str(s[PRIMARY].idxmax())
    res.facts = {
        "indexed_incidents": len(frame),
        "queries": len(gt.queries),
        "queries_earlier": int(res.summary["earlier"]["queries"].iloc[0]),
        "queries_directed": int(res.summary["directed"]["queries"].iloc[0]),
        "relevant_pairs": int(sum(len(v) for v in gt.relevant.values()) // 2),
        "relevant_per_query_mean": float(np.mean([len(v) for v in gt.relevant.values()])),
        "dissimilar_queries": int(sum(1 for q in gt.queries if q in gt.dissimilar)),
        "oracle": R.oracle(gt, gt.queries),
        "aiid_suggestion_overlap": R.suggestion_overlap(gt),
        "best_method": best,
        "best_mrr": float(s.loc[best, PRIMARY]),
        "bm25_mrr": float(s.loc["bm25", PRIMARY]),
        "mrr": {m: float(s.loc[m, PRIMARY]) for m in s.index},
        "ndcg10": {m: float(s.loc[m, "ndcg@10"]) for m in s.index},
        "p5": {m: float(s.loc[m, "p@5"]) for m in s.index},
        "hit5": {m: float(s.loc[m, "hit@5"]) for m in s.index},
        "deployed": res.deployed,
        "deployed_reason": res.deployed_reason,
        "fallback": BASELINE,
        "aiid_nlp_rr3": float(res.suggestions["rr@3"].iloc[0]),
    }
    return res


# --------------------------------------------------------------------------------------
# Artefacts
# --------------------------------------------------------------------------------------


def register(res: RetrievalResults) -> None:
    for key, value in res.facts.items():
        registry.record(f"rq4.{key}", value, SOURCE, PRODUCER)
    for setting in SETTINGS:
        registry.record(f"rq4.summary_{setting}", res.summary[setting].round(5), SOURCE, PRODUCER)
        registry.record(
            f"rq4.comparisons_{setting}", res.comparisons[setting].round(5), SOURCE, PRODUCER
        )


def save_tables(res: RetrievalResults) -> None:
    for setting in SETTINGS:
        style.save_table(res.summary[setting].round(4), f"rq4_summary_{setting}")
        style.save_table(res.comparisons[setting].round(4), f"rq4_comparisons_{setting}")
    style.save_table(res.per_query["all"].round(4), "rq4_per_query_all")
    style.save_table(res.suggestions.round(4), "rq4_aiid_suggestions")
    style.save_table(res.examples, "rq4_examples")


def make_figures(res: RetrievalResults) -> dict[str, Any]:
    figs = {
        "fig_rq4_1_method_comparison": plots.plot_retrieval_comparison(
            {SETTINGS_SHORT[s]: res.summary[s] for s in ("all", "earlier")},
            R.LABELS,
            res.facts["oracle"]["p@5"],
        ),
        "fig_rq4_2_recall_at_k": plots.plot_recall_at_k(
            {SETTINGS_SHORT[s]: res.summary[s] for s in ("all", "earlier")}, R.KS, R.LABELS
        ),
    }
    for name, fig in figs.items():
        style.save_figure(fig, name, str(load_config()["snapshot_date"]))
    return figs


def write_all(res: RetrievalResults) -> None:
    from aicidentia.models import retrieval_docs

    register(res)
    save_tables(res)
    make_figures(res)
    retrieval_docs.write_retrieval_doc(res, PROJECT_ROOT / "docs" / "rq4_similarity.md")
