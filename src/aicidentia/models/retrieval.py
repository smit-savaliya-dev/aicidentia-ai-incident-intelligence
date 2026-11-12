"""RQ4 - similar-incident retrieval: TF-IDF vs BM25 vs sentence embeddings (vs a hybrid), against editor links.

Every incident is indexed by its `text` (title + description). Given a query incident, a method scores all
other incidents; the ranking is compared with the incidents that AIID editors marked as similar
(`editor_similar_incidents` in the snapshot), symmetrised (a link A->B makes B relevant for A and A for B)
and restricted to incidents in the analysis set. Editor-marked *dissimilar* incidents are used as hard
negatives. Nothing is tuned on the ground truth: every parameter below is fixed in advance.

Methods
-------
* ``tfidf``  word 1-2-gram TF-IDF (sublinear tf, English stopwords, min_df 2), cosine similarity;
* ``bm25``   Okapi BM25 (k1 = 1.5, b = 0.75; rank-bm25) on the project's shared tokenisation (lemmatised,
  stopwords removed; no document-frequency filter);
* ``minilm`` / ``mpnet``  cosine similarity of normalised sentence embeddings (all-MiniLM-L6-v2, all-mpnet-base-v2);
* ``hybrid`` reciprocal rank fusion (k = 60) of BM25 and mpnet rankings.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from aicidentia import cache
from aicidentia.evaluation import metrics as M

logger = logging.getLogger(__name__)

METHODS = ("tfidf", "bm25", "minilm", "mpnet", "hybrid")
LEXICAL = ("tfidf", "bm25")
EMBEDDING_KEYS = {"minilm": "sentence_embedding_small", "mpnet": "sentence_embedding_large"}
LABELS = {
    "tfidf": "TF-IDF (1-2-grams)",
    "bm25": "BM25",
    "minilm": "MiniLM embeddings",
    "mpnet": "mpnet embeddings",
    "hybrid": "Hybrid (BM25 + mpnet, RRF)",
    "aiid_nlp": "AIID's own suggestions (top 3 only)",
}
HYBRID_PARTS = ("bm25", "mpnet")
RRF_K = 60
BM25_K1, BM25_B = 1.5, 0.75


# --------------------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------------------


@dataclass
class GroundTruth:
    """Editor judgements restricted to the analysis set (all dicts map incident id -> set of incident ids)."""

    relevant: dict[int, set[int]]  # symmetrised editor-similar links
    directed: dict[int, set[int]]  # links only in the direction the editor recorded them
    dissimilar: dict[int, set[int]]  # symmetrised editor-dissimilar links (hard negatives)
    aiid_suggestions: dict[int, list[int]]  # AIID's NLP suggestions, ranked (top 3)

    @property
    def queries(self) -> list[int]:
        return sorted(self.relevant)


def build_ground_truth(snapshot_incidents: pd.DataFrame, ids: set[int]) -> GroundTruth:
    """Symmetrised editor-similar / -dissimilar links among `ids` (self-links dropped)."""

    def links(column: str) -> dict[int, set[int]]:
        out: dict[int, set[int]] = {}
        for row in snapshot_incidents.itertuples():
            a = int(row.incident_id)
            for b in getattr(row, column):
                b = int(b)
                if a in ids and b in ids and a != b:
                    out.setdefault(a, set()).add(b)
        return out

    def symmetrise(directed: dict[int, set[int]]) -> dict[int, set[int]]:
        out: dict[int, set[int]] = {}
        for a, targets in directed.items():
            for b in targets:
                out.setdefault(a, set()).add(b)
                out.setdefault(b, set()).add(a)
        return out

    directed = links("editor_similar_incidents")
    suggestions = {}
    for row in snapshot_incidents.itertuples():
        a = int(row.incident_id)
        if a in ids:
            ranked = sorted(row.nlp_similar_incidents, key=lambda d: -float(d["similarity"]))
            suggestions[a] = [
                int(d["incident_id"])
                for d in ranked
                if int(d["incident_id"]) in ids and int(d["incident_id"]) != a
            ]
    return GroundTruth(
        symmetrise(directed),
        directed,
        symmetrise(links("editor_dissimilar_incidents")),
        suggestions,
    )


def suggestion_overlap(gt: GroundTruth) -> dict[str, float]:
    """How many editor links coincide with AIID's own NLP suggestions (a sign editors chose from them)."""
    pairs = [(a, b) for a, targets in gt.directed.items() for b in targets]
    in_suggestions = sum(
        1
        for a, b in pairs
        if b in gt.aiid_suggestions.get(a, []) or a in gt.aiid_suggestions.get(b, [])
    )
    return {
        "directed_links": len(pairs),
        "links_in_aiid_suggestions": in_suggestions,
        "share_in_aiid_suggestions": in_suggestions / len(pairs) if pairs else float("nan"),
    }


# --------------------------------------------------------------------------------------
# Index and scoring
# --------------------------------------------------------------------------------------


@dataclass
class RetrievalIndex:
    """Everything needed to score queries against the incident collection."""

    incident_ids: np.ndarray
    titles: list[str]
    dates: np.ndarray  # datetime64; used for the 'earlier incidents only' setting
    tfidf_vectoriser: Any
    tfidf_matrix: Any  # L2-normalised rows
    bm25: Any
    bm25_tokens: list[
        list[str]
    ]  # tokens of each indexed incident (its BM25 query when it is the query)
    tokenise: Callable[[Sequence[str]], list[list[str]]]
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)  # method -> normalised matrix
    embedding_models: dict[str, str] = field(default_factory=dict)  # method -> model name
    row_of: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.row_of = {int(i): k for k, i in enumerate(self.incident_ids)}

    @property
    def available_methods(self) -> list[str]:
        out = list(LEXICAL) + [m for m in ("minilm", "mpnet") if m in self.embeddings]
        if all(p in out for p in HYBRID_PARTS):
            out.append("hybrid")
        return out


def build_index(
    frame: pd.DataFrame,
    tokens: Sequence[Sequence[str]],
    tokenise: Callable[[Sequence[str]], list[list[str]]],
    embeddings: dict[str, np.ndarray] | None = None,
    embedding_models: dict[str, str] | None = None,
) -> RetrievalIndex:
    """Index `frame` (incident_id, title, text, date). `tokens[i]` are the BM25 tokens of row i."""
    from rank_bm25 import BM25Okapi
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectoriser = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"
    )
    matrix = vectoriser.fit_transform(frame["text"].tolist())  # rows are L2-normalised by default
    token_lists = [list(t) for t in tokens]
    bm25 = BM25Okapi([t if t else ["_empty_"] for t in token_lists], k1=BM25_K1, b=BM25_B)
    return RetrievalIndex(
        incident_ids=frame["incident_id"].to_numpy(dtype=int),
        titles=frame["title"].tolist(),
        dates=pd.to_datetime(frame["date"]).to_numpy(),
        tfidf_vectoriser=vectoriser,
        tfidf_matrix=matrix,
        bm25=bm25,
        bm25_tokens=token_lists,
        tokenise=tokenise,
        embeddings=dict(embeddings or {}),
        embedding_models=dict(embedding_models or {}),
    )


def _rrf(score_mats: list[np.ndarray], k: int = RRF_K) -> np.ndarray:
    """Reciprocal rank fusion: sum over methods of 1 / (k + rank), rank 1 = best."""
    fused = np.zeros_like(score_mats[0], dtype=float)
    for scores in score_mats:
        order = np.argsort(-scores, axis=1, kind="stable")
        ranks = np.empty_like(order)
        rows = np.arange(scores.shape[0])[:, None]
        ranks[rows, order] = np.arange(1, scores.shape[1] + 1)
        fused += 1.0 / (k + ranks)
    return fused


def score_rows(index: RetrievalIndex, method: str, rows: Sequence[int]) -> np.ndarray:
    """Similarity of indexed incidents `rows` (row positions) to every indexed incident: (len(rows), n)."""
    rows = np.asarray(rows, dtype=int)
    if method == "tfidf":
        return (index.tfidf_matrix[rows] @ index.tfidf_matrix.T).toarray()
    if method == "bm25":
        return np.vstack([index.bm25.get_scores(index.bm25_tokens[r]) for r in rows])
    if method in ("minilm", "mpnet"):
        emb = index.embeddings[method]
        return emb[rows] @ emb.T
    if method == "hybrid":
        return _rrf([score_rows(index, m, rows) for m in HYBRID_PARTS])
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def score_texts(
    index: RetrievalIndex,
    method: str,
    texts: Sequence[str],
    embed: Callable[[Sequence[str], str], np.ndarray] | None = None,
) -> np.ndarray:
    """Similarity of free-text queries to every indexed incident: (len(texts), n).

    Embedding methods need `embed(texts, model_name)` returning normalised vectors.
    """
    if method == "tfidf":
        return (index.tfidf_vectoriser.transform(list(texts)) @ index.tfidf_matrix.T).toarray()
    if method == "bm25":
        return np.vstack([index.bm25.get_scores(t) for t in index.tokenise(list(texts))])
    if method in ("minilm", "mpnet"):
        if embed is None:
            raise ValueError("an embedding function is needed for embedding methods")
        return embed(list(texts), index.embedding_models[method]) @ index.embeddings[method].T
    if method == "hybrid":
        return _rrf([score_texts(index, m, texts, embed) for m in HYBRID_PARTS])
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def rank_candidates(
    scores: np.ndarray,
    index: RetrievalIndex,
    exclude_ids: Sequence[int] = (),
    before: np.datetime64 | None = None,
) -> list[int]:
    """Incident ids ordered by decreasing score (ties broken by id), excluding `exclude_ids` and, with
    `before`, every incident not strictly earlier than that date."""
    keep = np.ones(len(index.incident_ids), dtype=bool)
    for i in exclude_ids:
        if int(i) in index.row_of:
            keep[index.row_of[int(i)]] = False
    if before is not None:
        keep &= index.dates < before
    rows = np.flatnonzero(keep)
    order = np.lexsort((index.incident_ids[rows], -scores[rows]))
    return [int(x) for x in index.incident_ids[rows][order]]


def search(
    index: RetrievalIndex,
    method: str,
    text: str,
    k: int = 10,
    embed: Callable[[Sequence[str], str], np.ndarray] | None = None,
    exclude_ids: Sequence[int] = (),
    before: np.datetime64 | None = None,
) -> pd.DataFrame:
    """Top-k incidents for a free-text query: incident_id, title, score, rank. With `before`, only incidents
    strictly earlier than that date are candidates (precedent search)."""
    scores = score_texts(index, method, [text], embed)[0]
    ranked = rank_candidates(scores, index, exclude_ids, before=before)[:k]
    rows = [index.row_of[i] for i in ranked]
    return pd.DataFrame(
        {
            "rank": range(1, len(ranked) + 1),
            "incident_id": ranked,
            "title": [index.titles[r] for r in rows],
            "score": scores[rows],
        }
    )


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

KS = (1, 3, 5, 10, 20, 50)


def evaluate_method(
    index: RetrievalIndex,
    method: str,
    gt: GroundTruth,
    earlier_only: bool = False,
    batch: int = 128,
) -> pd.DataFrame:
    """Per-query metrics for one method.

    Columns: query, n_relevant, first_rank (rank of the first relevant incident; NaN if none is retrievable),
    rr (reciprocal rank over the full ranking), p@5, r@10, hit@5, ndcg@10, recall@k for KS, and
    dissimilar@5 (share of the top 5 that editors marked dissimilar; NaN for queries without such links).
    With `earlier_only`, candidates and relevant incidents are restricted to those strictly earlier than the
    query, and queries without an earlier relevant incident are dropped.
    """
    queries = [q for q in gt.queries if q in index.row_of]
    rows_out = []
    for start in range(0, len(queries), batch):
        chunk = queries[start : start + batch]
        scores = score_rows(index, method, [index.row_of[q] for q in chunk])
        for q, s in zip(chunk, scores, strict=True):
            before = index.dates[index.row_of[q]] if earlier_only else None
            relevant = gt.relevant[q]
            if earlier_only:
                relevant = {r for r in relevant if index.dates[index.row_of[r]] < before}
                if not relevant:
                    continue
            ranked = rank_candidates(s, index, exclude_ids=[q], before=before)
            dissimilar = gt.dissimilar.get(q, set())
            first = next((i for i, r in enumerate(ranked, 1) if r in relevant), np.nan)
            row = {
                "query": q,
                "n_relevant": len(relevant),
                "first_rank": first,
                "rr": M.reciprocal_rank(ranked, relevant),
                "p@5": M.precision_at_k(ranked, relevant, 5),
                "r@10": M.recall_at_k(ranked, relevant, 10),
                "hit@5": M.hit_at_k(ranked, relevant, 5),
                "ndcg@10": M.ndcg_at_k(ranked, relevant, 10),
                "dissimilar@5": (sum(1 for r in ranked[:5] if r in dissimilar) / 5)
                if dissimilar
                else np.nan,
                "p@3": M.precision_at_k(ranked, relevant, 3),
                "rr@3": M.reciprocal_rank(ranked, relevant, 3),
                "hit@3": M.hit_at_k(ranked, relevant, 3),
            }
            for k in KS:
                row[f"recall@{k}"] = M.recall_at_k(ranked, relevant, k)
            rows_out.append(row)
    return pd.DataFrame(rows_out).assign(method=method)


def evaluate_suggestions(gt: GroundTruth, queries: Sequence[int]) -> pd.DataFrame:
    """AIID's own NLP suggestions (only the top 3 exist) scored like a method, at depth 3."""
    rows = []
    for q in queries:
        ranked, relevant = gt.aiid_suggestions.get(q, []), gt.relevant[q]
        rows.append(
            {
                "query": q,
                "p@3": M.precision_at_k(ranked, relevant, 3),
                "rr@3": M.reciprocal_rank(ranked, relevant, 3),
                "hit@3": M.hit_at_k(ranked, relevant, 3),
            }
        )
    return pd.DataFrame(rows).assign(method="aiid_nlp")


def oracle(gt: GroundTruth, queries: Sequence[int]) -> dict[str, float]:
    """Best achievable values given the number of relevant incidents per query (most have only one)."""
    n = np.array([len(gt.relevant[q]) for q in queries])
    return {
        "p@5": float((np.minimum(n, 5) / 5).mean()),
        "p@3": float((np.minimum(n, 3) / 3).mean()),
        "queries_with_one_relevant": float((n == 1).mean()),
    }


def summarise(
    per_query: pd.DataFrame, metrics: Sequence[str], n_boot: int = 2000, seed: int = 42
) -> pd.DataFrame:
    """Mean and bootstrap 95% CI (over queries) of each metric, per method."""
    rows = []
    for method, sub in per_query.groupby("method", sort=False):
        row: dict[str, Any] = {"method": method, "queries": len(sub)}
        for m in metrics:
            if m not in sub or sub[m].isna().all():
                continue
            values = sub[m].dropna().to_numpy()
            lo, hi = M.bootstrap_ci(values, n_boot, seed)
            row.update({m: float(values.mean()), f"{m}_low": lo, f"{m}_high": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(
    per_query: pd.DataFrame, baseline: str, metric: str = "rr", n_boot: int = 2000, seed: int = 42
) -> pd.DataFrame:
    """Each method vs `baseline` on the same queries: paired bootstrap of the mean difference, Wilcoxon
    signed-rank p-value, Holm adjustment across the comparisons."""
    from scipy.stats import wilcoxon

    wide = per_query.pivot(index="query", columns="method", values=metric)
    rows = []
    for method in wide.columns:
        if method == baseline:
            continue
        a, b = wide[method].to_numpy(), wide[baseline].to_numpy()
        boot = M.paired_bootstrap_mean_diff(a, b, n_boot, seed)
        diff = a - b
        p_w = float(wilcoxon(a, b).pvalue) if np.any(diff != 0) else 1.0
        rows.append(
            {
                "method": method,
                "baseline": baseline,
                "metric": metric,
                **boot,
                "wilcoxon_p": p_w,
                "better": int((diff > 0).sum()),
                "worse": int((diff < 0).sum()),
                "ties": int((diff == 0).sum()),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = M.holm_adjust(out["wilcoxon_p"].tolist())
    return out


def fingerprint(frame: pd.DataFrame) -> str:
    """Cache key for anything computed on this collection."""
    return cache.hash_parts(frame["incident_id"].tolist(), cache.hash_texts(frame["text"].tolist()))
