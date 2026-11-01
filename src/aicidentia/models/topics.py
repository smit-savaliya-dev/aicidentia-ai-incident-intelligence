"""RQ1 - topic models: LDA vs NMF vs BERTopic, with coherence, diversity and stability metrics.

All three families are fitted on the same `Corpus` (see `aicidentia.features.text`) and return a
`TopicFit`, so they can be scored identically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from aicidentia.features.text import Corpus, build_tfidf, topic_settings

logger = logging.getLogger(__name__)


@dataclass
class TopicFit:
    """Result of fitting one topic model.

    `terms[i]` are the top words of topic i; `assignments[j]` is the dominant topic index of document
    j (-1 if the document has no tokens); `weights` is the (documents x topics) matrix of
    row-normalised topic weights for LDA/NMF (None for BERTopic, which gives hard assignments);
    `topic_ids` maps topic index to the model's own topic id.
    """

    family: str
    params: dict[str, Any]
    seed: int
    terms: list[list[str]]
    assignments: np.ndarray
    weights: np.ndarray | None
    topic_ids: list[int] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_topics(self) -> int:
        return len(self.terms)


def _dominant(weights: np.ndarray, corpus: Corpus) -> np.ndarray:
    assignments = weights.argmax(axis=1)
    empty = np.array([len(t) == 0 for t in corpus.tokens])
    assignments[empty] = -1
    return assignments


def fit_lda(corpus: Corpus, n_topics: int, seed: int, **kwargs: Any) -> TopicFit:
    """Fit a gensim LDA model (batch variational Bayes, learned alpha/eta) on the shared bag-of-words."""
    from gensim.models import LdaModel

    cfg = {**topic_settings()["lda"], **kwargs}
    top_n = topic_settings()["top_n"]
    model = LdaModel(
        corpus.bow,
        id2word=corpus.dictionary,
        num_topics=n_topics,
        random_state=seed,
        passes=cfg["passes"],
        iterations=cfg["iterations"],
        alpha="auto",
        eta="auto",
        chunksize=len(corpus.bow),
        update_every=0,
        eval_every=None,
    )
    terms = [[w for w, _ in model.show_topic(t, topn=top_n)] for t in range(n_topics)]
    weights = np.zeros((len(corpus.bow), n_topics))
    for row, doc_topics in enumerate(
        model.get_document_topics(corpus.bow, minimum_probability=0.0)
    ):
        for topic, prob in doc_topics:
            weights[row, topic] = prob
    return TopicFit(
        "LDA",
        {"k": n_topics, **cfg},
        seed,
        terms,
        _dominant(weights, corpus),
        weights,
        list(range(n_topics)),
    )


def fit_nmf(corpus: Corpus, n_topics: int, seed: int, **kwargs: Any) -> TopicFit:
    """Fit scikit-learn NMF on TF-IDF (random init so that seeds differ; coordinate descent)."""
    from sklearn.decomposition import NMF

    cfg = {**topic_settings()["nmf"], **kwargs}
    top_n = topic_settings()["top_n"]
    matrix, _ = build_tfidf(corpus.tokens, corpus.vocabulary)
    model = NMF(n_components=n_topics, init="random", random_state=seed, max_iter=cfg["max_iter"])
    w = model.fit_transform(matrix)
    sums = w.sum(axis=1, keepdims=True)
    weights = np.divide(w, sums, out=np.zeros_like(w), where=sums > 0)
    vocab = np.array(corpus.vocabulary)
    terms = [list(vocab[np.argsort(comp)[::-1][:top_n]]) for comp in model.components_]
    return TopicFit(
        "NMF",
        {"k": n_topics, **cfg},
        seed,
        terms,
        _dominant(weights, corpus),
        weights,
        list(range(n_topics)),
        {"reconstruction_err": float(model.reconstruction_err_), "n_iter": int(model.n_iter_)},
    )


def fit_bertopic(
    corpus: Corpus,
    embeddings: np.ndarray,
    seed: int,
    min_cluster_size: int,
    embedding_name: str = "",
    **kwargs: Any,
) -> TopicFit:
    """Fit BERTopic: seeded UMAP, scikit-learn HDBSCAN, c-TF-IDF over the shared vocabulary, outlier reduction.

    `embeddings` come from the untouched incident text; the c-TF-IDF step sees the shared lemmatised
    tokens, so topic words come from the same vocabulary as LDA and NMF. Outliers are re-assigned to
    the nearest topic by embedding similarity; the share of outliers *before* reduction is kept in
    `extra['outlier_share_before']`.
    """
    from bertopic import BERTopic
    from sklearn.cluster import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    logging.getLogger("BERTopic").setLevel(
        logging.ERROR
    )  # silence the per-fit 'custom topics' warning
    cfg = {**topic_settings()["bertopic"], **kwargs}
    top_n = topic_settings()["top_n"]
    umap_model = UMAP(random_state=seed, **cfg["umap"])
    cluster_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=cfg["min_samples"])
    vectoriser = CountVectorizer(
        tokenizer=str.split, token_pattern=None, lowercase=False, vocabulary=list(corpus.vocabulary)
    )
    model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectoriser,
        calculate_probabilities=False,
        top_n_words=top_n,
        verbose=False,
    )
    topics, _ = model.fit_transform(corpus.docs, embeddings=embeddings)
    topics = np.asarray(topics)
    extra: dict[str, Any] = {
        "outlier_share_before": float((topics == -1).mean()),
        "n_topics_before_reduction": int(len(set(topics.tolist()) - {-1})),
    }
    if len(set(topics.tolist()) - {-1}) == 0:
        logger.warning("BERTopic found no clusters (min_cluster_size=%d)", min_cluster_size)
        return TopicFit(
            "BERTopic",
            {"min_cluster_size": min_cluster_size, "embedding": embedding_name},
            seed,
            [],
            np.full(len(topics), -1),
            None,
            [],
            extra,
        )
    if (topics == -1).any():
        topics = np.asarray(
            model.reduce_outliers(
                corpus.docs,
                topics.tolist(),
                strategy=cfg["outlier_strategy"],
                embeddings=embeddings,
            )
        )
        model.update_topics(
            corpus.docs, topics=topics.tolist(), vectorizer_model=vectoriser, top_n_words=top_n
        )
    ids = sorted(set(topics.tolist()) - {-1})
    terms = [[w for w, _ in model.get_topic(t)][:top_n] for t in ids]
    index = {t: i for i, t in enumerate(ids)}
    assignments = np.array([index.get(t, -1) for t in topics])
    params = {
        "min_cluster_size": min_cluster_size,
        "embedding": embedding_name,
        **{k: cfg[k] for k in ("min_samples", "umap", "outlier_strategy")},
    }
    return TopicFit("BERTopic", params, seed, terms, assignments, None, ids, extra)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def topic_coherence(terms: list[list[str]], corpus: Corpus, coherence: str = "c_v") -> float:
    """gensim CoherenceModel score of `terms` on the shared tokenised corpus ('c_v' or 'c_npmi')."""
    from gensim.models.coherencemodel import CoherenceModel

    if not terms:
        return float("nan")
    topn = min(len(t) for t in terms)
    model = CoherenceModel(
        topics=[t[:topn] for t in terms],
        texts=corpus.tokens,
        dictionary=corpus.dictionary,
        coherence=coherence,
        topn=topn,
        processes=1,
    )
    return float(model.get_coherence())


def topic_diversity(terms: list[list[str]], top_n: int | None = None) -> float:
    """Topic diversity (Dieng et al. 2020): unique words / total words in the top-N words of all topics."""
    if not terms:
        return float("nan")
    top_n = top_n or min(len(t) for t in terms)
    words = [w for t in terms for w in t[:top_n]]
    return len(set(words)) / len(words)


def matched_jaccard(terms_a: list[list[str]], terms_b: list[list[str]]) -> float:
    """Similarity of two topic sets: optimal one-to-one matching of topics by Jaccard overlap of their
    top words; the summed Jaccard is divided by the larger topic count (so different counts are penalised)."""
    if not terms_a or not terms_b:
        return 0.0
    sets_a, sets_b = [set(t) for t in terms_a], [set(t) for t in terms_b]
    sim = np.array([[len(a & b) / len(a | b) for b in sets_b] for a in sets_a])
    rows, cols = linear_sum_assignment(-sim)
    return float(sim[rows, cols].sum() / max(len(sets_a), len(sets_b)))


def stability(fits: list[TopicFit]) -> float:
    """Mean pairwise `matched_jaccard` between fits of the same configuration under different seeds."""
    pairs = list(combinations(fits, 2))
    if not pairs:
        return float("nan")
    return float(np.mean([matched_jaccard(a.terms, b.terms) for a, b in pairs]))


# --------------------------------------------------------------------------------------
# Assignments and representatives
# --------------------------------------------------------------------------------------


def assign_topics(fit: TopicFit, corpus: Corpus) -> pd.DataFrame:
    """Dominant topic per incident: incident_id, topic (index), topic_id (model id), weight (LDA/NMF only)."""
    weight = (
        fit.weights[np.arange(len(fit.assignments)), np.clip(fit.assignments, 0, None)]
        if fit.weights is not None
        else np.full(len(fit.assignments), np.nan)
    )
    weight = np.where(fit.assignments >= 0, weight, np.nan)
    ids = [fit.topic_ids[a] if a >= 0 else -1 for a in fit.assignments]
    return pd.DataFrame(
        {
            "incident_id": corpus.incident_ids,
            "topic": fit.assignments,
            "topic_id": ids,
            "weight": weight,
        }
    )


def representative_docs(
    fit: TopicFit, embeddings: np.ndarray | None = None, n: int = 5
) -> dict[int, list[int]]:
    """Row indices of the `n` most representative documents per topic index.

    LDA/NMF: highest topic weight among documents assigned to the topic. BERTopic: highest cosine
    similarity to the mean embedding of the topic's documents.
    """
    out: dict[int, list[int]] = {}
    for topic in range(fit.n_topics):
        members = np.flatnonzero(fit.assignments == topic)
        if len(members) == 0:
            out[topic] = []
            continue
        if fit.weights is not None:
            score = fit.weights[members, topic]
        elif embeddings is not None:
            centroid = embeddings[members].mean(axis=0)
            centroid /= np.linalg.norm(centroid) or 1.0
            score = embeddings[members] @ centroid
        else:
            score = np.zeros(len(members))
        out[topic] = [int(members[i]) for i in np.argsort(-score, kind="stable")[:n]]
    return out
