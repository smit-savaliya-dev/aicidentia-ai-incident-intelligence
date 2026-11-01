"""RQ4 (stretch) - similar-incident retrieval: TF-IDF vs BM25 vs sentence embeddings."""

from __future__ import annotations

from collections.abc import Sequence


def build_tfidf_index(texts: Sequence[str]):
    """Index incident texts with TF-IDF."""
    raise NotImplementedError


def build_bm25_index(tokens: Sequence[Sequence[str]]):
    """Index tokenised incident texts with BM25 (rank-bm25)."""
    raise NotImplementedError


def build_embedding_index(embeddings):
    """Index sentence embeddings for cosine-similarity search."""
    raise NotImplementedError


def retrieve_similar(index, query, k: int = 5) -> list[int]:
    """Return the ids of the top-k most similar incidents for `query`."""
    raise NotImplementedError


def evaluate_retrieval(retrieved: dict[int, list[int]], relevant: dict[int, set[int]]) -> dict:
    """Compute precision@5 and MRR against editor-marked similar incidents."""
    raise NotImplementedError
