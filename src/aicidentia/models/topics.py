"""RQ1 - topic models: LDA vs NMF vs BERTopic."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def fit_lda(tokens: Sequence[Sequence[str]], n_topics: int, seed: int | None = None):
    """Fit a gensim LDA model."""
    raise NotImplementedError


def fit_nmf(texts: Sequence[str], n_topics: int, seed: int | None = None):
    """Fit scikit-learn NMF on TF-IDF features."""
    raise NotImplementedError


def fit_bertopic(texts: Sequence[str], embeddings=None, seed: int | None = None):
    """Fit BERTopic (UMAP + scikit-learn HDBSCAN + c-TF-IDF)."""
    raise NotImplementedError


def topic_coherence(model, tokens: Sequence[Sequence[str]], coherence: str = "c_v") -> float:
    """Compute topic coherence for a fitted model."""
    raise NotImplementedError


def topic_diversity(topic_words: Sequence[Sequence[str]], top_n: int = 10) -> float:
    """Proportion of unique words among the top-N words of all topics."""
    raise NotImplementedError


def assign_topics(model, texts: Sequence[str]) -> pd.DataFrame:
    """Return the dominant topic and probability for each document."""
    raise NotImplementedError
