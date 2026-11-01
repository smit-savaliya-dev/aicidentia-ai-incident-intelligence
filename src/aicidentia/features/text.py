"""Text preprocessing and feature extraction (TF-IDF, BoW, sentence embeddings)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def preprocess_text(texts: Sequence[str]) -> list[str]:
    """Lower-case, strip boilerplate, remove stop-words and lemmatise."""
    raise NotImplementedError


def tokenize(texts: Sequence[str]) -> list[list[str]]:
    """Tokenise texts into lists of tokens (used by gensim LDA and BM25)."""
    raise NotImplementedError


def build_tfidf(texts: Sequence[str], **kwargs):
    """Fit a TF-IDF vectoriser; return (matrix, fitted vectoriser)."""
    raise NotImplementedError


def build_bow_corpus(tokens: Sequence[Sequence[str]]):
    """Build a gensim dictionary and bag-of-words corpus from tokenised texts."""
    raise NotImplementedError


def embed_sentences(texts: Sequence[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """Encode texts with a sentence-transformers model; cache results on disk."""
    raise NotImplementedError
