"""Shared evaluation metrics used across research questions."""

from __future__ import annotations

from collections.abc import Sequence


def classification_report_table(y_true: Sequence, y_pred: Sequence):
    """Per-class precision, recall, F1 as a dataframe, with macro/weighted averages."""
    raise NotImplementedError


def cohens_kappa(labels_a: Sequence, labels_b: Sequence, weights: str | None = None) -> float:
    """Cohen's kappa between two annotators / label sources."""
    raise NotImplementedError


def precision_at_k(retrieved: Sequence[int], relevant: set[int], k: int = 5) -> float:
    """Precision@k for one query."""
    raise NotImplementedError


def mean_reciprocal_rank(retrieved: Sequence[Sequence[int]], relevant: Sequence[set[int]]) -> float:
    """Mean reciprocal rank over queries."""
    raise NotImplementedError


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, seed: int | None = None):
    """Bootstrap confidence interval for a metric."""
    raise NotImplementedError
