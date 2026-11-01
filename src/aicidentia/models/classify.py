"""RQ3 - classify incidents into risk domains and audit AI-generated labels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def train_tfidf_classifier(texts: Sequence[str], labels: Sequence[str], kind: str = "logreg"):
    """Fit TF-IDF + logistic regression ('logreg') or linear SVM ('svm')."""
    raise NotImplementedError


def train_embedding_classifier(embeddings: np.ndarray, labels: Sequence[str], kind: str = "logreg"):
    """Fit a classifier on frozen sentence embeddings."""
    raise NotImplementedError


def fine_tune_distilbert(texts: Sequence[str], labels: Sequence[str], output_dir: str, **kwargs):
    """Fine-tune distilbert-base-uncased (local GPU or Google Colab)."""
    raise NotImplementedError


def cross_validate_models(models: dict, X, y, n_splits: int = 5, seed: int | None = None):
    """Stratified cross-validation of several models; return a results dataframe."""
    raise NotImplementedError


def audit_labels(ai_labels: Sequence[str], hand_labels: Sequence[str]) -> dict:
    """Compare AI-generated labels with the 150 hand labels (Cohen's kappa, confusion matrix)."""
    raise NotImplementedError


def predict_domains(model, texts: Sequence[str]) -> pd.DataFrame:
    """Predict risk domain and confidence for each text."""
    raise NotImplementedError
