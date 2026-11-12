"""Load the deployed models for the dashboard, offline-first, with automatic fallbacks.

Classifier: the deployed mpnet + LR bundle needs the all-mpnet-base-v2 model from the local Hugging Face cache
(HF_HOME, see .env). If that model cannot be loaded without the network, the TF-IDF + LR fallback bundle is used.
Retrieval: the deployed method (Phase 7) if its inputs are available, otherwise BM25.
Nothing here ever downloads a model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aicidentia import registry
from aicidentia.features.text import embed_sentences, load_sentence_model
from aicidentia.models import classify as C
from aicidentia.models import retrieval as R

logger = logging.getLogger(__name__)


@dataclass
class ActiveClassifier:
    """The classifier the dashboard is using and why."""

    name: str
    bundle: dict[str, Any]
    is_fallback: bool
    reason: str
    calibration: str | None
    embedding_model: str | None


def load_classifier(
    main_path: Path = C.MODEL_PATH, fallback_path: Path = C.FALLBACK_MODEL_PATH, loader: Any = None
) -> ActiveClassifier | None:
    """Deployed classifier if usable offline, else the TF-IDF + LR fallback, else None (with no model files).

    `loader(model_name)` loads the embedding model (default: from the local cache only); tests replace it.
    """
    loader = loader or (lambda name: load_sentence_model(name, local_files_only=True))
    problem = ""
    if main_path.exists():
        bundle = joblib.load(main_path)
        try:
            if bundle.get("embedding_model"):
                loader(bundle["embedding_model"])
            return ActiveClassifier(
                bundle["spec"].name,
                bundle,
                False,
                "deployed model (Phase 6), embedding model loaded from the local cache",
                bundle.get("calibration"),
                bundle.get("embedding_model"),
            )
        except Exception as exc:  # missing or corrupt cache, offline: use the fallback
            problem = f"{bundle.get('embedding_model')} could not be loaded offline ({type(exc).__name__}: {str(exc)[:160]})"
            logger.warning("Classifier fallback: %s", problem)
    else:
        problem = f"{main_path.name} not found (run scripts/run_phase6_classification.py)"
    if fallback_path.exists():
        bundle = joblib.load(fallback_path)
        return ActiveClassifier(
            bundle["spec"].name,
            bundle,
            True,
            f"fallback: {problem}",
            bundle.get("calibration"),
            None,
        )
    return None


def classify_text(active: ActiveClassifier, text: str) -> pd.DataFrame:
    """Predicted domain with calibrated confidence and every class probability for one text."""
    return C.predict_domains(active.bundle, [text], use_cache=False, local_files_only=True)


@dataclass
class ActiveRetriever:
    """The retrieval index, the method in use, and what is unavailable."""

    index: R.RetrievalIndex
    method: str
    deployed: str
    is_fallback: bool
    reason: str
    missing: list[str] = field(default_factory=list)
    query_models_ok: dict[str, bool] = field(default_factory=dict)


def query_embedder(texts: list[str], model_name: str) -> np.ndarray:
    """Embed free-text queries with a cached model, never downloading and never writing the disk cache."""
    return embed_sentences(texts, model_name, use_cache=False, local_files_only=True)


def load_retriever(loader: Any = None) -> ActiveRetriever:
    """Index with every available method; the Phase 7 deployed method unless it cannot be used offline."""
    from aicidentia.models.retrieval_outputs import BASELINE, build_default_index

    loader = loader or (lambda name: load_sentence_model(name, local_files_only=True))
    index, missing = build_default_index(local_files_only=True)
    ok = {}
    for method, name in index.embedding_models.items():
        try:
            loader(name)
            ok[method] = True
        except Exception as exc:
            logger.warning("Query encoder for %s unavailable: %s", method, exc)
            ok[method] = False
    usable = [
        m
        for m in index.available_methods
        if m in R.LEXICAL or (m == "hybrid" and ok.get("mpnet")) or ok.get(m)
    ]
    try:
        deployed = str(registry.get("rq4.deployed"))
    except KeyError:
        deployed = BASELINE
    if deployed in usable:
        return ActiveRetriever(
            index, deployed, deployed, False, "deployed method (Phase 7)", missing, ok
        )
    return ActiveRetriever(
        index, BASELINE, deployed, True, f"fallback: {deployed} is not usable offline", missing, ok
    )


def usable_methods(active: ActiveRetriever) -> list[str]:
    """Methods that can answer a free-text query right now."""
    ok = active.query_models_ok
    return [
        m
        for m in active.index.available_methods
        if m in R.LEXICAL or (m == "hybrid" and ok.get("mpnet")) or ok.get(m)
    ]
