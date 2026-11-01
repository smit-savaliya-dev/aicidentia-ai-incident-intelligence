"""Disk cache for expensive steps (embeddings, model fits) in data/interim/.

A cache entry is keyed by a SHA-256 hash of everything the result depends on (input fingerprint,
parameters, a code-version string), so changing any of them gives a new entry and reruns are fast.
Bump the `version` argument when the computing code changes in a way that alters results.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import joblib

from aicidentia.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)


def hash_parts(*parts: Any) -> str:
    """Stable SHA-256 of JSON-serialisable parts (dict keys sorted; unknown objects via str)."""
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_texts(texts: Iterable[str]) -> str:
    """Fingerprint of a list of strings (order matters)."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def cache_dir() -> Path:
    """data/interim/ (created if missing)."""
    path = PROJECT_ROOT / load_config()["paths"]["interim"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def cached[T](
    name: str,
    parts: tuple[Any, ...],
    compute: Callable[[], T],
    version: str = "1",
    force: bool = False,
) -> T:
    """Return the cached result for (name, parts, version), computing and storing it on a miss."""
    key = hash_parts(name, parts, version)[:20]
    path = cache_dir() / f"{name}-{key}.joblib"
    if path.exists() and not force:
        logger.debug("cache hit %s", path.name)
        return joblib.load(path)
    logger.info("cache miss %s: computing", path.name)
    result = compute()
    joblib.dump(result, path, compress=3)
    return result
