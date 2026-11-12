"""Results registry: one JSON file (reports/results.json) that the dashboard and report read.

Every number that is cited anywhere is recorded here with the file and function that produced
it, so figures in the dashboard and the report can never drift from the analysis.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aicidentia.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

RESULTS_PATH = PROJECT_ROOT / "reports" / "results.json"


def _to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars and containers to plain JSON types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, pd.Series):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, pd.DataFrame):
        return [_to_jsonable(r) for r in value.to_dict(orient="records")]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_results(path: Path = RESULTS_PATH) -> dict[str, dict[str, Any]]:
    """Read the whole registry (empty dict if it does not exist yet)."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def record(
    key: str,
    value: Any,
    source: str,
    function: str,
    path: Path = RESULTS_PATH,
) -> None:
    """Insert or overwrite one entry: key -> {value, source_file, function}.

    `source` is the repo-relative file the value was derived from (or the module that
    produced it); `function` is the dotted name of the producing function.
    """
    results = load_results(path)
    results[key] = {"value": _to_jsonable(value), "source_file": source, "function": function}
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(sorted(results.items())), indent=2, ensure_ascii=False) + "\n"
    # Write atomically so an interrupted run cannot leave a truncated registry.
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload)
    os.replace(tmp, path)
    logger.debug("Recorded %s", key)


def remove(keys: list[str], path: Path = RESULTS_PATH) -> list[str]:
    """Delete entries that an analysis no longer produces; returns the keys actually removed."""
    results = load_results(path)
    gone = [k for k in keys if k in results]
    if gone:
        for k in gone:
            del results[k]
        payload = json.dumps(dict(sorted(results.items())), indent=2, ensure_ascii=False) + "\n"
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    return gone


def get(key: str, path: Path = RESULTS_PATH) -> Any:
    """Return the value stored under `key` (KeyError if missing)."""
    return load_results(path)[key]["value"]
