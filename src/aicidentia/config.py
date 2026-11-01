"""Project configuration and global reproducibility seeds.

`load_config()` reads ``config/config.yaml``; `set_seeds()` fixes the random state of
``random``, NumPy and (if installed) PyTorch so every run is repeatable.
"""

from __future__ import annotations

import logging
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# src/aicidentia/config.py -> parents[2] is the project root (works for editable installs).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=1)
def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config. Override the location with `path` or $AICIDENTIA_CONFIG."""
    cfg_path = Path(path or os.environ.get("AICIDENTIA_CONFIG") or DEFAULT_CONFIG_PATH)
    # Return a copy so callers cannot mutate the cached config.
    return dict(_read_config(cfg_path.resolve()))


def resolve_path(key: str, cfg: dict[str, Any] | None = None) -> Path:
    """Absolute path for an entry of ``cfg['paths']`` (e.g. ``resolve_path('raw')``)."""
    cfg = cfg or load_config()
    return PROJECT_ROOT / cfg["paths"][key]


def configure_logging(level: int = logging.INFO) -> None:
    """Send package logs to stderr (call once from a script or notebook; the package never prints).

    Chatty third-party loggers (gensim's per-batch INFO lines, BERTopic) are held at WARNING.
    """
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True
    )
    for noisy in ("gensim", "BERTopic", "numba", "matplotlib", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


def set_seeds(seed: int | None = None) -> int:
    """Seed ``random``, NumPy and PyTorch (CPU and CUDA). Returns the seed used."""
    if seed is None:
        seed = int(load_config()["random_seed"])
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # torch is an optional extra
        return seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed
