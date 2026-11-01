"""Smoke tests: key packages import, config loads, the app pages render without data."""

import importlib
from pathlib import Path

import pytest

from aicidentia.config import PROJECT_ROOT, load_config, set_seeds

PACKAGES = [
    "aicidentia",
    "pandas",
    "numpy",
    "pyarrow",
    "openpyxl",
    "yaml",
    "tqdm",
    "dotenv",
    "sklearn",
    "gensim",
    "bertopic",
    "sentence_transformers",
    "umap",
    "nltk",
    "rank_bm25",
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "networkx",
    "igraph",
    "leidenalg",
    "pyvis",
    "scipy",
    "statsmodels",
    "pymannkendall",
    "plotly",
    "matplotlib",
    "seaborn",
    "streamlit",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name):
    importlib.import_module(name)


def test_sklearn_hdbscan_available():
    from sklearn.cluster import HDBSCAN  # noqa: F401


def test_config_loads():
    cfg = load_config()
    assert cfg["project_name"] == "AICIDENTIA"
    assert cfg["random_seed"] == 42
    assert {
        "sentence_embedding_small",
        "sentence_embedding_large",
        "transformer_classifier",
    } <= set(cfg["models"])
    assert (PROJECT_ROOT / cfg["paths"]["raw"]).is_dir()


def test_set_seeds_is_reproducible():
    import numpy as np

    set_seeds(42)
    a = np.random.rand(3)
    set_seeds(42)
    b = np.random.rand(3)
    assert (a == b).all()


APP_DIR = PROJECT_ROOT / "app"
APP_FILES = [APP_DIR / "Home.py", *sorted((APP_DIR / "pages").glob("*.py"))]


@pytest.mark.parametrize("path", APP_FILES, ids=lambda p: Path(p).name)
def test_app_pages_run_without_data(path):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(path), default_timeout=30).run()
    assert not at.exception
    assert any("AICIDENTIA Intelligence" in t.value for t in at.title)
