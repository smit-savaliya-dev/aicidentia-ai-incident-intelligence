"""Streamlit helpers shared by the pages: page setup, cached loaders, sidebar status, banners."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from aicidentia.dashboard import data, models

PAGE_ICON = "🛰️"


@st.cache_data(show_spinner=False)
def cached_incidents() -> pd.DataFrame:
    return data.incidents()


@st.cache_data(show_spinner=False)
def cached_table(name: str) -> pd.DataFrame:
    return data.table(name)


@st.cache_data(show_spinner=False)
def cached_network() -> dict[str, Any]:
    return data.network()


@st.cache_data(show_spinner=False)
def cached_theme_info() -> dict[str, Any]:
    return data.theme_names()[1]


@st.cache_resource(show_spinner="Loading the risk-domain classifier ...")
def cached_classifier() -> models.ActiveClassifier | None:
    return models.load_classifier()


@st.cache_resource(show_spinner="Building the search index ...")
def cached_retriever() -> models.ActiveRetriever:
    return models.load_retriever()


def setup(title: str, description: str = "") -> None:
    """Page config, title and the sidebar with snapshot information, banners and a reload button."""
    info = data.snapshot_info()
    st.set_page_config(page_title=f"{title} - {info['name']}", page_icon=PAGE_ICON, layout="wide")
    st.title(info["name"])
    st.header(title)
    if description:
        st.caption(description)
    with st.sidebar:
        st.markdown(f"**AIID snapshot** {info['snapshot']}  \n**Random seed** {info['seed']}")
        if cached_theme_info().get("provisional"):
            st.info("Theme names are PROVISIONAL until the topic labels are confirmed.")
        if st.button("Reload data after a refresh"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"
