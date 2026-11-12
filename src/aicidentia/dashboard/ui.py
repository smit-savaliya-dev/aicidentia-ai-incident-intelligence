"""Streamlit helpers shared by every page: page setup with header and footer, global sidebar filters, KPI and
finding cards, cached loaders and CSV downloads. The look comes from .streamlit/config.toml plus a little CSS."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from aicidentia.dashboard import data, models

PAGE_ICON = "🛰️"
ACCENT = "#0072B2"

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1350px;}
.aic-header {border-bottom: 3px solid #0072B2; padding-bottom: .4rem; margin-bottom: 1.0rem;}
.aic-brand {font-size: 1.05rem; font-weight: 700; letter-spacing: .08em; color: #0072B2; text-transform: uppercase;}
.aic-tagline {font-size: .9rem; color: #5B6675; margin-left: .6rem;}
.aic-title {font-size: 2.0rem; font-weight: 700; margin: .1rem 0 .1rem 0; color: #1B2430;}
.aic-subtitle {font-size: 1.02rem; color: #3D4756; margin: 0;}
.aic-kpi {background: #F2F5F9; border-radius: .6rem; padding: .85rem 1rem; border-left: 5px solid #0072B2; height: 100%;}
.aic-kpi-label {font-size: .82rem; color: #5B6675; text-transform: uppercase; letter-spacing: .04em;}
.aic-kpi-value {font-size: 2.1rem; font-weight: 700; color: #1B2430; line-height: 1.15;}
.aic-kpi-note {font-size: .82rem; color: #3D4756;}
.aic-card {background: #FFFFFF; border: 1px solid #DCE3EC; border-radius: .7rem; padding: 1rem 1.1rem; height: 100%;}
.aic-card-num {font-size: 1.9rem; font-weight: 700; color: #0072B2; line-height: 1.1;}
.aic-card-title {font-size: 1.05rem; font-weight: 650; color: #1B2430; margin: .3rem 0;}
.aic-card-text {font-size: .9rem; color: #3D4756;}
.aic-card-limit {font-size: .78rem; color: #6B7684; margin-top: .45rem;}
.aic-badge {display: inline-block; background: #E69F00; color: #1B2430; border-radius: .3rem; padding: 0 .4rem; font-size: .75rem; font-weight: 700;}
.aic-footer {border-top: 1px solid #DCE3EC; margin-top: 2.2rem; padding-top: .6rem; font-size: .82rem; color: #6B7684;}
</style>
"""


# --------------------------------------------------------------------------------------
# Cached loaders (everything comes from precomputed artefacts)
# --------------------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def cached_incidents() -> pd.DataFrame:
    return data.incidents()


@st.cache_data(show_spinner=False)
def cached_table(name: str) -> pd.DataFrame:
    return data.table(name)


@st.cache_data(show_spinner=False)
def cached_entities() -> pd.DataFrame:
    return data.entity_index()


@st.cache_data(show_spinner=False)
def cached_links() -> pd.DataFrame:
    return data.incident_entities()


@st.cache_data(show_spinner=False)
def cached_theme_info() -> dict[str, Any]:
    return data.theme_names()[1]


@st.cache_data(show_spinner=False)
def cached_network() -> dict[str, Any]:
    return data.network()


@st.cache_resource(show_spinner="Loading the risk-domain classifier from the local cache ...")
def cached_classifier() -> models.ActiveClassifier | None:
    return models.load_classifier()


@st.cache_resource(show_spinner="Building the precedent search index ...")
def cached_retriever() -> models.ActiveRetriever:
    return models.load_retriever()


# --------------------------------------------------------------------------------------
# Page frame
# --------------------------------------------------------------------------------------


def setup(title: str, subtitle: str = "", filters: bool = True) -> data.Filters | None:
    """Page config, CSS, header (name + tagline, page title, one-line purpose) and the sidebar.

    Returns the global filters (stored in session state, so they apply on every page)."""
    info = data.snapshot_info()
    st.set_page_config(
        page_title=f"{title} · {info['name']}",
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        f"<div class='aic-header'><span class='aic-brand'>{info['name']}</span>"
        f"<span class='aic-tagline'>{info['tagline']}</span>"
        f"<div class='aic-title'>{title}</div>"
        + (f"<p class='aic-subtitle'>{subtitle}</p>" if subtitle else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    flt = sidebar_filters() if filters else None
    with st.sidebar:
        st.divider()
        st.caption(f"AIID snapshot {info['snapshot_iso']} · seed {info['seed']} · works offline")
        if st.button("Reload data after a refresh", width="stretch"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    return flt


def sidebar_filters() -> data.Filters:
    """Year range, MIT risk domain and CSETv1 sector, kept in session state across pages."""
    inc = cached_incidents()
    opt = data.filter_options(inc)
    ss = st.session_state
    ss.setdefault("g_years", opt["default_years"])
    ss.setdefault("g_domains", [])
    ss.setdefault("g_sectors", [])
    with st.sidebar:
        st.markdown("### Filters")
        st.caption("Apply on every page that lists incidents.")
        years = st.slider("Year the harm occurred", *opt["years"], value=tuple(ss.g_years))
        domains = st.multiselect(
            "MIT risk domain", opt["domains"], default=ss.g_domains, placeholder="All domains"
        )
        sectors = st.multiselect(
            "Sector (CSETv1)",
            opt["sectors"],
            default=ss.g_sectors,
            placeholder="All sectors",
            help=f"Only {opt['sector_coverage']} incidents have a CSETv1 sector; choosing one keeps only those.",
        )
        if st.button("Clear filters", width="stretch"):
            ss.g_years, ss.g_domains, ss.g_sectors = opt["default_years"], [], []
            st.rerun()
    ss.g_years, ss.g_domains, ss.g_sectors = years, domains, sectors
    return data.Filters((int(years[0]), int(years[1])), tuple(domains), tuple(sectors))


def filter_note(flt: data.Filters, n: int, total: int) -> None:
    text = f"Showing **{n:,}** of {total:,} incidents ({flt.describe()})."
    if flt.sectors:
        text += " Sector filters keep only CSETv1-annotated incidents (2023 or earlier)."
    st.caption(text)


def footer() -> None:
    info = data.snapshot_info()
    st.markdown(
        f"<div class='aic-footer'>Source: AIID snapshot {info['snapshot_iso']} · reported incidents only · "
        "organisations are alleged developers or deployers · MIT risk labels are machine-generated</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------


def kpi(column: Any, label: str, value: str, note: str = "") -> None:
    column.markdown(
        f"<div class='aic-kpi'><div class='aic-kpi-label'>{label}</div>"
        f"<div class='aic-kpi-value'>{value}</div><div class='aic-kpi-note'>{note}</div></div>",
        unsafe_allow_html=True,
    )


def finding_card(column: Any, finding: dict[str, Any], page: str | None) -> None:
    with column:
        st.markdown(
            f"<div class='aic-card'><div class='aic-card-num'>{finding['number']}</div>"
            f"<div class='aic-card-title'>{finding['title']}</div>"
            f"<div class='aic-card-text'>{finding['sentence']}</div>"
            f"<div class='aic-card-limit'>Limitation: {finding['limitation']}</div></div>",
            unsafe_allow_html=True,
        )
        if page:
            st.page_link(page, label=f"Open {finding['page']}", icon=":material/arrow_forward:")


def download(frame: pd.DataFrame, name: str, label: str = "Download CSV") -> None:
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8"),
        file_name=f"aicidentia_{name}.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def incident_table(frame: pd.DataFrame, height: int = 360, key: str | None = None) -> None:
    """Incidents with a clickable AIID link."""
    show = frame.copy()
    if "date" in show:
        show["date"] = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        height=height,
        key=key,
        column_config={
            "url": st.column_config.LinkColumn("AIID", display_text="open ↗", width="small"),
            "incident_id": st.column_config.NumberColumn("ID", format="%d", width="small"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "date": st.column_config.TextColumn("Date", width="small"),
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "mit_domain": "MIT risk domain",
            "cset_tangible_harm": "CSET tangible harm",
            "given_label": "Given MIT label",
            "suggested_label": "Text suggests",
            "self_confidence": st.column_config.NumberColumn("Label quality", format="%.3f"),
        },
    )


def fmt_pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}%"
