"""AICIDENTIA Intelligence - Streamlit entry point.

Run with:  uv run streamlit run app/Home.py
"""

import streamlit as st

from aicidentia.config import load_config

cfg = load_config()

st.set_page_config(page_title=cfg["dashboard_name"], page_icon="🛰️", layout="wide")

st.title("AICIDENTIA Intelligence")
st.subheader(cfg["tagline"])
st.caption(cfg["full_title"])

st.markdown(
    """
Use the sidebar to explore:

1. **Overview** - the AI Incident Database at a glance
2. **Themes and Trends** - risk themes and how they change over time
3. **Risk Network** - incidents, organisations, harmed groups and risk domains
4. **Classify and Compare** - risk-domain classifiers and the AI-label audit
5. **Method and Limitations** - how the analysis was done and where it falls short
"""
)

st.info("No data loaded yet - this is the project scaffold. Analysis pages will fill in later.")
