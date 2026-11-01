"""Blind labelling helper for the RQ3 label audit.

Run:  uv run streamlit run scripts/label_audit_app.py

Shows one incident at a time (title and description only) and writes YOUR choice into
data/manual/label_audit_sample.csv. The MIT label is never loaded here, nothing is pre-selected, and each
answer is saved when you press 'Save and next'. Edit the CSV directly if you prefer; both work.
Set AICIDENTIA_LABEL_FILE to point at another file (used by the tests).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from aicidentia.models.label_audit import SAMPLE_PATH

PATH = Path(os.environ.get("AICIDENTIA_LABEL_FILE", SAMPLE_PATH))
# MIT AI Risk Repository domains (names exactly as in the data); order = MIT numbering.
DOMAINS = [
    (
        "Discrimination and Toxicity",
        "Unfair discrimination, toxic or harmful content, unequal performance across groups.",
    ),
    (
        "Privacy & Security",
        "Leaking or inferring sensitive information; attacks on and vulnerabilities in AI systems.",
    ),
    ("Misinformation", "False or misleading information; pollution of the information ecosystem."),
    (
        "Malicious Actors & Misuse",
        "Disinformation and influence at scale, cyberattacks and weapons, fraud, scams and targeted manipulation.",
    ),
    (
        "Human-Computer Interaction",
        "Overreliance and unsafe use; loss of human agency and autonomy.",
    ),
    (
        "Socioeconomic & Environmental Harms",
        "Power centralisation, inequality and job quality, devaluation of human effort, governance failure, environmental harm.",
    ),
    (
        "AI system safety, failures, and limitations",
        "AI pursuing its own goals, dangerous capabilities, lack of capability or robustness, lack of transparency, AI welfare.",
    ),
]
OPTIONS = [f"{i}. {name}" for i, (name, _) in enumerate(DOMAINS, 1)]
CONFIDENCE = {"1 = low": "1", "2 = medium": "2", "3 = high": "3"}


def load() -> pd.DataFrame:
    return pd.read_csv(PATH, dtype=str, keep_default_na=False)


def save(df: pd.DataFrame) -> None:
    tmp = PATH.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
    tmp.replace(PATH)


def on_save(idx: int, incident_id: str) -> None:
    """Button callback: validate, write this answer to the CSV, and jump to the next unannotated item."""
    label = st.session_state.get(f"label_{incident_id}")
    confidence = st.session_state.get(f"conf_{incident_id}")
    if label is None or confidence is None:
        st.session_state["warn"] = True
        return
    frame = load()
    frame.loc[idx, ["my_label", "my_confidence", "notes"]] = [
        label.split(". ", 1)[1],
        CONFIDENCE[confidence],
        st.session_state.get(f"notes_{incident_id}", ""),
    ]
    save(frame)
    st.session_state["warn"] = False
    remaining = [i for i in range(len(frame)) if frame.loc[i, "my_label"].strip() == ""]
    st.session_state["jump"] = (remaining[0] if remaining else idx) + 1


st.set_page_config(page_title="AICIDENTIA label audit", layout="centered")
st.title("AICIDENTIA label audit")
st.caption(
    "Blind annotation: assign the ONE risk domain that best describes the incident. The machine-generated label is not shown."
)

if not PATH.exists():
    st.error(f"{PATH} not found. Run scripts/run_phase6_classification.py first.")
    st.stop()

df = load()
answered = int((df["my_label"].str.strip() != "").sum())
st.progress(answered / len(df), text=f"{answered} of {len(df)} annotated")
first_open = next((i for i in range(len(df)) if df.loc[i, "my_label"].strip() == ""), 0)
idx = int(st.number_input("Item (1-based)", 1, len(df), value=first_open + 1, key="jump")) - 1
row = df.loc[idx]

with st.expander("Domain definitions (short; full text in docs/label_guidelines.md)"):
    for i, (name, text) in enumerate(DOMAINS, 1):
        st.markdown(f"**{i}. {name}**: {text}")

st.subheader(row["title"])
st.write(row["description"])
st.caption(f"Incident #{row['incident_id']}")

existing = next((o for o in OPTIONS if o.split(". ", 1)[1] == row["my_label"]), None)
st.radio(
    "Risk domain",
    OPTIONS,
    index=OPTIONS.index(existing) if existing else None,
    key=f"label_{row['incident_id']}",
)
conf_default = {v: k for k, v in CONFIDENCE.items()}.get(row["my_confidence"])
st.radio(
    "Confidence",
    list(CONFIDENCE),
    index=list(CONFIDENCE).index(conf_default) if conf_default else None,
    horizontal=True,
    key=f"conf_{row['incident_id']}",
)
st.text_input("Notes (optional)", value=row["notes"], key=f"notes_{row['incident_id']}")
st.button("Save and next", type="primary", on_click=on_save, args=(idx, row["incident_id"]))
if st.session_state.get("warn"):
    st.warning("Choose a domain and a confidence level first.")
