"""Figure style and output helpers: IEEE column widths, colour-blind-safe palette, PNG + SVG."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; notebooks still display inline via IPython
import matplotlib.pyplot as plt
import pandas as pd

from aicidentia.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

SINGLE_COLUMN_IN = 3.5
DOUBLE_COLUMN_IN = 7.16

# Okabe-Ito palette: distinguishable under the common forms of colour-blindness.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "grey": "#7F7F7F",
}
CATEGORICAL = [
    PALETTE[k] for k in ("blue", "orange", "green", "vermillion", "sky", "purple", "grey")
]


def apply_style() -> None:
    """Set matplotlib defaults for IEEE-sized figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=CATEGORICAL),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
        }
    )


def source_caption(snapshot_date: str | int | None = None) -> str:
    """Standard caption: 'Source: AIID snapshot <date>'."""
    snapshot_date = snapshot_date or load_config()["snapshot_date"]
    return f"Source: AIID snapshot {snapshot_date}"


def add_source_caption(fig: plt.Figure, snapshot_date: str | int | None = None) -> None:
    """Write the source caption at the bottom-left of the figure."""
    fig.text(
        0.01, 0.005, source_caption(snapshot_date), fontsize=6, color=PALETTE["grey"], va="bottom"
    )


def save_figure(fig: plt.Figure, name: str, snapshot_date: str | int | None = None) -> list[Path]:
    """Save to reports/figures/<name>.png (300 dpi) and .svg; returns the written paths."""
    out_dir = PROJECT_ROOT / load_config()["paths"]["figures"]
    out_dir.mkdir(parents=True, exist_ok=True)
    add_source_caption(fig, snapshot_date)
    paths = []
    for ext in ("png", "svg"):
        path = out_dir / f"{name}.{ext}"
        # Reproducible files: no save date in the metadata, and fixed SVG element ids (svg.hashsalt), so
        # re-running a phase only changes a figure when its content changes.
        metadata = {"Date": None} if ext == "svg" else {"Software": None}
        with plt.rc_context({"svg.hashsalt": f"aicidentia-{name}"}):
            fig.savefig(path, bbox_inches="tight", pad_inches=0.05, metadata=metadata)
        paths.append(path)
    logger.info("Saved figure %s", name)
    return paths


def to_markdown(df: pd.DataFrame, index: bool = False, float_format: str = "{:.3g}") -> str:
    """Render a dataframe as a GitHub-flavoured Markdown table (no extra dependency)."""
    frame = df.reset_index() if index else df
    cols = [str(c) for c in frame.columns]

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return "" if pd.isna(value) else float_format.format(value)
        if value is None or (not isinstance(value, str | list | dict) and pd.isna(value)):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += [
        "| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)
    ]
    return "\n".join(lines)


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> list[Path]:
    """Save to reports/tables/<name>.csv and <name>.md; returns the written paths."""
    out_dir = PROJECT_ROOT / load_config()["paths"]["tables"]
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out_dir / f"{name}.csv", out_dir / f"{name}.md"
    df.to_csv(csv_path, index=index, encoding="utf-8")
    md_path.write_text(to_markdown(df, index=index) + "\n", encoding="utf-8", newline="\n")
    logger.info("Saved table %s (%d rows)", name, len(df))
    return [csv_path, md_path]
