"""RQ3 label audit: a blind 150-incident sample for YOUR annotation, and the analysis of your labels.

The pipeline only creates the template (title and description, no MIT label, shuffled) and never fills in
`my_label`, `my_confidence` or `notes`. Once you have annotated, `audit_analysis` compares your labels with
the machine-generated MIT labels (Cohen's kappa with a bootstrap CI, confusion matrix, per-domain agreement)
and with the best classifier, and reports both the raw-sample and the design-weighted (population) estimate,
because the sample over-represents the small domains on purpose.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aicidentia.config import PROJECT_ROOT
from aicidentia.evaluation import metrics

logger = logging.getLogger(__name__)
SAMPLE_PATH = PROJECT_ROOT / "data" / "manual" / "label_audit_sample.csv"
KEY_PATH = PROJECT_ROOT / "data" / "processed" / "label_audit_key.parquet"
TEMPLATE_COLUMNS = ["incident_id", "title", "description", "my_label", "my_confidence", "notes"]
CONFIDENCE_LEVELS = {"1": "low", "2": "medium", "3": "high"}


def allocate(counts: pd.Series, total: int = 150, floor: int = 10) -> pd.Series:
    """Sample size per stratum: `floor` each (or the whole stratum if smaller), the rest proportional to size
    by the largest-remainder method, never more than the stratum holds."""
    base = counts.clip(upper=floor)
    remaining = total - int(base.sum())
    capacity = counts - base
    if remaining < 0:
        raise ValueError("floor x strata exceeds the total sample size")
    share = (capacity / capacity.sum() * remaining) if capacity.sum() else capacity * 0.0
    extra = np.minimum(np.floor(share), capacity).astype(int)
    leftover = remaining - int(extra.sum())
    remainders = (share - np.floor(share)).where(capacity - extra > 0).sort_values(ascending=False)
    for stratum in remainders.index[:leftover]:
        extra[stratum] += 1
    return (base + extra).astype(int)


def draw_sample(
    frame: pd.DataFrame, total: int = 150, floor: int = 10, seed: int = 42
) -> pd.DataFrame:
    """Stratified (by MIT domain), seeded, shuffled sample. `frame` needs incident_id, title, description, domain."""
    sizes = allocate(frame["domain"].value_counts(), total, floor)
    parts = [
        frame[frame["domain"] == d].sample(n=int(n), random_state=seed)
        for d, n in sizes.items()
        if n > 0
    ]
    sample = (
        pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    )  # shuffled: order reveals nothing
    return sample


def write_template(
    sample: pd.DataFrame, path: Path = SAMPLE_PATH, key_path: Path = KEY_PATH
) -> str:
    """Write the blind template and the (git-ignored) key. Never overwrites an existing annotation file.

    Returns 'created', 'unchanged' (same incidents already there) or 'exists_different' (a file with other
    incidents exists; it is left untouched).
    """
    key = sample[["incident_id", "domain"]].rename(columns={"domain": "mit_domain"})
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key.to_parquet(key_path, index=False)
    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        same = set(existing["incident_id"].astype(int)) == set(sample["incident_id"].astype(int))
        return "unchanged" if same else "exists_different"
    out = sample[["incident_id", "title", "description"]].assign(
        my_label="", my_confidence="", notes=""
    )[TEMPLATE_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return "created"


def normalise_label(value: str, classes: list[str]) -> str | None:
    """Map what you typed to a domain name: exact name (any case), MIT-numbered name ('1. ...') or just the digit
    (1-7, MIT numbering: 1 Discrimination, 2 Privacy, 3 Misinformation, 4 Malicious, 5 HCI, 6 Socioeconomic, 7 AI safety)."""
    text = str(value).strip()
    if not text:
        return None
    by_lower = {c.lower(): c for c in classes}
    stripped = re.sub(r"^\d+\.?\s*", "", text).strip().lower()
    if text.lower() in by_lower:
        return by_lower[text.lower()]
    if stripped in by_lower:
        return by_lower[stripped]
    numbered = {
        "1": "discrimination and toxicity",
        "2": "privacy & security",
        "3": "misinformation",
        "4": "malicious actors & misuse",
        "5": "human-computer interaction",
        "6": "socioeconomic & environmental harms",
        "7": "ai system safety, failures, and limitations",
    }
    if text in numbered and numbered[text] in by_lower:
        return by_lower[numbered[text]]
    return None


def load_annotations(path: Path = SAMPLE_PATH, classes: list[str] | None = None) -> pd.DataFrame:
    """Your annotations with normalised `label` (None where blank or unrecognised) and `confidence` 1-3."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw["incident_id"] = raw["incident_id"].astype(int)
    if classes is not None:
        raw["label"] = raw["my_label"].map(lambda v: normalise_label(v, classes))
    else:
        raw["label"] = raw["my_label"].replace("", None)
    raw["confidence"] = pd.to_numeric(raw["my_confidence"], errors="coerce")
    return raw


def annotation_progress(
    path: Path = SAMPLE_PATH, classes: list[str] | None = None
) -> dict[str, int]:
    """Rows, rows with a recognised label, rows whose label text could not be recognised."""
    if not path.exists():
        return {"rows": 0, "labelled": 0, "unrecognised": 0}
    ann = load_annotations(path, classes)
    typed = ann["my_label"].str.strip() != ""
    return {
        "rows": len(ann),
        "labelled": int(ann["label"].notna().sum()),
        "unrecognised": int((typed & ann["label"].isna()).sum()),
    }


# --------------------------------------------------------------------------------------
# Analysis of your annotations
# --------------------------------------------------------------------------------------


def weighted_kappa_from_pairs(
    a: np.ndarray, b: np.ndarray, weights: np.ndarray, labels: list[str]
) -> float:
    """Cohen's kappa from the design-weighted contingency table (weights = stratum population / stratum sample size)."""
    index = {lab: i for i, lab in enumerate(labels)}
    table = np.zeros((len(labels), len(labels)))
    for x, y, w in zip(a, b, weights, strict=True):
        table[index[x], index[y]] += w
    total = table.sum()
    po = np.trace(table) / total
    pe = float((table.sum(axis=1) / total) @ (table.sum(axis=0) / total))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def design_weights(reference: pd.Series, population_counts: pd.Series) -> np.ndarray:
    """Per-item weight = stratum population size / stratum sample size, strata = the MIT label."""
    sample_counts = reference.value_counts()
    return reference.map(lambda d: population_counts[d] / sample_counts[d]).to_numpy(dtype=float)


def weighted_kappa_bootstrap(
    mit: pd.Series,
    human: pd.Series,
    population_counts: pd.Series,
    labels: list[str],
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Design-weighted kappa with a stratified bootstrap (resampling within each MIT stratum)."""
    rng = np.random.default_rng(seed)
    mit_a, human_a = mit.to_numpy(), human.to_numpy()
    strata = {d: np.flatnonzero(mit_a == d) for d in np.unique(mit_a)}
    weights = design_weights(mit, population_counts)
    point = weighted_kappa_from_pairs(mit_a, human_a, weights, labels)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in strata.values()])
        draws.append(weighted_kappa_from_pairs(mit_a[idx], human_a[idx], weights[idx], labels))
    return point, float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975))


def audit_analysis(
    annotations: pd.DataFrame,
    key: pd.DataFrame,
    population_counts: pd.Series,
    classes: list[str],
    model_pred: pd.DataFrame | None = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare your labels with the MIT labels (and a classifier) on the annotated items.

    `annotations` from `load_annotations`; `key` has incident_id, mit_domain; `population_counts` is the number of
    labelled incidents per MIT domain (for design weights); `model_pred` (optional) has incident_id, predicted
    (out-of-fold prediction of the best classifier). Reports the raw-sample kappa and the design-weighted kappa.
    """
    df = annotations.merge(key, on="incident_id", how="left")
    df = df[df["label"].notna()].copy()
    if df.empty:
        raise ValueError("no recognised annotations yet")
    complete = len(df) == len(annotations)
    kappa, lo, hi = metrics.kappa_bootstrap(df["mit_domain"], df["label"], n_boot, seed)
    wk, wlo, whi = weighted_kappa_bootstrap(
        df["mit_domain"], df["label"], population_counts, classes, n_boot, seed
    )
    weights = design_weights(df["mit_domain"], population_counts)
    agree = (df["mit_domain"] == df["label"]).to_numpy()
    result: dict[str, Any] = {
        "n_annotated": int(len(df)),
        "population": int(population_counts.sum()),
        "complete": complete,
        "agreement_raw": float(agree.mean()),
        "agreement_design_weighted": float((agree * weights).sum() / weights.sum()),
        "kappa_raw": kappa,
        "kappa_raw_ci": (lo, hi),
        "kappa_raw_band": metrics.landis_koch(kappa),
        "kappa_weighted": wk,
        "kappa_weighted_ci": (wlo, whi),
        "kappa_weighted_band": metrics.landis_koch(wk),
        "confusion": pd.crosstab(df["mit_domain"], df["label"]).reindex(
            index=classes, columns=classes, fill_value=0
        ),
        "per_domain": metrics.per_class_agreement(df["mit_domain"], df["label"], classes),
        "by_confidence": df.groupby("confidence")
        .apply(
            lambda g: pd.Series(
                {"n": len(g), "agreement": float((g["mit_domain"] == g["label"]).mean())}
            ),
            include_groups=False,
        )
        .reset_index(),
    }
    if model_pred is not None:
        both = df.merge(model_pred, on="incident_id", how="inner")
        result["classifier"] = {
            "n": int(len(both)),
            "kappa_mit_vs_human": metrics.cohens_kappa(both["mit_domain"], both["label"]),
            "kappa_model_vs_human": metrics.cohens_kappa(both["predicted"], both["label"]),
            "kappa_model_vs_mit": metrics.cohens_kappa(both["predicted"], both["mit_domain"]),
        }
    return result


# --------------------------------------------------------------------------------------
# CSET comparison (needs none of your labels)
# --------------------------------------------------------------------------------------


def cset_comparison(incidents: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compare MIT labels with the human-assigned CSETv1 fields where a mapping is defensible.

    No CSET field is a risk-domain label, so two PARTIAL proxies are examined (documented in the doc):
    (1) MIT `Intent` against CSET `Intentional Harm`; (2) CSET `Protected Characteristic` against the MIT
    domain 'Discrimination and Toxicity'. Low agreement partly reflects definitional mismatch, not only noise.
    """
    both = incidents[incidents["has_mit"] & incidents["has_cset"]].copy()
    out: dict[str, pd.DataFrame] = {}
    harm = both[both["cset_intentional_harm"].notna() & both["mit_intent"].notna()].copy()
    harm["cset"] = (
        harm["cset_intentional_harm"]
        .str.slice(0, 3)
        .map({"Yes": "intentional", "No.": "not intentional"})
        .fillna("unclear")
    )
    out["intent"] = pd.crosstab(harm["mit_intent"], harm["cset"])
    prot = both[both["cset_protected_characteristic"].notna()].copy()
    prot["mit_is_discrimination"] = np.where(
        prot["mit_risk_domain"] == "Discrimination and Toxicity",
        "MIT: Discrimination and Toxicity",
        "MIT: other domain",
    )
    out["protected"] = pd.crosstab(
        prot["cset_protected_characteristic"], prot["mit_is_discrimination"]
    )
    return out
