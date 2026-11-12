"""RQ3 (second half): how reliable are the AI-generated MIT risk-domain labels, judged against independent
signals already in AIID? Three complementary checks (docs/decisions.md D6-D8):

a) Agreement with human annotations (CSETv1, GMF) where a defensible binary mapping exists. These are partial,
   convergent-validity checks: no human field is a risk-domain label, so agreement is bounded by definitional
   mismatch and is not ground truth.
b) Consistency on editor-linked pairs: how often two incidents an AIID editor marked as similar carry the same MIT
   domain, against a permutation baseline of random pairs matched on year, and for editor-marked dissimilar pairs.
c) Confident learning (Northcutt, Jiang & Chuang, 2021, JAIR 70): out-of-fold predicted probabilities of the best
   classifier estimate the joint distribution of given and latent labels (cleanlab), the noise rate per domain,
   and flag likely mislabelled incidents. A sensitivity check retrains the classifier without the flagged
   incidents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from aicidentia.evaluation import metrics as M

logger = logging.getLogger(__name__)

DISCRIMINATION = "Discrimination and Toxicity"
SAFETY = "AI system safety, failures, and limitations"
MISINFORMATION = "Misinformation"
MALICIOUS = "Malicious Actors & Misuse"


# --------------------------------------------------------------------------------------
# a) Agreement with human annotations
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Mapping:
    """A binary comparison: MIT says one of `mit_domains` vs a human field says 'yes'."""

    key: str
    mit_domains: tuple[str, ...]
    source: str  # CSETv1 | GMF
    field_label: str
    rationale: str


def _cset_yes_no(value: Any) -> bool | None:
    v = str(value).strip().lower() if value is not None and not pd.isna(value) else ""
    return {"yes": True, "no": False}.get(v)


def _tangible(value: Any) -> bool | None:
    v = str(value).strip().lower() if value is not None and not pd.isna(value) else ""
    if v.startswith("tangible harm definitively") or v.startswith("imminent risk"):
        return True
    if v.startswith("no tangible") or v.startswith("non-imminent"):
        return False
    return None  # unclear / missing


def _basis(values: Any) -> bool | None:
    if values is None or (not isinstance(values, list | np.ndarray) and pd.isna(values)):
        return None
    items = {str(v).strip().lower() for v in values}
    if not items or items <= {"unclear"}:
        return None
    return bool(items - {"none", "unclear"})


def _technical_failure(values: Any) -> bool | None:
    if values is None or (not isinstance(values, list | np.ndarray) and pd.isna(values)):
        return None
    return len(values) > 0


MAPPINGS: list[tuple[Mapping, str, Any]] = [
    (
        Mapping(
            "discrimination_vs_protected_characteristic",
            (DISCRIMINATION,),
            "CSETv1",
            "Protected Characteristic = yes",
            "harm linked to a protected characteristic is the core of MIT's discrimination subdomains",
        ),
        "cset_protected_characteristic",
        _cset_yes_no,
    ),
    (
        Mapping(
            "discrimination_vs_harm_distribution_basis",
            (DISCRIMINATION,),
            "CSETv1",
            "Harm Distribution Basis is not 'none'",
            "harm unevenly distributed by race, sex, religion etc. signals discrimination",
        ),
        "cset_harm_distribution_basis",
        _basis,
    ),
    (
        Mapping(
            "safety_vs_tangible_harm",
            (SAFETY,),
            "CSETv1",
            "Tangible Harm occurred or imminent (near miss)",
            "AI-system failures are where physical/tangible harm and near misses concentrate",
        ),
        "cset_tangible_harm",
        _tangible,
    ),
    (
        Mapping(
            "safety_vs_gmf_technical_failure",
            (SAFETY,),
            "GMF",
            "a Technical Failure is recorded",
            "GMF names the technical failure behind an incident; MIT 'AI system safety, failures' is the failure domain",
        ),
        "gmf_technical_failure",
        _technical_failure,
    ),
    (
        Mapping(
            "misinformation_or_misuse_vs_detrimental_content",
            (MISINFORMATION, MALICIOUS),
            "CSETv1",
            "Detrimental Content = yes",
            "CSET's detrimental content lists deepfakes, identity misrepresentation, misinformation and scams",
        ),
        "cset_detrimental_content",
        _cset_yes_no,
    ),
]


def mapping_agreement(
    incidents: pd.DataFrame, n_boot: int = 2000, seed: int = 42
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """One row per mapping: coverage n, positives on each side, raw agreement, Cohen's kappa [95% CI]."""
    rows, tables = [], {}
    for mapping, column, parse in MAPPINGS:
        flag = "has_cset" if mapping.source == "CSETv1" else "has_gmf"
        sub = incidents[incidents["has_mit"] & incidents[flag]].copy()
        sub["human"] = sub[column].map(parse)
        excluded = int(sub["human"].isna().sum())
        sub = sub[sub["human"].notna()]
        mit = sub["mit_risk_domain"].isin(mapping.mit_domains).to_numpy()
        human = sub["human"].astype(bool).to_numpy()
        kappa, lo, hi = M.kappa_bootstrap(mit, human, n_boot, seed) if len(sub) else (np.nan,) * 3
        both = int((mit & human).sum())
        rows.append(
            {
                "mapping": mapping.key,
                "mit_domain": " or ".join(mapping.mit_domains),
                "human_field": f"{mapping.source} {mapping.field_label}",
                "n": len(sub),
                "excluded_unclear": excluded,
                "mit_positive": int(mit.sum()),
                "human_positive": int(human.sum()),
                "both_positive": both,
                "agreement": float((mit == human).mean()) if len(sub) else np.nan,
                "kappa": kappa,
                "kappa_low": lo,
                "kappa_high": hi,
                "band": M.landis_koch(kappa) if not np.isnan(kappa) else "",
                "precision_of_mit": both / mit.sum() if mit.sum() else np.nan,
                "recall_of_mit": both / human.sum() if human.sum() else np.nan,
                "years": f"{int(sub['year'].min())}-{int(sub['year'].max())}" if len(sub) else "",
                "rationale": mapping.rationale,
            }
        )
        tables[mapping.key] = pd.crosstab(
            pd.Series(np.where(mit, "MIT: yes", "MIT: no"), name="MIT"),
            pd.Series(np.where(human, "human: yes", "human: no"), name=mapping.source),
        )
    return pd.DataFrame(rows), tables


# --------------------------------------------------------------------------------------
# b) Consistency on editor-linked pairs
# --------------------------------------------------------------------------------------


def unique_pairs(links: dict[int, set[int]]) -> list[tuple[int, int]]:
    """Undirected pairs (a < b) from a symmetric adjacency dict."""
    return sorted({(min(a, b), max(a, b)) for a, targets in links.items() for b in targets})


def pair_consistency(
    pairs: list[tuple[int, int]],
    domain: dict[int, str],
    year: dict[int, int],
    n_perm: int = 2000,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Same-domain rate on pairs where both incidents are MIT-labelled, with a bootstrap CI over pairs and a
    permutation baseline that replaces each incident by a random labelled incident from the same year."""
    rng = np.random.default_rng(seed)
    usable = [(a, b) for a, b in pairs if a in domain and b in domain]
    if not usable:
        return {"pairs_total": len(pairs), "pairs_labelled": 0}
    same = np.array([domain[a] == domain[b] for a, b in usable], dtype=float)
    boot = [same[rng.integers(0, len(same), len(same))].mean() for _ in range(n_boot)]
    code = {d: k for k, d in enumerate(sorted(set(domain.values())))}
    pools: dict[int, list[int]] = {}
    for i in sorted(domain):
        pools.setdefault(year[i], []).append(code[domain[i]])
    pool_codes = {y: np.array(v) for y, v in pools.items()}

    def draw(endpoints: list[int]) -> np.ndarray:
        """n_perm x pairs matrix of domain codes of random same-year incidents."""
        out = np.empty((n_perm, len(endpoints)), dtype=int)
        years = np.array([year[i] for i in endpoints])
        for y in np.unique(years):
            cols = np.flatnonzero(years == y)
            pool = pool_codes[int(y)]
            out[:, cols] = pool[rng.integers(0, len(pool), (n_perm, len(cols)))]
        return out

    perm = (draw([a for a, _ in usable]) == draw([b for _, b in usable])).mean(axis=1)
    observed = float(same.mean())
    return {
        "pairs_total": len(pairs),
        "pairs_labelled": len(usable),
        "same_domain_rate": observed,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "baseline_rate": float(perm.mean()),
        "baseline_low": float(np.quantile(perm, 0.025)),
        "baseline_high": float(np.quantile(perm, 0.975)),
        "p_value": float(((perm >= observed).sum() + 1) / (n_perm + 1)),
        "lift": observed / float(perm.mean()) if perm.mean() else np.nan,
    }


def pair_domain_table(pairs: list[tuple[int, int]], domain: dict[int, str]) -> pd.DataFrame:
    """For editor-similar pairs whose MIT domains differ: which domain combinations occur (unordered)."""
    rows = [
        tuple(sorted((domain[a], domain[b])))
        for a, b in pairs
        if a in domain and b in domain and domain[a] != domain[b]
    ]
    if not rows:
        return pd.DataFrame(columns=["domain_1", "domain_2", "pairs"])
    out = pd.DataFrame(rows, columns=["domain_1", "domain_2"]).value_counts().rename("pairs")
    return out.reset_index().sort_values(["pairs", "domain_1"], ascending=[False, True])


# --------------------------------------------------------------------------------------
# c) Confident learning
# --------------------------------------------------------------------------------------


@dataclass
class ConfidentLearning:
    flagged: np.ndarray  # boolean mask over items
    ranked: pd.DataFrame  # flagged items with given label, suggested label, self-confidence
    joint: pd.DataFrame  # estimated P(given label, latent label), rows = given
    confident_joint: pd.DataFrame  # counts
    noise_by_domain: pd.DataFrame
    noise_rate: float
    facts: dict[str, Any] = field(default_factory=dict)


def confident_learning(
    labels: np.ndarray, pred_probs: np.ndarray, classes: list[str], incident_ids: np.ndarray
) -> ConfidentLearning:
    """cleanlab's confident learning with its default rules (prune_by_noise_rate)."""
    from cleanlab.count import compute_confident_joint, estimate_joint
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores

    probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)
    cj = compute_confident_joint(labels, probs)
    joint = estimate_joint(labels, probs)
    flagged = find_label_issues(labels, probs, filter_by="prune_by_noise_rate")
    quality = get_label_quality_scores(labels, probs, method="self_confidence")
    given_share = joint.sum(axis=1)
    noise = 1 - np.diag(joint) / np.where(given_share > 0, given_share, 1)
    counts = np.bincount(labels, minlength=len(classes))
    flagged_counts = np.bincount(labels[flagged], minlength=len(classes))
    off = joint.copy()
    np.fill_diagonal(off, 0)
    main_latent = [
        classes[int(np.argmax(off[i]))] if off[i].sum() > 0 else "" for i in range(len(classes))
    ]
    noise_by_domain = pd.DataFrame(
        {
            "domain": classes,
            "n": counts,
            "estimated_noise_rate": noise,
            "flagged": flagged_counts,
            "flagged_share": flagged_counts / np.maximum(counts, 1),
            "most_likely_latent_if_wrong": main_latent,
        }
    ).sort_values("estimated_noise_rate", ascending=False)
    suggested = probs.argmax(axis=1)
    ranked = (
        pd.DataFrame(
            {
                "incident_id": incident_ids,
                "given_label": [classes[i] for i in labels],
                "suggested_label": [classes[i] for i in suggested],
                "self_confidence": quality,
                "suggested_confidence": probs.max(axis=1),
            }
        )[flagged]
        .sort_values(["self_confidence", "incident_id"])
        .reset_index(drop=True)
    )
    cl = ConfidentLearning(
        flagged=np.asarray(flagged, dtype=bool),
        ranked=ranked,
        joint=pd.DataFrame(joint, index=classes, columns=classes),
        confident_joint=pd.DataFrame(cj, index=classes, columns=classes),
        noise_by_domain=noise_by_domain,
        noise_rate=float(1 - np.trace(joint)),
    )
    cl.facts = {
        "n": int(len(labels)),
        "flagged": int(flagged.sum()),
        "flagged_share": float(flagged.mean()),
        "estimated_noise_rate": cl.noise_rate,
    }
    return cl


def confusion_pairs(joint: pd.DataFrame, n: int, top: int = 8) -> pd.DataFrame:
    """Largest off-diagonal cells of the estimated joint, as incident counts (given -> latent)."""
    rows = [
        {
            "given_label": g,
            "likely_latent": t,
            "estimated_incidents": float(joint.loc[g, t] * n),
            "share_of_given": float(joint.loc[g, t] / joint.loc[g].sum()),
        }
        for g in joint.index
        for t in joint.columns
        if g != t
    ]
    out = pd.DataFrame(rows).sort_values("estimated_incidents", ascending=False).head(top)
    return out.reset_index(drop=True)


def symmetric_confusion(joint: pd.DataFrame, n: int) -> pd.DataFrame:
    """Off-diagonal mass per unordered domain pair and how lopsided it is (taxonomy overlap is often two-way)."""
    rows = []
    classes = list(joint.index)
    for i, a in enumerate(classes):
        for b in classes[i + 1 :]:
            ab, ba = joint.loc[a, b] * n, joint.loc[b, a] * n
            if ab + ba > 0:
                rows.append(
                    {
                        "domain_a": a,
                        "domain_b": b,
                        "a_to_b": float(ab),
                        "b_to_a": float(ba),
                        "total": float(ab + ba),
                        "two_way_share": float(min(ab, ba) / max(ab, ba)) if max(ab, ba) else 0.0,
                    }
                )
    return pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)


def retrain_without_flagged(
    spec: Any, data: Any, seed: int = 42, inner_folds: int = 5
) -> pd.DataFrame:
    """Leakage-free cleaning check over the SAME outer folds: inside each training part, inner cross-validation
    gives out-of-fold probabilities, confident learning flags likely mislabelled TRAINING incidents, the model is
    refitted without them and predicts the untouched test fold. No test label influences which incidents are
    removed. Returns incident_id, fold, y_true, y_pred, n_removed (per fold). Cached."""
    from cleanlab.filter import find_label_issues
    from sklearn.model_selection import StratifiedKFold

    from aicidentia import cache
    from aicidentia.models import classify as C

    x = C.features(spec, data)

    def compute() -> pd.DataFrame:
        parts = []
        for fold in range(C.N_FOLDS):
            test = data.folds == fold
            train_idx = np.flatnonzero(~test)
            y_train = data.y[train_idx]
            inner = np.zeros((len(train_idx), data.n_classes))
            splitter = StratifiedKFold(inner_folds, shuffle=True, random_state=seed + fold)
            for fit_part, held_out in splitter.split(train_idx, y_train):
                _, proba, _ = C.fit_predict(
                    spec,
                    x[train_idx[fit_part]],
                    y_train[fit_part],
                    x[train_idx[held_out]],
                    data.n_classes,
                    seed,
                    None,
                )
                inner[held_out] = proba
            inner = inner / inner.sum(axis=1, keepdims=True)
            issues = find_label_issues(y_train, inner, filter_by="prune_by_noise_rate")
            keep = train_idx[~issues]
            pred, _, _ = C.fit_predict(
                spec, x[keep], data.y[keep], x[test], data.n_classes, seed, None
            )
            parts.append(
                pd.DataFrame(
                    {
                        "incident_id": data.frame.loc[test, "incident_id"].to_numpy(),
                        "fold": fold,
                        "y_true": data.y[test],
                        "y_pred": pred,
                        "n_removed": int(issues.sum()),
                        "n_train": len(train_idx),
                    }
                )
            )
        return pd.concat(parts, ignore_index=True)

    key = (data.fingerprint, spec.name, spec.kind, spec.embedding_key, seed, inner_folds)
    # sklearn's inner parallelism on threads (process workers hang or exhaust memory on Windows here)
    out = cache.cached("rq3clean", key, C._threaded(compute), version="2")
    return out.sort_values("incident_id").reset_index(drop=True)


def retrain_sensitivity(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    flagged_ids: set[int],
    n_classes: int,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Macro-F1 of the original and the cleaned-training model, on all incidents (primary: the test labels
    are not touched) and on the unflagged incidents only (an optimistic view, shown for completeness)."""
    a = original.sort_values("incident_id").reset_index(drop=True)
    b = cleaned.sort_values("incident_id").reset_index(drop=True)
    assert (a["incident_id"].to_numpy() == b["incident_id"].to_numpy()).all()
    rows = []
    for subset, mask in (
        ("all incidents (primary)", np.ones(len(a), bool)),
        ("unflagged incidents only", ~a["incident_id"].isin(flagged_ids).to_numpy()),
    ):
        y = a.loc[mask, "y_true"].to_numpy()
        pa, pb = a.loc[mask, "y_pred"].to_numpy(), b.loc[mask, "y_pred"].to_numpy()
        diff = M.paired_bootstrap_diff(y, pb, pa, n_classes, n_boot, seed)
        rows.append(
            {
                "evaluated_on": subset,
                "n": int(mask.sum()),
                "macro_f1_original": M.macro_f1_codes(y, pa, n_classes),
                "macro_f1_cleaned_training": M.macro_f1_codes(y, pb, n_classes),
                "difference": diff["diff"],
                "ci_low": diff["ci_low"],
                "ci_high": diff["ci_high"],
                "p_value": diff["p_value"],
            }
        )
    return pd.DataFrame(rows)
