"""Shared evaluation metrics: classification, bootstrap confidence intervals, paired tests, calibration, agreement.

Retrieval metrics (precision@k, MRR, nDCG) are added in Phase 7.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support

LANDIS_KOCH = (
    (0.0, "poor (below chance)"),
    (0.20, "slight"),
    (0.40, "fair"),
    (0.60, "moderate"),
    (0.80, "substantial"),
    (1.01, "almost perfect"),
)


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------


def macro_f1_codes(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-averaged F1 for integer-coded labels (fast; classes absent from both arrays count as 0)."""
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(
        n_classes, n_classes
    )
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(n_classes), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(n_classes), where=cm.sum(axis=1) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n_classes), where=denom > 0)
    return float(f1.mean())


def classification_report_table(
    y_true: Sequence, y_pred: Sequence, labels: Sequence | None = None
) -> pd.DataFrame:
    """Per-class precision, recall, F1 and support, plus macro and weighted averages."""
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    table = pd.DataFrame({"class": labels, "precision": p, "recall": r, "f1": f, "support": s})
    macro = {
        "class": "macro avg",
        "precision": p.mean(),
        "recall": r.mean(),
        "f1": f.mean(),
        "support": s.sum(),
    }
    weights = s / s.sum()
    weighted = {
        "class": "weighted avg",
        "precision": (p * weights).sum(),
        "recall": (r * weights).sum(),
        "f1": (f * weights).sum(),
        "support": s.sum(),
    }
    return pd.concat([table, pd.DataFrame([macro, weighted])], ignore_index=True)


def weighted_f1(y_true: Sequence, y_pred: Sequence) -> float:
    """Support-weighted F1."""
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap CI of macro-F1 (integer-coded labels, resampling incidents)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = macro_f1_codes(y_true, y_pred, n_classes)
    draws = np.array(
        [
            macro_f1_codes(y_true[i], y_pred[i], n_classes)
            for i in (rng.integers(0, n, n) for _ in range(n_boot))
        ]
    )
    return point, float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def paired_bootstrap_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_classes: int,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap of macro-F1(A) - macro-F1(B) on the same resampled incidents.

    Returns the observed difference, its 95% CI and a two-sided p-value (twice the smaller tail share of
    resampled differences on either side of zero, capped at 1).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = macro_f1_codes(y_true[idx], pred_a[idx], n_classes) - macro_f1_codes(
            y_true[idx], pred_b[idx], n_classes
        )
    observed = macro_f1_codes(y_true, pred_a, n_classes) - macro_f1_codes(y_true, pred_b, n_classes)
    p = min(1.0, 2 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean())))
    # (diffs == 0 on both sides counts fully; with p = 0 reported as 1/(n_boot+1) so it is never exactly zero)
    return {
        "diff": float(observed),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "p_value": max(p, 1 / (n_boot + 1)),
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test on the paired correct/incorrect outcomes of two classifiers."""
    from statsmodels.stats.contingency_tables import mcnemar

    ok_a, ok_b = pred_a == y_true, pred_b == y_true
    table = [
        [int((ok_a & ok_b).sum()), int((ok_a & ~ok_b).sum())],
        [int((~ok_a & ok_b).sum()), int((~ok_a & ~ok_b).sum())],
    ]
    res = mcnemar(table, exact=True)
    return {
        "a_only_correct": table[0][1],
        "b_only_correct": table[1][0],
        "p_value": float(res.pvalue),
    }


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 1000, seed: int | None = 42
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean of `values`."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(arr, len(arr), replace=True).mean() for _ in range(n_boot)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------


def reliability_table(
    y_true_codes: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Top-label reliability: per confidence bin the mean confidence, accuracy and count."""
    confidence = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true_codes).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        rows.append(
            {
                "bin": b,
                "low": edges[b],
                "high": edges[b + 1],
                "n": int(mask.sum()),
                "confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
                "accuracy": float(correct[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true_codes: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    """Top-label expected calibration error: sum over bins of (n_bin / n) * |accuracy - confidence|."""
    table = reliability_table(y_true_codes, proba, n_bins)
    valid = table[table["n"] > 0]
    return float(
        (valid["n"] / valid["n"].sum() * (valid["accuracy"] - valid["confidence"]).abs()).sum()
    )


def multiclass_brier(y_true_codes: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared error between the probability vector and the one-hot truth (lower is better)."""
    onehot = np.eye(proba.shape[1])[y_true_codes]
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


# --------------------------------------------------------------------------------------
# Agreement (label audit)
# --------------------------------------------------------------------------------------


def cohens_kappa(labels_a: Sequence, labels_b: Sequence, weights: str | None = None) -> float:
    """Cohen's kappa between two label sources."""
    return float(cohen_kappa_score(list(labels_a), list(labels_b), weights=weights))


def kappa_bootstrap(
    labels_a: Sequence, labels_b: Sequence, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    """Cohen's kappa with a percentile bootstrap 95% CI (resampling items; degenerate resamples are skipped)."""
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    rng = np.random.default_rng(seed)
    n = len(a)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(a[idx]) | set(b[idx])) < 2:
            continue
        values.append(cohens_kappa(a[idx], b[idx]))
    values = [v for v in values if not np.isnan(v)]
    if not values:  # every resample degenerate (e.g. one side constant): no interval
        return cohens_kappa(a, b) if len(set(a) | set(b)) > 1 else float("nan"), float("nan"), float("nan")
    return cohens_kappa(a, b), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def landis_koch(kappa: float) -> str:
    """Landis & Koch (1977) verbal band for a kappa value."""
    for upper, label in LANDIS_KOCH:
        if kappa < upper:
            return label
    return LANDIS_KOCH[-1][1]


def per_class_agreement(reference: Sequence, other: Sequence, labels: Sequence) -> pd.DataFrame:
    """For each class of `reference`: items, share where `other` agrees (= recall of the reference label), and
    for each class of `other`: share of its items that the reference also labels so (precision)."""
    ref, oth = np.asarray(reference), np.asarray(other)
    rows = []
    for lab in labels:
        in_ref, in_oth = ref == lab, oth == lab
        both = (in_ref & in_oth).sum()
        rows.append(
            {
                "class": lab,
                "reference_n": int(in_ref.sum()),
                "other_n": int(in_oth.sum()),
                "both": int(both),
                "agreement_of_reference": float(both / in_ref.sum()) if in_ref.any() else np.nan,
                "agreement_of_other": float(both / in_oth.sum()) if in_oth.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Retrieval (RQ4). `retrieved` is a ranked list of ids (best first, query itself excluded);
# `relevant` is the set of ids judged relevant for the query. Relevance is binary.
# --------------------------------------------------------------------------------------


def precision_at_k(retrieved: Sequence[int], relevant: set[int], k: int = 5) -> float:
    """Share of the top-k results that are relevant (the denominator is always k)."""
    return sum(1 for r in list(retrieved)[:k] if r in relevant) / k


def recall_at_k(retrieved: Sequence[int], relevant: set[int], k: int = 10) -> float:
    """Share of the relevant items found in the top k."""
    if not relevant:
        return float("nan")
    return sum(1 for r in list(retrieved)[:k] if r in relevant) / len(relevant)


def hit_at_k(retrieved: Sequence[int], relevant: set[int], k: int = 5) -> float:
    """1 if any relevant item is in the top k, else 0."""
    return float(any(r in relevant for r in list(retrieved)[:k]))


def reciprocal_rank(
    retrieved: Sequence[int], relevant: set[int], depth: int | None = None
) -> float:
    """1 / rank of the first relevant item (0 if none within `depth`, or within the whole list)."""
    for rank, r in enumerate(list(retrieved)[:depth] if depth else retrieved, start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[int], relevant: set[int], k: int = 10) -> float:
    """Binary-relevance nDCG@k: DCG of the ranking divided by the DCG of a perfect ranking."""
    gains = [1.0 / np.log2(i + 2) for i, r in enumerate(list(retrieved)[:k]) if r in relevant]
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(sum(gains) / ideal) if ideal else float("nan")


def mean_reciprocal_rank(retrieved: Sequence[Sequence[int]], relevant: Sequence[set[int]]) -> float:
    """Mean reciprocal rank over queries."""
    return float(
        np.mean([reciprocal_rank(r, rel) for r, rel in zip(retrieved, relevant, strict=True)])
    )


def paired_bootstrap_mean_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 42
) -> dict[str, float]:
    """Paired bootstrap of mean(a) - mean(b) over queries: difference, 95% CI and two-sided p-value."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    diffs = (x[idx] - y[idx]).mean(axis=1)
    p = min(1.0, 2 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean())))
    return {
        "diff": float((x - y).mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "p_value": max(p, 1 / (n_boot + 1)),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as given)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()
