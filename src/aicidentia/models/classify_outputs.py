"""Phase 6 orchestration: evaluate every classifier, calibrate and save the deployable one."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from aicidentia import registry
from aicidentia.config import PROJECT_ROOT, load_config
from aicidentia.data import clean
from aicidentia.evaluation import metrics as M
from aicidentia.models import classify as C
from aicidentia.viz import plots, style

logger = logging.getLogger(__name__)
PRODUCER = "aicidentia.models.classify_outputs.run_classification"
SOURCE = "data/processed/incidents_clean.parquet"
DISTIL = "DistilBERT fine-tuned"
BASELINE = "TF-IDF + logistic regression"
CALIBRATION_MARGIN = 0.005
DUMMY = "majority class"
SPECS = [
    C.ModelSpec(DUMMY, "dummy"),
    C.ModelSpec(BASELINE, "tfidf_lr"),
    C.ModelSpec("TF-IDF + linear SVM", "tfidf_svm"),
    C.ModelSpec("MiniLM embeddings + logistic regression", "emb_lr", "sentence_embedding_small"),
    C.ModelSpec("mpnet embeddings + logistic regression", "emb_lr", "sentence_embedding_large"),
]
ABLATION = C.ModelSpec(
    "TF-IDF + LR, text + first report words (ablation)", "tfidf_lr", text_column="text_long"
)


@dataclass
class ClassificationResults:
    """Everything Phase 6 produces."""

    data: C.Dataset
    oof: dict[str, pd.DataFrame] = field(default_factory=dict)
    time_pred: dict[str, pd.DataFrame] = field(default_factory=dict)
    metrics: pd.DataFrame | None = None
    per_class: pd.DataFrame | None = None
    significance: pd.DataFrame | None = None
    best_model: str = ""
    deployable: str = ""
    deployable_reason: str = ""
    calibration: dict[str, Any] = field(default_factory=dict)
    confusion: pd.DataFrame | None = None
    confusions_top: pd.DataFrame | None = None
    features: pd.DataFrame | None = None
    reliability: Any = None  # reliability_outputs.ReliabilityResults
    facts: dict[str, Any] = field(default_factory=dict)
    colab_present: bool = False


def add_report_text(data: C.Dataset, words: int) -> None:
    """Add `text_long` = text + the first `words` words of the longest linked non-stub report (ablation input)."""
    reports = clean.load_processed("reports_clean")[["incident_ids", "text", "text_words"]]
    exploded = (
        reports.explode("incident_ids")
        .dropna(subset=["incident_ids"])
        .astype({"incident_ids": "int64"})
    )
    longest = (
        exploded.sort_values("text_words", ascending=False)
        .drop_duplicates("incident_ids")
        .set_index("incident_ids")["text"]
    )
    extra = (
        data.frame["incident_id"]
        .map(longest)
        .fillna("")
        .map(lambda t: " ".join(str(t).split()[:words]))
    )
    data.frame["text_long"] = (data.frame["text"] + " " + extra).str.strip()


def _row(
    name: str,
    setting: str,
    y: np.ndarray,
    pred: np.ndarray,
    k: int,
    n_boot: int,
    seed: int,
    folds: np.ndarray | None,
) -> dict[str, Any]:
    point, lo, hi = M.bootstrap_metric(y, pred, k, n_boot, seed)
    row = {
        "model": name,
        "setting": setting,
        "n": len(y),
        "macro_f1": point,
        "ci_low": lo,
        "ci_high": hi,
        "weighted_f1": M.weighted_f1(y, pred),
        "accuracy": float((y == pred).mean()),
    }
    if folds is not None:
        per_fold = [
            M.macro_f1_codes(y[folds == f], pred[folds == f], k) for f in sorted(set(folds))
        ]
        row.update(
            {
                "fold_macro_f1_mean": float(np.mean(per_fold)),
                "fold_macro_f1_sd": float(np.std(per_fold, ddof=1)),
            }
        )
    return row


def choose_deployable(
    rows: pd.DataFrame, oof: dict[str, pd.DataFrame], data: C.Dataset, n_boot: int, seed: int
) -> tuple[str, str]:
    """Pre-specified rule: the scikit-learn model (not the dummy, not DistilBERT) with the highest cross-validated macro-F1;
    but if its paired-bootstrap advantage over TF-IDF + LR is not significant (95% CI includes 0), choose
    TF-IDF + LR, which needs no model download and works offline."""
    local = [s.name for s in SPECS if s.kind != "dummy"]
    cv = rows[(rows["setting"] == "cv") & rows["model"].isin(local)].sort_values(
        "macro_f1", ascending=False
    )
    best = cv.iloc[0]["model"]
    if best == BASELINE:
        return best, "highest cross-validated macro-F1 among local models"
    y = oof[best]["y_true"].to_numpy()
    diff = M.paired_bootstrap_diff(
        y,
        oof[best]["y_pred"].to_numpy(),
        oof[BASELINE]["y_pred"].to_numpy(),
        data.n_classes,
        n_boot,
        seed,
    )
    if diff["ci_low"] <= 0 <= diff["ci_high"]:
        return BASELINE, (
            f"{best} has the highest macro-F1 but its advantage over {BASELINE} is not significant "
            f"(paired bootstrap difference {diff['diff']:+.3f}, 95% CI [{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}]); "
            "the simpler model is chosen because it needs no embedding model and works offline"
        )
    return best, (
        f"highest cross-validated macro-F1 and significantly better than {BASELINE} "
        f"(difference {diff['diff']:+.3f}, 95% CI [{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}])"
    )


def run_classification() -> ClassificationResults:
    """Evaluate all classifiers and produce the deployable model and every table."""
    cfg = load_config()
    seed = int(cfg["random_seed"])
    n_boot = int(cfg["classification"]["n_boot"])
    data = C.load_dataset()
    add_report_text(data, int(cfg["classification"]["report_words"]))
    C.export_colab_input(data, PROJECT_ROOT / "reports" / "colab" / "rq3_colab_input.csv")
    res = ClassificationResults(data)
    k = data.n_classes

    for spec in [*SPECS, ABLATION]:
        res.oof[spec.name] = C.cross_validate(spec, data, seed=seed)
        res.time_pred[spec.name] = C.time_split_predict(spec, data, seed=seed)
    try:
        colab = C.load_colab_predictions(data)
        colab_problem = ""
    except (
        ValueError
    ) as exc:  # e.g. folds changed after a duplicate merge: re-run the Colab notebook
        colab, colab_problem = None, str(exc)
        logger.warning("Colab predictions not used: %s", exc)
    res.colab_present = colab is not None
    if colab is not None:
        res.oof[DISTIL], res.time_pred[DISTIL] = colab

    # ---- metrics (pooled out-of-fold, bootstrap CI) and per-class table
    rows = []
    per_class = []
    for name, oof in res.oof.items():
        rows.append(
            _row(
                name,
                "cv",
                oof["y_true"].to_numpy(),
                oof["y_pred"].to_numpy(),
                k,
                n_boot,
                seed,
                oof["fold"].to_numpy(),
            )
        )
        tp = res.time_pred[name]
        rows.append(
            _row(
                name,
                "time",
                tp["y_true"].to_numpy(),
                tp["y_pred"].to_numpy(),
                k,
                n_boot,
                seed,
                None,
            )
        )
        report = M.classification_report_table(
            oof["y_true"].map(dict(enumerate(data.classes))),
            oof["y_pred"].map(dict(enumerate(data.classes))),
            data.classes,
        )
        per_class.append(report.assign(model=name))
    res.metrics = pd.DataFrame(rows)
    res.per_class = pd.concat(per_class, ignore_index=True)

    cv = res.metrics[res.metrics["setting"] == "cv"].sort_values("macro_f1", ascending=False)
    res.best_model = str(cv[~cv["model"].str.contains("ablation")].iloc[0]["model"])

    # ---- paired significance
    sig = []
    pairs = [(m, BASELINE) for m in res.oof if m not in (BASELINE,)] + [(res.best_model, DUMMY)]
    for a, b in dict.fromkeys(pairs):
        if a == b:
            continue
        for setting, source in (("cv", res.oof), ("time", res.time_pred)):
            ya, pa, pb = (
                source[a]["y_true"].to_numpy(),
                source[a]["y_pred"].to_numpy(),
                source[b]["y_pred"].to_numpy(),
            )
            if not (
                source[a]["incident_id"].to_numpy() == source[b]["incident_id"].to_numpy()
            ).all():
                continue
            boot = M.paired_bootstrap_diff(ya, pa, pb, k, n_boot, seed)
            mc = M.mcnemar_test(ya, pa, pb)
            sig.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "setting": setting,
                    **boot,
                    "mcnemar_p": mc["p_value"],
                    "a_only_correct": mc["a_only_correct"],
                    "b_only_correct": mc["b_only_correct"],
                }
            )
    res.significance = pd.DataFrame(sig)

    # ---- deployable model, calibration
    res.deployable, res.deployable_reason = choose_deployable(
        res.metrics, res.oof, data, n_boot, seed
    )
    spec = next(s for s in SPECS if s.name == res.deployable)
    raw = res.oof[res.deployable]
    y = raw["y_true"].to_numpy()
    cols = [f"proba_{i}" for i in range(k)]
    oof_by_method = {"raw": raw}
    for method in ("sigmoid", "temperature"):
        oof_by_method[method] = C.cross_validate(spec, data, calibrate=method, seed=seed)
    cal: dict[str, Any] = {}
    for name, frame in oof_by_method.items():
        p = frame[cols].to_numpy()
        cal[name] = {
            "ece": M.expected_calibration_error(y, p),
            "brier": M.multiclass_brier(y, p),
            "accuracy": float((p.argmax(1) == y).mean()),
            "mean_confidence": float(p.max(1).mean()),
            "macro_f1": M.macro_f1_codes(y, p.argmax(1), k),
            "table": M.reliability_table(y, p),
        }
    # Rule fixed in advance: lowest expected calibration error, but the uncalibrated model is kept unless a
    # calibrator lowers ECE by more than CALIBRATION_MARGIN (a calibrator also adds complexity).
    chosen = min(cal, key=lambda m: cal[m]["ece"] + (0.0 if m == "raw" else CALIBRATION_MARGIN))
    res.calibration = {**cal, "chosen": chosen, "oof": oof_by_method}
    p_chosen = oof_by_method[chosen][cols].to_numpy()

    # ---- error analysis
    best_oof = res.oof[res.best_model]
    res.confusion = C.confusion(
        best_oof["y_true"].to_numpy(), best_oof["y_pred"].to_numpy(), data.classes, normalise=True
    )
    res.confusions_top = C.top_confusions(
        res.oof[res.deployable].assign(**{f"proba_{i}": p_chosen[:, i] for i in range(k)}),
        data,
    )
    res.features = pd.concat(
        [C.top_features(s, data, seed=seed) for s in SPECS if s.kind in ("tfidf_lr", "tfidf_svm")],
        ignore_index=True,
    )

    # ---- save the deployable model and predictions for every incident
    bundle = C.fit_final_model(spec, data, calibrate=None if chosen == "raw" else chosen, seed=seed)
    C.save_model(bundle)
    # Fallback for the dashboard when the embedding model cannot be loaded: TF-IDF + LR, calibration chosen
    # by the same rule on its own out-of-fold predictions.
    base_spec = next(s_ for s_ in SPECS if s_.name == BASELINE)
    base_oof = {
        "raw": res.oof[BASELINE],
        "temperature": C.cross_validate(base_spec, data, calibrate="temperature", seed=seed),
    }
    base_ece = {
        m: M.expected_calibration_error(y, f_[cols].to_numpy()) for m, f_ in base_oof.items()
    }
    fallback_cal = min(
        base_ece, key=lambda m: base_ece[m] + (0.0 if m == "raw" else CALIBRATION_MARGIN)
    )
    C.save_model(
        C.fit_final_model(
            base_spec, data, calibrate=None if fallback_cal == "raw" else fallback_cal, seed=seed
        ),
        C.FALLBACK_MODEL_PATH,
    )
    inc = clean.load_processed("incidents_clean")
    final = C.predict_domains(bundle, inc["text"].tolist()).assign(
        incident_id=inc["incident_id"].to_numpy(), source="final_model"
    )
    oof_pred = oof_by_method[chosen]
    probs = oof_pred[[f"proba_{i}" for i in range(k)]].to_numpy()
    oof_df = pd.DataFrame(
        {
            "incident_id": oof_pred["incident_id"].to_numpy(),
            "predicted_domain": [data.classes[i] for i in probs.argmax(1)],
            "confidence": probs.max(1),
            "source": "out_of_fold",
        }
    )
    predictions = pd.concat(
        [
            oof_df,
            final[~final["incident_id"].isin(oof_df["incident_id"])][
                ["incident_id", "predicted_domain", "confidence", "source"]
            ],
        ]
    )
    predictions = predictions.merge(
        inc[["incident_id", "mit_risk_domain"]], on="incident_id", how="left"
    ).sort_values("incident_id")
    clean.save_processed(predictions, "rq3_domain_predictions")

    # ---- label reliability (RQ3, second half): CSETv1/GMF agreement, editor-linked pairs, confident learning.
    # Confident learning uses the best classifier's out-of-fold probabilities (calibrated if it is the deployed
    # model); if the best model cannot be refitted here (DistilBERT), the deployed model is used instead.
    from aicidentia.models import reliability_outputs

    rel_spec = next((s_ for s_ in SPECS if s_.name == res.best_model), spec)
    rel_probs = oof_by_method[chosen] if rel_spec.name == res.deployable else res.oof[rel_spec.name]
    res.reliability = reliability_outputs.run_reliability(
        inc, data, rel_spec.name, rel_spec, rel_probs, res.oof[rel_spec.name], seed, n_boot
    )

    population = data.frame["domain"].value_counts()
    res.facts = {
        "labelled": len(data.y),
        "classes": data.classes,
        "class_counts": population.to_dict(),
        "folds": np.bincount(data.folds).tolist(),
        "time_train": int((~data.test_mask).sum()),
        "time_test": int(data.test_mask.sum()),
        "best_model": res.best_model,
        "deployable": res.deployable,
        "deployable_reason": res.deployable_reason,
        "calibration_method": chosen,
        "temperature": float(getattr(bundle["model"], "temperature_", float("nan"))),
        "colab_present": res.colab_present,
        "ece": {m: cal[m]["ece"] for m in cal},
        "brier": {m: cal[m]["brier"] for m in cal},
        "calibrated_macro_f1": {m: cal[m]["macro_f1"] for m in cal},
        "model_path": str(C.MODEL_PATH.relative_to(PROJECT_ROOT).as_posix()),
        "fallback_model": BASELINE,
        "fallback_calibration": fallback_cal,
        "fallback_ece": base_ece,
        "fallback_macro_f1": float(
            res.metrics.query("setting == 'cv'").set_index("model").loc[BASELINE, "macro_f1"]
        ),
        "colab_problem": colab_problem,
    }
    return res


# --------------------------------------------------------------------------------------
# Artefacts
# --------------------------------------------------------------------------------------


def register(res: ClassificationResults) -> None:
    """Record Phase 6 facts under `rq3.*`."""
    for key, value in res.facts.items():
        registry.record(f"rq3.{key}", value, SOURCE, PRODUCER)
    registry.record("rq3.metrics", res.metrics.round(5), SOURCE, PRODUCER)
    registry.record("rq3.significance", res.significance.round(5), SOURCE, PRODUCER)


def save_tables(res: ClassificationResults) -> None:
    """reports/tables/rq3_*.{csv,md}."""
    style.save_table(res.metrics.round(4), "rq3_metrics")
    style.save_table(res.per_class.round(4), "rq3_per_class")
    style.save_table(res.significance.round(4), "rq3_significance")
    style.save_table(res.confusion.round(3), "rq3_confusion_best_model", index=True)
    style.save_table(res.confusions_top, "rq3_top_confusions")
    style.save_table(res.features, "rq3_top_features")
    style.save_table(
        pd.DataFrame(
            {
                m: [
                    res.calibration[m][x]
                    for x in ("ece", "brier", "accuracy", "mean_confidence", "macro_f1")
                ]
                for m in ("raw", "sigmoid", "temperature")
            },
            index=["ece", "brier", "accuracy", "mean_confidence", "macro_f1"],
        ).round(4),
        "rq3_calibration",
        index=True,
    )


def make_figures(res: ClassificationResults) -> dict[str, Any]:
    """Four figures."""
    d = res.data
    shown = res.metrics[~res.metrics["model"].str.contains("ablation")]
    cal = res.calibration
    support = {c: int((d.y == i).sum()) for i, c in enumerate(d.classes)}
    figs = {
        "fig_rq3_1_model_comparison": plots.plot_classifier_comparison(shown),
        "fig_rq3_2_confusion": plots.plot_confusion_heatmap(
            res.confusion, f"{res.best_model}: row-normalised confusion"
        ),
        "fig_rq3_3_per_class_f1": plots.plot_per_class_f1(
            res.per_class[
                res.per_class["class"].isin(d.classes)
                & ~res.per_class["model"].str.contains("ablation|majority")
            ][["model", "class", "f1"]],
            support,
        ),
        "fig_rq3_4_reliability": plots.plot_reliability(
            [
                (label, cal[m]["table"], cal[m]["ece"])
                for m, label in (
                    ("raw", "uncalibrated"),
                    ("sigmoid", "Platt (sigmoid)"),
                    ("temperature", "temperature scaling"),
                )
            ],
            cal["chosen"],
        ),
    }
    for name, fig in figs.items():
        style.save_figure(fig, name, str(load_config()["snapshot_date"]))
    return figs


def write_all(res: ClassificationResults) -> None:
    """Registry, tables, figures and docs/rq3_classification.md."""
    from aicidentia.models import classify_docs

    register(res)
    save_tables(res)
    make_figures(res)
    if res.reliability is not None:
        from aicidentia.models import reliability_outputs

        reliability_outputs.write_all(res.reliability)
    classify_docs.write_classification_doc(res, PROJECT_ROOT / "docs" / "rq3_classification.md")
