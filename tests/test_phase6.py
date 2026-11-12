"""Phase 6 tests: metrics, classifiers, calibration, Colab I/O.

Fixtures are tiny hand-made datasets with known answers (not analysis data).
"""

import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from aicidentia.config import PROJECT_ROOT
from aicidentia.evaluation import metrics as M
from aicidentia.models import classify as C

CLASSES = ["alpha", "beta", "gamma"]


# ------------------------------------------------------------------ metrics
def test_macro_f1_codes_matches_sklearn_including_absent_classes():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 3, 200), rng.integers(0, 3, 200)
    assert M.macro_f1_codes(y, p, 3) == pytest.approx(f1_score(y, p, average="macro"))
    assert M.macro_f1_codes(np.array([0, 0, 1]), np.array([0, 0, 1]), 4) == pytest.approx(
        2 / 4
    )  # two absent classes count as 0


def test_classification_report_table_structure():
    t = M.classification_report_table(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    assert t["class"].tolist() == ["a", "b", "macro avg", "weighted avg"]
    assert t.loc[0, "recall"] == 0.5 and t.loc[1, "precision"] == pytest.approx(2 / 3)
    assert t.loc[2, "support"] == 4


def test_bootstrap_ci_contains_point_and_is_deterministic():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    p = np.where(rng.random(300) < 0.7, y, rng.integers(0, 3, 300))
    point, lo, hi = M.bootstrap_metric(y, p, 3, 300, 7)
    assert lo < point < hi and (point, lo, hi) == M.bootstrap_metric(y, p, 3, 300, 7)


def test_paired_bootstrap_detects_a_clearly_better_model_and_a_tie():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 400)
    good = np.where(rng.random(400) < 0.9, y, rng.integers(0, 3, 400))
    bad = np.where(rng.random(400) < 0.4, y, rng.integers(0, 3, 400))
    res = M.paired_bootstrap_diff(y, good, bad, 3, 400, 3)
    assert res["diff"] > 0.2 and res["ci_low"] > 0 and res["p_value"] < 0.05
    tie = M.paired_bootstrap_diff(y, good, good.copy(), 3, 200, 3)
    assert tie["diff"] == 0 and tie["p_value"] == 1.0
    mc = M.mcnemar_test(y, good, bad)
    assert mc["a_only_correct"] > mc["b_only_correct"] and mc["p_value"] < 0.05


def test_calibration_metrics():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    perfect = np.array([[0.5, 0.5]] * 8)
    assert M.expected_calibration_error(np.array([0, 1] * 4), perfect) == pytest.approx(
        0.0
    )  # 50% confident, 50% right
    overconfident = np.array([[0.99, 0.01]] * 8)
    assert M.expected_calibration_error(y, overconfident) == pytest.approx(0.49, abs=1e-6)
    assert M.multiclass_brier(y, np.eye(2)[y]) == 0.0
    table = M.reliability_table(y, overconfident, 10)
    assert table["n"].sum() == 8 and table.loc[9, "n"] == 8


def test_kappa_landis_koch_and_agreement():
    a = ["x", "y", "x", "y", "x", "y"]
    assert M.cohens_kappa(a, a) == 1.0
    k, lo, hi = M.kappa_bootstrap(a, ["x", "y", "x", "y", "y", "x"], 300, 1)
    assert lo <= k <= hi
    assert [M.landis_koch(v) for v in (-0.1, 0.1, 0.3, 0.5, 0.7, 0.9)] == [
        "poor (below chance)",
        "slight",
        "fair",
        "moderate",
        "substantial",
        "almost perfect",
    ]
    pa = M.per_class_agreement(["x", "x", "y"], ["x", "y", "y"], ["x", "y"]).set_index("class")
    assert pa.loc["x", "agreement_of_reference"] == 0.5 and pa.loc["y", "agreement_of_other"] == 0.5


# ------------------------------------------------------------------ classifiers on a tiny dataset
THEMES = {
    "alpha": [
        "car crash road driver vehicle",
        "autonomous vehicle road crash",
        "driver killed car autopilot road",
        "vehicle crash highway driver",
    ],
    "beta": [
        "facial recognition arrest police match",
        "police face recognition wrongful arrest",
        "face match police arrest man",
        "recognition arrest police wrongly",
    ],
    "gamma": [
        "chatbot said harmful advice user",
        "chatbot user advice harmful teen",
        "harmful advice chatbot conversation",
        "chatbot conversation user harmful",
    ],
}


def _dataset(n_rep=6):
    texts, labels, years = [], [], []
    for r in range(n_rep):
        for k, sents in enumerate(THEMES.values()):
            for s in sents:
                texts.append(f"{s} case {r}")
                labels.append(k)
                years.append(2020 + (r % 6))
    y = np.array(labels)
    frame = pd.DataFrame(
        {
            "incident_id": np.arange(len(y)),
            "title": [f"title {i}" for i in range(len(y))],
            "description": texts,
            "text": texts,
            "year": years,
            "domain": [CLASSES[i] for i in y],
        }
    )
    return C.Dataset(frame, CLASSES, y, C.make_folds(y, 5, 42), (frame["year"] >= 2024).to_numpy())


def test_make_folds_is_stratified_and_deterministic():
    y = np.array([0] * 40 + [1] * 20 + [2] * 10)
    folds = C.make_folds(y, 5, 42)
    assert (folds == C.make_folds(y, 5, 42)).all() and set(folds) == set(range(5))
    for f in range(5):
        assert np.bincount(y[folds == f], minlength=3).tolist() == [8, 4, 2]


@pytest.mark.parametrize("kind", ["dummy", "tfidf_lr", "tfidf_svm"])
def test_fit_predict_text_models(kind):
    d = _dataset()
    spec = C.ModelSpec(kind, kind)
    x = C.features(spec, d)
    train = d.folds != 0
    pred, proba, info = C.fit_predict(spec, x[train], d.y[train], x[~train], 3)
    assert pred.shape == (int((~train).sum()),) and proba.shape == (len(pred), 3)
    if kind == "dummy":
        assert (pred == np.bincount(d.y[train]).argmax()).all()
    else:
        assert (pred == d.y[~train]).mean() > 0.9  # three separable themes
        assert info["best_params"]
    if kind == "tfidf_lr":
        assert np.allclose(proba.sum(axis=1), 1.0)


def test_embedding_model_on_precomputed_vectors():
    d = _dataset()
    rng = np.random.default_rng(0)
    centres = np.eye(3) * 3
    d.embeddings["fake"] = (centres[d.y] + rng.normal(0, 0.3, (len(d.y), 3))).astype("float32")
    spec = C.ModelSpec("emb", "emb_lr", "fake")
    x = C.features(spec, d)
    train = d.folds != 0
    pred, proba, _ = C.fit_predict(spec, x[train], d.y[train], x[~train], 3)
    assert (pred == d.y[~train]).mean() > 0.95 and np.allclose(proba.sum(axis=1), 1.0)


def test_temperature_scaling_preserves_predictions_and_fixes_overconfidence():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 6))
    y = (x[:, 0] + 0.7 * rng.normal(size=600) > 0).astype(
        int
    )  # noisy labels: a flexible model is over-confident
    base = LogisticRegression(C=1000, max_iter=5000)
    raw = base.fit(x[:400], y[:400]).predict_proba(x[400:])
    scaled_model = C.TemperatureScaled(LogisticRegression(C=1000, max_iter=5000), 3, 0).fit(
        x[:400], y[:400]
    )
    scaled = scaled_model.predict_proba(x[400:])
    assert np.allclose(scaled.sum(axis=1), 1.0) and scaled_model.temperature_ > 1.0  # softens
    assert (scaled.argmax(1) == raw.argmax(1)).all()  # class unchanged
    ece_raw = M.expected_calibration_error(y[400:], raw)
    ece_scaled = M.expected_calibration_error(y[400:], scaled)
    assert ece_scaled < ece_raw
    assert (scaled_model.predict(x[400:]) == scaled.argmax(1)).all()


def test_calibrate_model_dispatch():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(300, 4)), rng.integers(0, 3, 300)
    base = LogisticRegression(max_iter=500)
    assert C.calibrate_model(base, None, x, y, 0) is base
    assert hasattr(C.calibrate_model(base, "temperature", x, y, 0), "temperature_")
    assert C.calibrate_model(base, "sigmoid", x, y, 0).predict_proba(x[:5]).shape == (5, 3)
    with pytest.raises(ValueError):
        C.calibrate_model(base, "isotonic-ish", x, y, 0)


def test_confusion_and_top_confusions():
    y_true, y_pred = np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 1, 0])
    cm = C.confusion(y_true, y_pred, CLASSES)
    assert cm.loc["alpha", "beta"] == 1 and cm.loc["gamma", "alpha"] == 1
    norm = C.confusion(y_true, y_pred, CLASSES, normalise=True)
    assert norm.loc["alpha"].sum() == pytest.approx(1.0) and norm.loc["beta", "beta"] == 1.0
    d = _dataset()
    oof = pd.DataFrame(
        {
            "incident_id": [0, 1, 2],
            "y_true": [0, 0, 1],
            "y_pred": [1, 1, 2],
            "proba_0": [0.1] * 3,
            "proba_1": [0.9, 0.6, 0.1],
            "proba_2": [0.0, 0.0, 0.8],
        }
    )
    top = C.top_confusions(oof, d, n_pairs=2, n_examples=1)
    assert (
        top.iloc[0]["true_domain"] == "alpha"
        and top.iloc[0]["count"] == 2
        and top.iloc[0]["examples"].startswith("#0")
    )


def test_top_features_returns_terms_per_class():
    d = _dataset()
    out = C.top_features(C.ModelSpec("lr", "tfidf_lr"), d, n=3)
    assert out["domain"].tolist() == CLASSES
    assert (
        "chatbot" in out.set_index("domain").loc["gamma", "top_terms"]
        and "crash" in out.set_index("domain").loc["alpha", "top_terms"]
    )


def test_final_model_bundle_roundtrip_and_prediction(tmp_path):
    d = _dataset()
    bundle = C.fit_final_model(C.ModelSpec("lr", "tfidf_lr"), d, calibrate="temperature")
    path = C.save_model(bundle, tmp_path / "m.joblib")
    import joblib

    loaded = joblib.load(path)
    out = C.predict_domains(
        loaded,
        ["chatbot gave harmful advice to a user", "driver killed in autopilot crash on road"],
    )
    assert out["predicted_domain"].tolist() == ["gamma", "alpha"]
    assert out["confidence"].between(0, 1).all() and np.allclose(
        out[[f"p_{c}" for c in CLASSES]].sum(axis=1), 1.0
    )
    assert loaded["calibration"] == "temperature" and loaded["trained_on"] == len(d.y)


# ------------------------------------------------------------------ Colab input / output
def test_export_colab_input_and_load_predictions_validation(tmp_path):
    d = _dataset()
    path = C.export_colab_input(d, tmp_path / "in.csv")
    exported = pd.read_csv(path)
    assert list(exported.columns) == [
        "incident_id",
        "text",
        "year",
        "label",
        "fold",
        "is_test_time_split",
    ]
    assert (exported["fold"].to_numpy() == d.folds).all() and exported["label"].isin(CLASSES).all()

    assert C.load_colab_predictions(d, tmp_path / "missing.csv") is None
    rows = []
    for setting, mask in (("cv", np.ones(len(d.y), bool)), ("time", d.test_mask)):
        sub = d.frame[mask]
        for i, row in sub.iterrows():
            r = {
                "incident_id": row["incident_id"],
                "setting": setting,
                "fold": int(d.folds[i]) if setting == "cv" else -1,
                "y_true": row["domain"],
                "y_pred": row["domain"],
            }
            r.update({f"proba_{c}": float(c == row["domain"]) for c in CLASSES})
            rows.append(r)
    good = pd.DataFrame(rows)
    good.to_csv(tmp_path / "ok.csv", index=False)
    cv, time = C.load_colab_predictions(d, tmp_path / "ok.csv")
    assert (
        len(cv) == len(d.y)
        and len(time) == int(d.test_mask.sum())
        and (cv["y_true"] == cv["y_pred"]).all()
    )
    bad_folds = good.copy()
    bad_folds.loc[bad_folds["setting"] == "cv", "fold"] = 0
    bad_folds.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="Folds"):
        C.load_colab_predictions(d, tmp_path / "bad.csv")
    missing = good.iloc[1:]
    missing.to_csv(tmp_path / "miss.csv", index=False)
    with pytest.raises(ValueError, match="cover"):
        C.load_colab_predictions(d, tmp_path / "miss.csv")


def _colab_module():
    spec = importlib.util.spec_from_file_location(
        "colab_code_test", PROJECT_ROOT / "scripts" / "colab_distilbert_code.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["colab_code_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_colab_code_class_weights_and_seeding():
    mod = _colab_module()
    w = mod.balanced_class_weights(np.array([0, 0, 0, 1]), 3).numpy()
    assert (
        w[0] == pytest.approx(4 / (3 * 3)) and w[1] == pytest.approx(4 / 3) and w[2] == 0.0
    )  # absent class weight 0
    mod.set_seed(1)
    a = mod.torch.rand(3)
    mod.set_seed(1)
    assert (a == mod.torch.rand(3)).all()


def test_colab_notebook_embeds_the_code_verbatim():
    import json

    nb = json.loads(
        (PROJECT_ROOT / "notebooks" / "colab_distilbert_finetune.ipynb").read_text(encoding="utf-8")
    )
    code = (PROJECT_ROOT / "scripts" / "colab_distilbert_code.py").read_text(encoding="utf-8")
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert code.strip() in [s.strip() for s in sources], (
        "re-generate the notebook after editing colab_distilbert_code.py"
    )


def test_threaded_decorator_uses_threads_not_processes():
    """Regression: process workers import torch from a notebook kernel and exhaust Windows' commit limit."""
    from joblib.parallel import get_active_backend

    @C._threaded
    def inside() -> str:
        return type(get_active_backend()[0]).__name__

    assert inside() == "ThreadingBackend"
    assert (
        type(get_active_backend()[0]).__name__ != "ThreadingBackend"
    )  # the setting does not leak out
