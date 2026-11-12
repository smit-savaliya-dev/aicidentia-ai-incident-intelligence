"""RQ3 - classify incidents into MIT risk domains (TF-IDF + linear models, sentence embeddings, DistilBERT).

Target: MIT Risk Domain (machine-generated labels; see docs/data_audit.md D1). Input: `text` (title +
description). Every model is evaluated with the same stratified 5-fold split (saved so the Colab DistilBERT
run uses identical folds) and, separately, with a time split (train <= 2023, test >= 2024). Hyperparameters
are chosen by an inner 3-fold CV on the training part only, so test folds never influence tuning.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_config
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from aicidentia import cache
from aicidentia.config import PROJECT_ROOT, load_config
from aicidentia.data import clean
from aicidentia.features.text import embed_sentences

logger = logging.getLogger(__name__)
TEST_FROM_YEAR = 2024
N_FOLDS = 5
MODEL_PATH = PROJECT_ROOT / "models" / "rq3_domain_classifier.joblib"
FALLBACK_MODEL_PATH = (
    PROJECT_ROOT / "models" / "rq3_domain_classifier_fallback.joblib"
)  # TF-IDF + LR, no download
COLAB_PREDICTIONS = PROJECT_ROOT / "reports" / "colab" / "rq3_distilbert_predictions.csv"


@dataclass
class Dataset:
    """The labelled incidents: ids, text, integer-coded labels, years, saved CV folds."""

    frame: (
        pd.DataFrame
    )  # incident_id, title, description, text, year, domain (sorted by incident_id)
    classes: list[str]
    y: np.ndarray
    folds: np.ndarray
    test_mask: np.ndarray  # time split: True for incidents from TEST_FROM_YEAR on
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def fingerprint(self) -> str:
        return cache.hash_parts(
            self.frame["incident_id"].tolist(), self.frame["text"].tolist(), self.folds.tolist()
        )


@dataclass(frozen=True)
class ModelSpec:
    """A model to evaluate: `kind` is dummy | tfidf_lr | tfidf_svm | emb_lr."""

    name: str
    kind: str
    embedding_key: str | None = None
    text_column: str = "text"


def make_folds(y: np.ndarray, n_splits: int = N_FOLDS, seed: int = 42) -> np.ndarray:
    """Stratified fold id (0..n_splits-1) per item."""
    folds = np.empty(len(y), dtype=int)
    for fold, (_, test) in enumerate(
        StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    ):
        folds[test] = fold
    return folds


def load_dataset(with_embeddings: bool = True) -> Dataset:
    """MIT-labelled incidents from data/processed, with folds and time split; embeddings come from the Phase 3 cache."""
    cfg = load_config()
    inc = clean.load_processed("incidents_clean")
    labelled = inc[inc["mit_risk_domain"].notna()].sort_values("incident_id").reset_index(drop=True)
    classes = sorted(labelled["mit_risk_domain"].unique())
    y = labelled["mit_risk_domain"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    frame = labelled[
        ["incident_id", "title", "description", "text", "year", "mit_risk_domain"]
    ].rename(columns={"mit_risk_domain": "domain"})
    data = Dataset(
        frame,
        classes,
        y,
        make_folds(y, N_FOLDS, int(cfg["random_seed"])),
        (frame["year"] >= TEST_FROM_YEAR).to_numpy(),
    )
    if with_embeddings:
        by_id = {int(i): k for k, i in enumerate(inc["incident_id"])}
        rows = [by_id[int(i)] for i in frame["incident_id"]]
        for key in ("sentence_embedding_small", "sentence_embedding_large"):
            full = embed_sentences(inc["text"].tolist(), cfg["models"][key])
            data.embeddings[key] = full[rows]
    return data


def features(spec: ModelSpec, data: Dataset) -> np.ndarray:
    """Input matrix for a model: raw texts (TF-IDF models) or embeddings (embedding models)."""
    if spec.kind == "emb_lr":
        return data.embeddings[spec.embedding_key]
    return data.frame[spec.text_column].to_numpy(dtype=object)


def _threaded(fn: Any) -> Any:
    """Run `fn` with scikit-learn's inner parallelism on threads, not processes.

    Process workers (the default) import every module the parent has loaded; in a notebook kernel that has
    imported PyTorch, 16 workers each map the CUDA libraries and exhaust the Windows commit limit
    (WinError 1455). The grids here are tiny, so threads cost nothing and avoid the problem.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with parallel_config(backend="threading"):
            return fn(*args, **kwargs)

    return wrapper


def _inner_cv(seed: int) -> StratifiedKFold:
    return StratifiedKFold(3, shuffle=True, random_state=seed)


class TemperatureScaled(ClassifierMixin, BaseEstimator):
    """Single-parameter calibration of a probabilistic classifier (temperature scaling).

    The temperature T is fitted on out-of-fold probabilities from an inner CV on the training data (minimising
    the negative log-likelihood of softmax(log p / T)); the final base model is then fitted on all the training
    data and its probabilities are rescaled by T. T > 1 softens over-confident models, T < 1 sharpens
    under-confident ones. The predicted class is unchanged by the scaling.
    """

    def __init__(self, estimator: Any, cv: int = 3, random_state: int = 42) -> None:
        self.estimator = estimator
        self.cv = cv
        self.random_state = random_state

    def fit(self, X: Any, y: np.ndarray) -> TemperatureScaled:  # noqa: N803 - scikit-learn convention
        folds = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        proba = cross_val_predict(clone(self.estimator), X, y, cv=folds, method="predict_proba")
        logits = np.log(np.clip(proba, 1e-9, 1.0))

        def nll(log_t: float) -> float:
            p = softmax(logits / np.exp(log_t), axis=1)
            return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())

        self.temperature_ = float(
            np.exp(minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded").x)
        )
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:  # noqa: N803
        logits = np.log(np.clip(self.estimator_.predict_proba(X), 1e-9, 1.0))
        return softmax(logits / self.temperature_, axis=1)

    def predict(self, X: Any) -> np.ndarray:  # noqa: N803
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def calibrate_model(model: Any, method: str | bool | None, x: Any, y: np.ndarray, seed: int) -> Any:
    """Wrap `model` in a calibrator fitted on (x, y): 'sigmoid' (Platt, OvR + normalisation) or 'temperature'."""
    if not method:
        return model
    if method is True or method == "sigmoid":
        return CalibratedClassifierCV(clone(model), method="sigmoid", cv=_inner_cv(seed)).fit(x, y)
    if method == "temperature":
        return TemperatureScaled(clone(model), 3, seed).fit(x, y)
    raise ValueError(f"unknown calibration method {method!r}")


def build_estimator(spec: ModelSpec, seed: int = 42) -> Any:
    """Estimator for a spec; tuned models are wrapped in a GridSearchCV over the regularisation strength."""
    scoring = "f1_macro"
    if spec.kind == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if spec.kind == "tfidf_lr":
        pipe = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed),
                ),
            ]
        )
        return GridSearchCV(
            pipe, {"clf__C": [0.3, 1, 3, 10, 30]}, scoring=scoring, cv=_inner_cv(seed), n_jobs=-1
        )
    if spec.kind == "tfidf_svm":
        pipe = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"
                    ),
                ),
                ("clf", LinearSVC(class_weight="balanced", random_state=seed, max_iter=10000)),
            ]
        )
        return GridSearchCV(
            pipe, {"clf__C": [0.03, 0.1, 0.3, 1]}, scoring=scoring, cv=_inner_cv(seed), n_jobs=-1
        )
    if spec.kind == "emb_lr":
        clf = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed)
        return GridSearchCV(
            clf, {"C": [0.3, 1, 3, 10, 30]}, scoring=scoring, cv=_inner_cv(seed), n_jobs=-1
        )
    raise ValueError(f"unknown model kind {spec.kind!r}")


@_threaded
def fit_predict(
    spec: ModelSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    n_classes: int,
    seed: int = 42,
    calibrate: str | bool | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit on the training part and predict the test part.

    Returns (predicted codes, probability matrix (test x classes), info). Probabilities are the model's own
    (logistic regression), uniform-prior (dummy), or empty NaN for an uncalibrated linear SVM. With
    `calibrate`, the tuned estimator is wrapped in sigmoid (Platt) calibration fitted by 3-fold CV inside the
    training part.
    """
    estimator = build_estimator(spec, seed)
    estimator.fit(x_train, y_train)
    info: dict[str, Any] = {}
    model = estimator
    if isinstance(estimator, GridSearchCV):
        info["best_params"] = estimator.best_params_
        model = estimator.best_estimator_
    model = calibrate_model(model, calibrate, x_train, y_train, seed)
    pred = np.asarray(model.predict(x_test))
    proba = np.full((len(x_test), n_classes), np.nan)
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(x_test)
        proba[:, [int(c) for c in model.classes_]] = raw
        proba = np.nan_to_num(proba, nan=0.0)
    return pred, proba, info


def cross_validate(
    spec: ModelSpec, data: Dataset, calibrate: str | bool | None = None, seed: int = 42
) -> pd.DataFrame:
    """Out-of-fold predictions over the saved folds: incident_id, fold, y_true, y_pred, proba_<class>."""
    x = features(spec, data)

    def compute() -> pd.DataFrame:
        parts = []
        for fold in range(N_FOLDS):
            test = data.folds == fold
            pred, proba, info = fit_predict(
                spec, x[~test], data.y[~test], x[test], data.n_classes, seed, calibrate
            )
            part = pd.DataFrame(
                {
                    "incident_id": data.frame.loc[test, "incident_id"].to_numpy(),
                    "fold": fold,
                    "y_true": data.y[test],
                    "y_pred": pred,
                }
            )
            for k in range(data.n_classes):
                part[f"proba_{k}"] = proba[:, k]
            part["best_params"] = str(info.get("best_params", ""))
            parts.append(part)
        return pd.concat(parts, ignore_index=True)

    key = (
        data.fingerprint,
        spec.name,
        spec.kind,
        spec.embedding_key,
        spec.text_column,
        calibrate,
        seed,
    )
    oof = cache.cached("rq3cv", key, compute, version="1")
    return oof.sort_values("incident_id").reset_index(drop=True)


def time_split_predict(
    spec: ModelSpec, data: Dataset, calibrate: str | bool | None = None, seed: int = 42
) -> pd.DataFrame:
    """Train on incidents before TEST_FROM_YEAR, predict those from that year on."""
    x = features(spec, data)
    test = data.test_mask

    def compute() -> pd.DataFrame:
        pred, proba, info = fit_predict(
            spec, x[~test], data.y[~test], x[test], data.n_classes, seed, calibrate
        )
        out = pd.DataFrame(
            {
                "incident_id": data.frame.loc[test, "incident_id"].to_numpy(),
                "fold": -1,
                "y_true": data.y[test],
                "y_pred": pred,
            }
        )
        for k in range(data.n_classes):
            out[f"proba_{k}"] = proba[:, k]
        out["best_params"] = str(info.get("best_params", ""))
        return out

    key = (
        data.fingerprint,
        spec.name,
        spec.kind,
        spec.embedding_key,
        spec.text_column,
        calibrate,
        seed,
        "time",
    )
    return cache.cached("rq3time", key, compute, version="1").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Colab / DistilBERT
# --------------------------------------------------------------------------------------


def export_colab_input(data: Dataset, path: Path) -> Path:
    """CSV for the Colab notebook: incident_id, text, label, fold (identical folds), is_test (time split), year."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = data.frame[["incident_id", "text", "year"]].copy()
    out["label"] = [data.classes[i] for i in data.y]
    out["fold"] = data.folds
    out["is_test_time_split"] = data.test_mask.astype(int)
    out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


def load_colab_predictions(
    data: Dataset, path: Path = COLAB_PREDICTIONS
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """DistilBERT predictions returned from Colab, split into (cross-validation, time-split) frames.

    Expected columns: incident_id, setting ('cv' | 'time'), fold, y_true (label name), y_pred (label name),
    and optionally proba_<label name>. Returns None if the file is absent. Validates that the incidents, true
    labels and folds match this repository's, because identical folds are what makes the comparison fair.
    """
    if not path.exists():
        return None
    raw = pd.read_csv(path)
    code = {c: i for i, c in enumerate(data.classes)}
    frame = data.frame.assign(label=data.y, fold_ref=data.folds, test_ref=data.test_mask).set_index(
        "incident_id"
    )
    out = {}
    for setting in ("cv", "time"):
        sub = raw[raw["setting"] == setting].copy()
        expected = frame.index if setting == "cv" else frame.index[frame["test_ref"]]
        if set(sub["incident_id"]) != set(expected):
            raise ValueError(
                f"Colab predictions ({setting}) do not cover exactly the expected incidents"
            )
        sub["y_true"] = sub["y_true"].map(code)
        sub["y_pred"] = sub["y_pred"].map(code)
        if sub["y_true"].isna().any() or sub["y_pred"].isna().any():
            raise ValueError("Unknown label names in the Colab predictions file")
        if (sub["y_true"].to_numpy() != frame.loc[sub["incident_id"], "label"].to_numpy()).any():
            raise ValueError("True labels in the Colab file differ from this repository's labels")
        if (
            setting == "cv"
            and (
                sub["fold"].to_numpy() != frame.loc[sub["incident_id"], "fold_ref"].to_numpy()
            ).any()
        ):
            raise ValueError("Folds in the Colab file differ from this repository's folds")
        for cls, k in code.items():
            sub[f"proba_{k}"] = sub[f"proba_{cls}"] if f"proba_{cls}" in sub else np.nan
        out[setting] = (
            sub[
                [
                    "incident_id",
                    "fold",
                    "y_true",
                    "y_pred",
                    *[f"proba_{k}" for k in range(len(code))],
                ]
            ]
            .astype({"y_true": int, "y_pred": int})
            .sort_values("incident_id")
            .reset_index(drop=True)
        )
    return out["cv"], out["time"]


# --------------------------------------------------------------------------------------
# Error analysis
# --------------------------------------------------------------------------------------


def confusion(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], normalise: bool = False
) -> pd.DataFrame:
    """Confusion matrix (rows = true class, columns = predicted); optionally row-normalised."""
    n = len(classes)
    cm = np.bincount(y_true * n + y_pred, minlength=n * n).reshape(n, n).astype(float)
    if normalise:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    return pd.DataFrame(cm, index=classes, columns=classes)


def top_confusions(
    oof: pd.DataFrame,
    data: Dataset,
    n_pairs: int = 6,
    n_examples: int = 3,
    exclude_examples: set[int] | None = None,
) -> pd.DataFrame:
    """Most frequent (true -> predicted) errors with example incidents (the most confidently wrong, if probabilities exist).

    Counts use every error; incidents in `exclude_examples` (the blind label-audit sample) are never shown as
    examples, because an example reveals that incident's MIT label.
    """
    wrong = oof[oof["y_true"] != oof["y_pred"]].copy()
    shown = wrong[~wrong["incident_id"].isin(exclude_examples or set())]
    titles = data.frame.set_index("incident_id")["title"]
    pairs = wrong.groupby(["y_true", "y_pred"]).size().sort_values(ascending=False).head(n_pairs)
    rows = []
    for (t, p), count in pairs.items():
        sub = shown[(shown["y_true"] == t) & (shown["y_pred"] == p)].copy()
        sub["conf"] = sub[f"proba_{p}"].fillna(0)
        sub = sub.sort_values("conf", ascending=False).head(n_examples)
        rows.append(
            {
                "true_domain": data.classes[t],
                "predicted_as": data.classes[p],
                "count": int(count),
                "share_of_true_class": float(count / (data.y == t).sum()),
                "examples": " || ".join(f"#{i}: {titles[i]}" for i in sub["incident_id"]),
            }
        )
    return pd.DataFrame(rows)


@_threaded
def top_features(spec: ModelSpec, data: Dataset, n: int = 10, seed: int = 42) -> pd.DataFrame:
    """Highest-weight TF-IDF terms per class for a linear model fitted on all labelled incidents."""
    estimator = build_estimator(spec, seed).fit(features(spec, data), data.y)
    best = estimator.best_estimator_
    vocab = np.array(best.named_steps["tfidf"].get_feature_names_out())
    coef = best.named_steps["clf"].coef_
    rows = []
    for k, cls in enumerate(data.classes):
        top = np.argsort(coef[k])[::-1][:n]
        rows.append({"model": spec.name, "domain": cls, "top_terms": ", ".join(vocab[top])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Deployable model
# --------------------------------------------------------------------------------------


@_threaded
def fit_final_model(
    spec: ModelSpec, data: Dataset, calibrate: str | bool | None = "temperature", seed: int = 42
) -> dict[str, Any]:
    """Fit the deployable classifier on all labelled incidents and return a bundle for joblib."""
    x = features(spec, data)
    estimator = build_estimator(spec, seed).fit(x, data.y)
    model = estimator.best_estimator_ if isinstance(estimator, GridSearchCV) else estimator
    model = calibrate_model(model, calibrate, x, data.y, seed)
    return {
        "model": model,
        "classes": data.classes,
        "spec": spec,
        "calibration": calibrate or None,
        "embedding_model": load_config()["models"][spec.embedding_key]
        if spec.embedding_key
        else None,
        "trained_on": int(len(data.y)),
        "snapshot_date": str(load_config()["snapshot_date"]),
        "best_params": getattr(estimator, "best_params_", {}),
    }


def save_model(bundle: dict[str, Any], path: Path = MODEL_PATH) -> Path:
    """Persist the bundle in models/ (git-ignored)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def predict_domains(
    bundle: dict[str, Any], texts: list[str], use_cache: bool = True, local_files_only: bool = False
) -> pd.DataFrame:
    """Predicted domain, calibrated confidence and the full probability vector for each text.

    `use_cache=False` and `local_files_only=True` are used by the dashboard (no disk cache per query,
    never download the embedding model).
    """
    x: Any = np.array(texts, dtype=object)
    if bundle["embedding_model"]:
        x = embed_sentences(
            texts, bundle["embedding_model"], use_cache=use_cache, local_files_only=local_files_only
        )
    proba = bundle["model"].predict_proba(x)
    classes = bundle["classes"]
    best = proba.argmax(axis=1)
    out = pd.DataFrame(
        {"predicted_domain": [classes[i] for i in best], "confidence": proba.max(axis=1)}
    )
    for k, cls in enumerate(classes):
        out[f"p_{cls}"] = proba[:, k]
    return out
