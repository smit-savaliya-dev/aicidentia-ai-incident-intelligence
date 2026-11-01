"""Phase 3 tests: cache, tokeniser, corpus, metrics, the three model families, selection rule, label file.

Model-level tests use tiny hand-written corpora with two obvious themes (test fixtures, not analysis
data) to check shapes, determinism and that the code separates themes.
"""

import numpy as np
import pandas as pd
import pytest

from aicidentia import cache
from aicidentia.features import text as T
from aicidentia.models import topic_experiment as E
from aicidentia.models import topic_outputs as O
from aicidentia.models import topics as M

CAR = [
    "tesla autopilot car crashed into truck driver killed",
    "self driving car hit pedestrian road crash driver",
    "autonomous vehicle road crash driver injured truck",
    "car driver autopilot road crash vehicle pedestrian",
    "vehicle truck crash road driver autopilot killed",
    "pedestrian killed autonomous car road crash vehicle",
]
FACE = [
    "facial recognition wrongly arrested man police face match",
    "police face recognition match wrongful arrest man",
    "face match police arrest wrongly identified man recognition",
    "recognition police arrest wrongful face identified match",
    "man arrested police wrongly face recognition identified",
    "wrongful arrest face match police man recognition",
]


@pytest.fixture
def small_settings(monkeypatch):
    settings = {
        "min_df": 1,
        "max_df": 1.0,
        "min_tokens": 1,
        "top_n": 5,
        "domain_stopwords": ["allegedly"],
        "lda": {"passes": 10, "iterations": 50},
        "nmf": {"max_iter": 300},
        "bertopic": {
            "min_samples": 2,
            "umap": {"n_neighbors": 5, "n_components": 2, "min_dist": 0.0, "metric": "cosine"},
            "outlier_strategy": "embeddings",
            "embedding_models": [],
            "min_cluster_size": [5],
        },
    }
    for module in ("aicidentia.features.text", "aicidentia.models.topics"):
        monkeypatch.setattr(f"{module}.topic_settings", lambda: settings)
    return settings


@pytest.fixture
def corpus(small_settings):
    texts = CAR * 3 + FACE * 3
    return T.build_corpus(range(len(texts)), texts)


# ------------------------------------------------------------------ cache
def test_hash_parts_is_stable_and_sensitive():
    assert cache.hash_parts("a", {"x": 1, "y": 2}) == cache.hash_parts("a", {"y": 2, "x": 1})
    assert cache.hash_parts("a", 1) != cache.hash_parts("a", 2)
    assert cache.hash_texts(["a", "b"]) != cache.hash_texts(["b", "a"])
    assert cache.hash_texts(["ab"]) != cache.hash_texts(["a", "b"])


def test_cached_computes_once_and_respects_key_and_force(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "cache_dir", lambda: tmp_path)
    calls = []

    def compute():
        calls.append(1)
        return {"value": len(calls)}

    assert cache.cached("demo", (1,), compute)["value"] == 1
    assert cache.cached("demo", (1,), compute)["value"] == 1  # hit
    assert len(calls) == 1
    assert cache.cached("demo", (2,), compute)["value"] == 2  # different parts
    assert cache.cached("demo", (1,), compute, version="2")["value"] == 3  # different version
    assert cache.cached("demo", (1,), compute, force=True)["value"] == 4
    assert len(list(tmp_path.glob("demo-*.joblib"))) == 3


# ------------------------------------------------------------------ text pipeline
def test_tokenize_stopwords_lemmas_and_domain_words(small_settings):
    tokens = T.tokenize(["The AI system allegedly crashed cars; Teslas were arrested."])[0]
    assert "allegedly" not in tokens and "the" not in tokens and "were" not in tokens
    assert "crash" in tokens and "car" in tokens  # lemmatised
    assert all(len(t) >= 3 for t in tokens)


def test_build_corpus_alignment_and_filtering(monkeypatch, small_settings):
    monkeypatch.setitem(small_settings, "min_df", 2)
    texts = ["robot fails badly", "robot fails again", "unique zeppelin"]
    corpus = T.build_corpus([10, 11, 12], texts)
    assert corpus.incident_ids == [10, 11, 12]
    assert set(corpus.vocabulary) == {"robot", "fail"}  # rare words dropped by min_df
    assert corpus.tokens[2] == [] and corpus.short_docs == [2]
    assert corpus.docs[0] == "robot fail"
    assert len(corpus.bow) == 3
    assert corpus.fingerprint == T.build_corpus([10, 11, 12], texts).fingerprint


def test_build_tfidf_uses_fixed_vocabulary(corpus):
    matrix, _ = T.build_tfidf(corpus.tokens, corpus.vocabulary)
    assert matrix.shape == (len(corpus.tokens), len(corpus.vocabulary))
    assert matrix.min() >= 0


# ------------------------------------------------------------------ metrics
def test_topic_diversity():
    assert M.topic_diversity([["a", "b"], ["c", "d"]]) == 1.0
    assert M.topic_diversity([["a", "b"], ["a", "b"]]) == 0.5
    assert np.isnan(M.topic_diversity([]))


def test_matched_jaccard_and_stability():
    a = [["x", "y"], ["p", "q"]]
    assert (
        M.matched_jaccard(a, [["q", "p"], ["y", "x"]]) == 1.0
    )  # order of topics and words does not matter
    assert M.matched_jaccard(a, [["u", "v"], ["w", "z"]]) == 0.0
    assert M.matched_jaccard(a, [["x", "y"]]) == pytest.approx(0.5)  # unequal counts are penalised
    fits = [M.TopicFit("LDA", {}, s, a, np.array([0]), None) for s in (1, 2, 3)]
    assert M.stability(fits) == 1.0
    assert np.isnan(M.stability(fits[:1]))


def test_coherence_prefers_coherent_topics(corpus):
    coherent = [["car", "crash", "road", "driver"], ["face", "police", "arrest", "match"]]
    mixed = [["car", "police", "road", "match"], ["face", "crash", "arrest", "driver"]]
    assert M.topic_coherence(coherent, corpus, "c_v") > M.topic_coherence(mixed, corpus, "c_v")
    assert np.isnan(M.topic_coherence([], corpus))


# ------------------------------------------------------------------ model families
@pytest.mark.parametrize("fit_fn", [M.fit_lda, M.fit_nmf])
def test_lda_and_nmf_separate_two_themes_deterministically(corpus, fit_fn):
    fit = fit_fn(corpus, 2, 42)
    again = fit_fn(corpus, 2, 42)
    assert fit.terms == again.terms and (fit.assignments == again.assignments).all()
    assert fit.n_topics == 2 and fit.weights.shape == (len(corpus.docs), 2)
    assert np.allclose(fit.weights.sum(axis=1), 1.0)
    assert all(len(t) == 5 for t in fit.terms)
    half = len(corpus.docs) // 2
    first, second = fit.assignments[:half], fit.assignments[half:]
    assert len(set(first)) == 1 and len(set(second)) == 1 and first[0] != second[0]


def test_assign_topics_and_representatives(corpus):
    fit = M.fit_nmf(corpus, 2, 1)
    assignments = M.assign_topics(fit, corpus)
    assert list(assignments.columns) == ["incident_id", "topic", "topic_id", "weight"]
    assert assignments["weight"].between(0, 1).all()
    reps = M.representative_docs(fit, n=3)
    assert set(reps) == {0, 1} and all(len(v) == 3 for v in reps.values())
    for topic, rows in reps.items():
        assert (fit.assignments[rows] == topic).all()


def test_fit_bertopic_smoke_assigns_every_document(small_settings):
    rng = np.random.default_rng(0)
    # Embedding fixture: two well-separated blobs paired with the two themes.
    emb = np.vstack([rng.normal(1, 0.05, (30, 8)), rng.normal(-1, 0.05, (30, 8))]).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    texts = (CAR * 5)[:30] + (FACE * 5)[:30]
    corpus = T.build_corpus(range(60), texts)
    fit = M.fit_bertopic(corpus, emb, seed=42, min_cluster_size=5, embedding_name="fixture")
    assert fit.family == "BERTopic" and fit.n_topics >= 2
    assert (fit.assignments >= 0).all()  # outliers were reassigned
    assert all(w in set(corpus.vocabulary) for t in fit.terms for w in t)  # shared vocabulary
    assert len(fit.topic_ids) == fit.n_topics
    reps = M.representative_docs(fit, emb, n=2)
    assert all(len(v) <= 2 for v in reps.values())


# ------------------------------------------------------------------ selection rule
def _summary():
    rows = [
        # label, contender, family, topics, c_v, npmi, diversity, stability, eligible
        ("LDA k=10", "LDA", "LDA", 10, 0.50, 0.00, 0.90, 0.30, True),
        ("LDA k=12", "LDA", "LDA", 12, 0.48, -0.02, 0.92, 0.35, True),
        (
            "LDA k=40",
            "LDA",
            "LDA",
            40,
            0.70,
            0.10,
            0.99,
            0.50,
            False,
        ),  # ineligible, would win otherwise
        ("NMF k=10", "NMF", "NMF", 10, 0.55, 0.03, 0.95, 0.40, True),
        ("BERT mcs=20", "BERTopic[X]", "BERTopic", 20, 0.60, 0.05, 0.97, 0.80, True),
        ("BERT mcs=30", "BERTopic[X]", "BERTopic", 18, 0.52, 0.01, 0.80, 0.70, True),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "label",
            "contender",
            "family",
            "n_topics_mean",
            "c_v_mean",
            "c_npmi_mean",
            "diversity_mean",
            "stability",
            "eligible",
        ],
    )
    for m in ("c_v", "c_npmi", "diversity"):
        df[f"{m}_sd"] = 0.01
    return df


def test_select_model_applies_eligibility_and_ranks():
    sel = E.select_model(_summary())
    best = dict(
        zip(sel.best_per_contender["contender"], sel.best_per_contender["label"], strict=True)
    )
    assert best == {
        "LDA": "LDA k=10",
        "NMF": "NMF k=10",
        "BERTopic[X]": "BERT mcs=20",
    }  # k=40 is ineligible
    assert sel.winner == "BERTopic[X]" and sel.winner_config == "BERT mcs=20"
    assert set(sel.sensitivity) >= {
        "C_v only",
        "NPMI only",
        "coherence only (C_v + NPMI)",
        "without stability",
    }
    assert sel.ranks["mean_rank"].is_monotonic_increasing


def test_select_model_sensitivity_can_differ():
    s = _summary()
    s.loc[s["label"] == "NMF k=10", ["c_v_mean", "c_npmi_mean"]] = [0.9, 0.2]  # best coherence
    s.loc[s["label"] == "NMF k=10", "stability"] = 0.01  # but unstable
    sel = E.select_model(s)
    assert sel.sensitivity["coherence only (C_v + NPMI)"] == "NMF"
    assert sel.sensitivity["all four metrics (the rule)"] != "NMF"


def test_summarise_means_sds_and_eligibility(monkeypatch):
    monkeypatch.setattr(E, "topic_settings", lambda: {"eligible_topics": [8, 25]})
    cfg = E.Config("LDA", "LDA k=10", "LDA", 10)
    runs = pd.DataFrame(
        {
            "label": ["LDA k=10"] * 2,
            "family": "LDA",
            "contender": "LDA",
            "param": 10,
            "seed": [1, 2],
            "n_topics": [10, 10],
            "c_v": [0.4, 0.6],
            "c_npmi": [0.0, 0.2],
            "diversity": [0.9, 0.9],
            "outlier_share_before": [np.nan, np.nan],
        }
    )
    fits = {
        ("LDA k=10", s): M.TopicFit("LDA", {}, s, [["a", "b"]], np.array([0]), None) for s in (1, 2)
    }
    out = E.summarise(runs, fits, [cfg], [1, 2]).iloc[0]
    assert out["c_v_mean"] == pytest.approx(0.5) and out["c_v_sd"] == pytest.approx(
        0.1414, abs=1e-3
    )
    assert out["eligible"] and out["stability"] == 1.0


# ------------------------------------------------------------------ outputs
def _table():
    return pd.DataFrame(
        {
            "topic": [0, 1],
            "topic_id": [0, 1],
            "n_incidents": [3, 2],
            "share": [0.6, 0.4],
            "terms": [["a", "b", "c"], ["d", "e", "f"]],
            "top_10_terms": ["a, b, c", "d, e, f"],
            "representative_incident_ids": ["1;2", "3;4"],
            "representative_titles": ["t1 || t2", "t3 || t4"],
            "suggested_label": ["a / b / c", "d / e / f"],
        }
    )


def test_suggested_label_and_model_key():
    assert O.suggested_label(["x", "y", "z", "w"]) == "x / y / z"
    assert O.model_key("LDA k=10", [["a"]]) != O.model_key("LDA k=10", [["b"]])
    assert O.model_key("LDA k=10", [["a"]]) == O.model_key("LDA k=10", [["a"]])


def test_topic_labels_template_blank_and_answers_protected(tmp_path):
    path = tmp_path / "topic_labels.csv"
    assert O.write_topic_labels(_table(), "key1", path) == "created"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert (df["final_label"] == "").all() and (df["notes"] == "").all()  # blank for the human
    assert df["suggested_label"].tolist() == ["a / b / c", "d / e / f"]
    assert O.write_topic_labels(_table(), "key1", path) == "unchanged"
    assert (
        O.write_topic_labels(_table(), "key2", path) == "replaced"
    )  # no answers yet: safe to replace
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.loc[0, "final_label"] = "Vehicle crashes"
    df.to_csv(path, index=False)
    assert (
        O.write_topic_labels(_table(), "key3", path) == "proposed"
    )  # answers exist: never overwritten
    assert (
        pd.read_csv(path, dtype=str, keep_default_na=False).loc[0, "final_label"]
        == "Vehicle crashes"
    )
    assert (tmp_path / "topic_labels.proposed.csv").exists()
