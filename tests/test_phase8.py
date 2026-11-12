"""Phase 8 tests: dashboard data and blinding, model fallback, manual-input status, refresh runner, charts, pages."""

import json

import numpy as np
import pandas as pd
import pytest

from aicidentia import pipeline
from aicidentia.config import PROJECT_ROOT
from aicidentia.dashboard import charts, data, models, status
from aicidentia.models import classify as C
from aicidentia.models import label_audit


# ------------------------------------------------------------------ blinding
def test_incidents_hide_mit_labels_of_audit_incidents():
    inc = data.incidents(hide=set())
    some = set(inc.loc[inc["mit_domain"] != "(no MIT label)", "incident_id"].head(3))
    hidden = data.incidents(hide=some)
    assert (hidden.loc[hidden["incident_id"].isin(some), "mit_domain"] == data.HIDDEN).all()
    others = hidden[~hidden["incident_id"].isin(some)]
    assert (others["mit_domain"] != data.HIDDEN).all()
    assert {"incident_id", "title", "theme", "mit_domain", "reports"} <= set(hidden.columns)


def test_blind_ids_follow_annotation_progress(monkeypatch):
    monkeypatch.setattr(label_audit, "blind_audit_active", lambda: True)
    monkeypatch.setattr(label_audit, "audit_incident_ids", lambda: {1, 2})
    assert data.blind_ids() == {1, 2}
    monkeypatch.setattr(label_audit, "blind_audit_active", lambda: False)
    assert data.blind_ids() == set()


def test_node_profile_withholds_domain_mix_while_blind():
    node = {
        "label": "X",
        "type": "organisation",
        "community": 0,
        "n_incidents": 3,
        "first_year": 2020,
        "last_year": 2024,
        "pagerank": 0.01,
        "domains": {"Misinformation": 3},
        "recent_incidents": [{"id": 1, "title": "t", "year": 2024}],
    }
    assert data.node_profile(node, hide={99})["domains"] is None
    assert data.node_profile(node, hide=set())["domains"] == {"Misinformation": 3}


# ------------------------------------------------------------------ model loading and fallback
def _tiny_bundles(tmp_path):
    texts = [
        "robot car crash road",
        "car crash highway robot",
        "deepfake scam voice",
        "voice clone scam call",
    ] * 5
    y = np.array([0, 0, 1, 1] * 5)
    frame = pd.DataFrame(
        {
            "incident_id": range(len(y)),
            "title": texts,
            "description": texts,
            "text": texts,
            "year": 2020,
            "domain": [["a", "b"][i] for i in y],
        }
    )
    d = C.Dataset(frame, ["a", "b"], y, C.make_folds(y, 5, 0), np.zeros(len(y), bool))
    fallback = C.fit_final_model(
        C.ModelSpec("TF-IDF + logistic regression", "tfidf_lr"), d, calibrate=None
    )
    main = dict(
        fallback,
        embedding_model="some/embedding-model",
        spec=C.ModelSpec(
            "mpnet embeddings + logistic regression", "emb_lr", "sentence_embedding_large"
        ),
    )
    C.save_model(main, tmp_path / "main.joblib")
    C.save_model(fallback, tmp_path / "fallback.joblib")
    return tmp_path / "main.joblib", tmp_path / "fallback.joblib"


def test_classifier_uses_deployed_model_when_embedding_model_loads(tmp_path):
    main, fb = _tiny_bundles(tmp_path)
    active = models.load_classifier(main, fb, loader=lambda name: object())
    assert (
        not active.is_fallback and active.name.startswith("mpnet") and "deployed" in active.reason
    )


def test_classifier_falls_back_to_tfidf_when_embedding_model_missing(tmp_path):
    main, fb = _tiny_bundles(tmp_path)

    def offline(name):
        raise OSError("not in the local cache")

    active = models.load_classifier(main, fb, loader=offline)
    assert active.is_fallback and active.name == "TF-IDF + logistic regression"
    assert "could not be loaded offline" in active.reason and active.embedding_model is None
    pred = models.classify_text(active, "voice clone scam")
    assert pred["predicted_domain"].iloc[0] == "b" and 0 <= pred["confidence"].iloc[0] <= 1
    assert models.load_classifier(tmp_path / "none.joblib", fb, loader=offline).is_fallback
    assert models.load_classifier(tmp_path / "none.joblib", tmp_path / "none2.joblib") is None


def test_retriever_falls_back_to_bm25_without_query_encoder(monkeypatch):
    def offline(name):
        raise OSError("offline")

    active = models.load_retriever(loader=offline)
    assert models.usable_methods(active) == ["tfidf", "bm25"]
    assert active.method in ("tfidf", "bm25")
    if active.deployed not in ("tfidf", "bm25"):
        assert active.is_fallback and active.method == "bm25"


# ------------------------------------------------------------------ manual-input status
def test_manual_status_rows_and_states():
    table = status.manual_status()
    assert list(table["input"]) == [
        "Topic labels",
        "Entity review",
        "Duplicate review",
        "Blind label audit",
        "DistilBERT predictions (Colab)",
    ]
    assert set(table.columns) >= {"file", "done", "total", "state", "used_by", "how"}
    assert (
        status._state(0, 5) == "not started"
        and status._state(2, 5) == "partial"
        and status._state(5, 5) == "complete"
    )
    assert status._state(0, 0) == "missing"


def test_topic_label_status_detects_stale_labels(tmp_path):
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "topic_id": [0, 1],
            "suggested_label": ["a", "b"],
            "final_label": ["X", ""],
            "model_key": "K1",
        }
    ).to_csv(labels, index=False)
    topics = tmp_path / "topics.parquet"
    pd.DataFrame({"incident_id": [1], "model_key": ["K1"]}).to_parquet(topics)
    assert status.topic_label_status(labels, topics) == {"done": 1, "total": 2, "stale": False}
    pd.DataFrame({"incident_id": [1], "model_key": ["K2"]}).to_parquet(topics)
    assert status.topic_label_status(labels, topics)["stale"]


# ------------------------------------------------------------------ refresh runner
def _fake_steps(log, fail=(), skip=(), block=()):
    def make(name):
        def run():
            log.append(name)
            if name in fail:
                raise RuntimeError("boom")
            if name in skip:
                raise pipeline.StepSkipped("nothing to do")
            if name in block:
                raise pipeline.StepBlocked("labels stale")
            return f"{name} done"

        return run

    graph = {
        "phase2": (),
        "phase3": ("phase2",),
        "phase4": ("phase3",),
        "phase5": ("phase2",),
        "phase6": ("phase2",),
        "phase6_audit": ("phase6",),
        "phase7": ("phase2",),
        "dashboard": ("phase4", "phase5", "phase6", "phase7"),
    }
    return [pipeline.Step(n, n, make(n), req) for n, req in graph.items()]


def test_runner_runs_in_order_and_isolates_failures():
    log = []
    res = {r.name: r for r in pipeline.run_steps(_fake_steps(log, fail={"phase5"}))}
    assert log == [
        "phase2",
        "phase3",
        "phase4",
        "phase5",
        "phase6",
        "phase6_audit",
        "phase7",
    ]  # phase5 ran and failed; independent steps still ran; the dashboard check was blocked
    assert res["phase5"].status == "failed" and "boom" in res["phase5"].note
    assert res["dashboard"].status == "blocked" and "phase5" in res["dashboard"].note
    assert res["phase7"].status == "ok"


def test_runner_blocked_and_skipped_steps():
    log = []
    res = {
        r.name: r
        for r in pipeline.run_steps(_fake_steps(log, block={"phase4"}, skip={"phase6_audit"}))
    }
    assert res["phase4"].status == "blocked" and res["dashboard"].status == "blocked"
    assert res["phase6_audit"].status == "skipped" and res["phase6"].status == "ok"
    log2 = []
    res2 = {r.name: r for r in pipeline.run_steps(_fake_steps(log2, fail={"phase2"}))}
    assert log2 == ["phase2"] and all(
        r.status == "blocked" for n, r in res2.items() if n != "phase2"
    )


def test_select_only_and_from():
    steps = _fake_steps([])
    assert [s.name for s in pipeline.select(steps, only=["phase4", "phase7"])] == [
        "phase4",
        "phase7",
    ]
    assert [s.name for s in pipeline.select(steps, start="phase6")] == [
        "phase6",
        "phase6_audit",
        "phase7",
        "dashboard",
    ]
    with pytest.raises(ValueError):
        pipeline.select(steps, only=["phase9"])
    assert [s.name for s in pipeline.steps(pipeline.RefreshOptions(dashboard_check=False))][
        -1
    ] == "phase7"


def test_audit_step_waits_for_complete_annotation(monkeypatch):
    monkeypatch.setattr(
        label_audit,
        "annotation_progress",
        lambda *a, **k: {"rows": 150, "labelled": 0, "unrecognised": 0},
    )
    with pytest.raises(pipeline.StepSkipped, match="no annotations"):
        pipeline.run_phase6_audit()
    monkeypatch.setattr(
        label_audit,
        "annotation_progress",
        lambda *a, **k: {"rows": 150, "labelled": 40, "unrecognised": 0},
    )
    with pytest.raises(pipeline.StepSkipped, match="40/150"):
        pipeline.run_phase6_audit()


def test_write_summary(tmp_path):
    results = [pipeline.StepResult("phase2", "Phase 2", "ok", 1.0, "done")]
    path = tmp_path / "s.md"
    pipeline.write_summary(results, pd.DataFrame({"input": ["x"], "state": ["complete"]}), path)
    text = path.read_text(encoding="utf-8")
    assert "Last refresh" in text and "phase2" in text and "complete" in text


def test_refresh_script_check_mode(capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "refresh_all", PROJECT_ROOT / "scripts" / "refresh_all.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--check"]) == 0
    assert "Blind label audit" in capsys.readouterr().out


# ------------------------------------------------------------------ charts
def test_charts_build_from_real_outputs():
    assert len(charts.incidents_per_year(data.yearly_by_year()).data) == 1
    yearly = data.table("rq1b_yearly_themes")
    groups = yearly["group"].unique()[:2].tolist()
    fig = charts.theme_lines(yearly, groups, "share")
    assert len(fig.data) >= 2
    assert charts.theme_heatmap(yearly, groups).data[0].z.shape[0] == 2
    net = json.loads(data.NETWORK_JSON.read_text(encoding="utf-8"))
    for n in net["nodes"]:
        n["x"], n["y"] = 0.0, 0.0
    early = charts.network_figure(net, 2016, ["organisation"])
    late = charts.network_figure(net, 2026, ["organisation"])
    count = lambda f: sum(len(t.x) for t in f.data if t.mode == "markers")  # noqa: E731
    assert count(early) <= count(late)
    pred = pd.DataFrame(
        {"predicted_domain": ["x"], "confidence": [0.7], "p_x": [0.7], "p_y": [0.3]}
    )
    assert list(charts.probability_bars(pred).data[0].y) == ["y", "x"]


# ------------------------------------------------------------------ pages
@pytest.mark.parametrize(
    "page",
    [
        "Home.py",
        "pages/1_Overview.py",
        "pages/2_Themes_and_Trends.py",
        "pages/3_Risk_Network.py",
        "pages/4_Classify_and_Compare.py",
        "pages/5_Method_and_Limitations.py",
    ],
)
def test_page_renders_without_error(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app" / page), default_timeout=300).run()
    assert not at.exception, at.exception[0].value if at.exception else ""


def test_method_page_names_the_active_models():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "pages" / "5_Method_and_Limitations.py"), default_timeout=300
    ).run()
    active = at.dataframe[0].value
    assert list(active["component"]) == ["Risk-domain classifier", "Similar-incident search"]
    assert active["active"].str.len().gt(0).all() and set(active["fallback"]) <= {"yes", "no"}
