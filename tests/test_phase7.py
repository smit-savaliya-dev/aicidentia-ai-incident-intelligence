"""Phase 7 tests: retrieval metrics, ground truth, indexes, ranking, evaluation, deployment rule, fallback.

Fixtures are tiny hand-made collections with known answers (not analysis data).
"""

import numpy as np
import pandas as pd
import pytest

from aicidentia.evaluation import metrics as M
from aicidentia.models import retrieval as R
from aicidentia.models import retrieval_outputs as O


# ------------------------------------------------------------------ metrics
def test_retrieval_metrics_known_values():
    ranked, rel = [5, 7, 9, 1, 2], {9, 2}
    assert M.precision_at_k(ranked, rel, 5) == pytest.approx(2 / 5)
    assert M.precision_at_k([9], rel, 5) == pytest.approx(1 / 5)  # denominator is always k
    assert M.recall_at_k(ranked, rel, 3) == 0.5 and np.isnan(M.recall_at_k(ranked, set(), 3))
    assert M.hit_at_k(ranked, rel, 2) == 0.0 and M.hit_at_k(ranked, rel, 3) == 1.0
    assert M.reciprocal_rank(ranked, rel) == pytest.approx(1 / 3)
    assert M.reciprocal_rank(ranked, rel, depth=2) == 0.0 and M.reciprocal_rank(ranked, {42}) == 0.0
    ideal = 1 + 1 / np.log2(3)
    assert M.ndcg_at_k(ranked, rel, 5) == pytest.approx((1 / np.log2(4) + 1 / np.log2(6)) / ideal)
    assert M.ndcg_at_k([9, 2], rel, 5) == pytest.approx(1.0)
    assert M.mean_reciprocal_rank([[1, 2], [3, 4]], [{2}, {3}]) == pytest.approx((0.5 + 1) / 2)


def test_paired_bootstrap_and_holm():
    a, b = np.ones(50), np.zeros(50)
    res = M.paired_bootstrap_mean_diff(a, b, 300, 1)
    assert res["diff"] == 1.0 and res["ci_low"] == 1.0 and res["p_value"] < 0.01
    tie = M.paired_bootstrap_mean_diff(a, a, 300, 1)
    assert tie["diff"] == 0 and tie["p_value"] == 1.0
    assert M.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert M.holm_adjust([0.5]) == [0.5]


# ------------------------------------------------------------------ ground truth
def _snapshot():
    return pd.DataFrame(
        {
            "incident_id": [1, 2, 3, 4, 5],
            "editor_similar_incidents": [
                [2],
                [],
                [4, 99],
                [],
                [5],
            ],  # 99 is out of scope, 5->5 is a self-link
            "editor_dissimilar_incidents": [[3], [], [], [], []],
            "nlp_similar_incidents": [
                [{"incident_id": 3, "similarity": 0.5}, {"incident_id": 2, "similarity": 0.9}],
                [],
                [],
                [],
                [],
            ],
        }
    )


def test_ground_truth_symmetrised_scoped_and_suggestions():
    gt = R.build_ground_truth(_snapshot(), {1, 2, 3, 4, 5})
    assert gt.relevant == {1: {2}, 2: {1}, 3: {4}, 4: {3}}
    assert gt.directed == {1: {2}, 3: {4}}
    assert gt.dissimilar == {1: {3}, 3: {1}}
    assert gt.aiid_suggestions[1] == [2, 3]  # ranked by similarity
    assert gt.queries == [1, 2, 3, 4]
    ov = R.suggestion_overlap(gt)
    assert ov["directed_links"] == 2 and ov["links_in_aiid_suggestions"] == 1


# ------------------------------------------------------------------ index and ranking
def _frame():
    return pd.DataFrame(
        {
            "incident_id": [1, 2, 3, 4, 5, 6],
            "title": [
                "robotaxi crash",
                "robotaxi hits cyclist",
                "deepfake scam call",
                "voice clone scam",
                "chatbot bad advice",
                "unrelated weather",
            ],
            "text": [
                "robotaxi crash on road",
                "robotaxi hits cyclist on road",
                "deepfake voice scam call bank",
                "voice clone deepfake scam bank",
                "chatbot gave bad medical advice",
                "weather report sunny day",
            ],
            "date": pd.to_datetime(
                ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"]
            ),
        }
    )


def _index(with_embeddings=True):
    f = _frame()
    toks = [t.split() for t in f["text"]]
    emb, names = {}, {}
    if with_embeddings:
        topic = np.array(
            [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        emb = {"minilm": topic, "mpnet": topic}
        names = {"minilm": "fake-small", "mpnet": "fake-large"}
    return R.build_index(f, toks, lambda texts: [t.split() for t in texts], emb, names)


@pytest.mark.parametrize("method", R.METHODS)
def test_every_method_ranks_the_obvious_neighbour_first(method):
    idx = _index()
    scores = R.score_rows(idx, method, [idx.row_of[1], idx.row_of[3]])
    assert scores.shape == (2, 6)
    assert R.rank_candidates(scores[0], idx, exclude_ids=[1])[0] == 2
    assert R.rank_candidates(scores[1], idx, exclude_ids=[3])[0] == 4


def test_rank_candidates_excludes_query_and_later_incidents_and_breaks_ties_by_id():
    idx = _index()
    flat = np.zeros(6)
    assert R.rank_candidates(flat, idx, exclude_ids=[3]) == [1, 2, 4, 5, 6]
    assert R.rank_candidates(flat, idx, before=idx.dates[idx.row_of[4]]) == [1, 2, 3]


def test_rrf_prefers_items_ranked_high_by_both():
    a = np.array([[3.0, 2.0, 1.0]])
    b = np.array([[1.0, 3.0, 2.0]])
    fused = R._rrf([a, b], k=60)[0]
    assert fused.argmax() == 1  # ranks (2,1) beat (1,3) and (3,2)


def test_free_text_search_and_embedding_requirement():
    idx = _index()
    out = R.search(idx, "bm25", "scam call with a cloned voice", k=3)
    assert set(out["incident_id"].iloc[:2]) == {3, 4} and list(out["rank"]) == [1, 2, 3]
    embed = lambda texts, name: np.array([[0, 1, 0, 0]], dtype=float)  # noqa: E731 - tiny stand-in encoder
    assert set(R.search(idx, "mpnet", "anything", k=2, embed=embed)["incident_id"]) == {3, 4}
    with pytest.raises(ValueError):
        R.score_texts(idx, "mpnet", ["x"])
    with pytest.raises(ValueError):
        R.score_texts(idx, "lsa", ["x"])


def test_available_methods_without_embeddings():
    assert _index(with_embeddings=False).available_methods == ["tfidf", "bm25"]
    assert _index().available_methods == ["tfidf", "bm25", "minilm", "mpnet", "hybrid"]


# ------------------------------------------------------------------ evaluation
def _gt():
    return R.GroundTruth(
        {1: {2}, 2: {1}, 3: {4}, 4: {3}}, {1: {2}, 3: {4}}, {1: {3}, 3: {1}}, {1: [6], 3: [4]}
    )


def test_evaluate_method_perfect_scores_and_earlier_only():
    idx = _index()
    pq = R.evaluate_method(idx, "mpnet", _gt())
    assert (
        len(pq) == 4
        and (pq["rr"] == 1).all()
        and (pq["first_rank"] == 1).all()
        and (pq["hit@5"] == 1).all()
    )
    assert pq.set_index("query").loc[1, "dissimilar@5"] == pytest.approx(
        0.2
    ) and np.isnan(  # 5 of 5 candidates shown, incl. dissimilar #3
        pq.set_index("query").loc[2, "dissimilar@5"]
    )
    earlier = R.evaluate_method(idx, "mpnet", _gt(), earlier_only=True)
    assert sorted(earlier["query"]) == [2, 4]  # 1 and 3 have no earlier relevant incident


def test_evaluate_suggestions_oracle_and_summary():
    gt = _gt()
    sug = R.evaluate_suggestions(gt, [1, 3])
    assert (
        sug.set_index("query").loc[1, "hit@3"] == 0 and sug.set_index("query").loc[3, "rr@3"] == 1.0
    )
    assert R.oracle(gt, gt.queries) == {
        "p@5": pytest.approx(0.2),
        "p@3": pytest.approx(1 / 3),
        "queries_with_one_relevant": 1.0,
    }
    pq = pd.concat([R.evaluate_method(_index(), m, gt) for m in ("bm25", "mpnet")])
    summary = R.summarise(pq, ["rr", "p@5"], 200)
    assert (
        set(summary["method"]) == {"bm25", "mpnet"} and (summary["rr_low"] <= summary["rr"]).all()
    )
    comp = R.paired_comparisons(pq, "bm25", "rr", 200)
    assert list(comp["method"]) == ["mpnet"] and "p_holm" in comp


def test_choose_deployed_rule():
    summary = pd.DataFrame({"method": ["bm25", "hybrid", "mpnet"], "rr": [0.49, 0.51, 0.45]})
    not_sig = pd.DataFrame(
        {
            "method": ["hybrid", "mpnet"],
            "diff": [0.02, -0.04],
            "ci_low": [-0.01, -0.07],
            "ci_high": [0.05, -0.01],
        }
    )
    assert O.choose_deployed(summary, not_sig)[0] == "bm25"
    sig = not_sig.assign(ci_low=[0.005, -0.07])
    method, reason = O.choose_deployed(summary, sig)
    assert method == "hybrid" and "significantly" in reason
    assert O.choose_deployed(summary.assign(rr=[0.6, 0.5, 0.4]), sig)[0] == "bm25"


def test_build_default_index_falls_back_to_lexical_when_models_unavailable(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("model not in cache and offline")

    monkeypatch.setattr(O, "embed_sentences", broken)
    monkeypatch.setattr(O, "cached_tokens", lambda frame: [t.split() for t in frame["text"]])
    idx, missing = O.build_default_index(_frame(), local_files_only=True)
    assert missing == ["minilm", "mpnet"] and idx.available_methods == ["tfidf", "bm25"]
    assert R.search(idx, "bm25", "robotaxi crash", k=1)["incident_id"].iloc[0] == 1
