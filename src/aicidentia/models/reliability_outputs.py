"""Run the three label-reliability checks inside Phase 6 and write their registry entries, tables and figures."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from aicidentia import registry
from aicidentia.config import load_config
from aicidentia.data import load
from aicidentia.models import reliability as R
from aicidentia.models import retrieval
from aicidentia.viz import plots, style

logger = logging.getLogger(__name__)
PRODUCER = "aicidentia.models.reliability_outputs.run_reliability"
SOURCE = "data/processed/incidents_clean.parquet"


@dataclass
class ReliabilityResults:
    mappings: pd.DataFrame
    crosstabs: dict[str, pd.DataFrame]
    pairs: dict[str, dict[str, Any]]
    pair_domains: pd.DataFrame
    cl: R.ConfidentLearning
    confusion_pairs: pd.DataFrame
    symmetric: pd.DataFrame
    retrain: pd.DataFrame
    model: str
    facts: dict[str, Any] = field(default_factory=dict)


def run_reliability(
    incidents: pd.DataFrame,
    data: Any,
    best_model: str,
    best_spec: Any,
    oof_probs: pd.DataFrame,
    oof_pred: pd.DataFrame,
    seed: int,
    n_boot: int,
) -> ReliabilityResults:
    """`oof_probs`: out-of-fold probabilities of the best classifier (incident_id, proba_<k>);
    `oof_pred`: its uncalibrated out-of-fold predictions (incident_id, y_true, y_pred) for the retrain check."""
    mappings, crosstabs = R.mapping_agreement(incidents, n_boot=2000, seed=seed)

    labelled = incidents[incidents["has_mit"]]
    domain = dict(
        zip(labelled["incident_id"].astype(int), labelled["mit_risk_domain"], strict=True)
    )
    year = dict(
        zip(incidents["incident_id"].astype(int), incidents["year"].astype(int), strict=True)
    )
    gt = retrieval.build_ground_truth(
        load.load_snapshot_incidents(), set(incidents["incident_id"].astype(int))
    )
    similar, dissimilar = R.unique_pairs(gt.relevant), R.unique_pairs(gt.dissimilar)
    pairs = {
        "similar": R.pair_consistency(similar, domain, year, seed=seed),
        "dissimilar": R.pair_consistency(dissimilar, domain, year, seed=seed),
    }
    pair_domains = R.pair_domain_table(similar, domain)

    k = data.n_classes
    frame = oof_probs.set_index("incident_id").loc[data.frame["incident_id"]]
    probs = frame[[f"proba_{i}" for i in range(k)]].to_numpy()
    cl = R.confident_learning(data.y, probs, data.classes, data.frame["incident_id"].to_numpy())
    n = len(data.y)
    cleaned = R.retrain_without_flagged(best_spec, data, seed)
    removed = cleaned.drop_duplicates("fold")[["fold", "n_removed", "n_train"]].sort_values("fold")
    retrain = R.retrain_sensitivity(
        oof_pred[["incident_id", "y_true", "y_pred"]],
        cleaned,
        set(cl.ranked["incident_id"].astype(int)),
        k,
        n_boot,
        seed,
    )
    res = ReliabilityResults(
        mappings=mappings,
        crosstabs=crosstabs,
        pairs=pairs,
        pair_domains=pair_domains,
        cl=cl,
        confusion_pairs=R.confusion_pairs(cl.joint, n),
        symmetric=R.symmetric_confusion(cl.joint, n),
        retrain=retrain,
        model=best_model,
    )
    primary = retrain.iloc[0]
    res.facts = {
        "mappings": mappings.drop(columns=["rationale"]).round(4),
        "mapping_kappa": dict(zip(mappings["mapping"], mappings["kappa"].round(4), strict=True)),
        "mapping_n": dict(zip(mappings["mapping"], mappings["n"], strict=True)),
        "cset_mit_overlap": int((incidents["has_mit"] & incidents["has_cset"]).sum()),
        "gmf_mit_overlap": int((incidents["has_mit"] & incidents["has_gmf"]).sum()),
        "cset_year_max": int(incidents.loc[incidents["has_cset"], "year"].max()),
        "pairs_similar": pairs["similar"],
        "pairs_dissimilar": pairs["dissimilar"],
        "cl_model": best_model,
        "cl_n": cl.facts["n"],
        "cl_flagged": cl.facts["flagged"],
        "cl_flagged_share": cl.facts["flagged_share"],
        "cl_noise_rate": cl.facts["estimated_noise_rate"],
        "cl_noise_by_domain": dict(
            zip(
                cl.noise_by_domain["domain"],
                cl.noise_by_domain["estimated_noise_rate"].round(4),
                strict=True,
            )
        ),
        "cl_top_confusions": res.confusion_pairs.round(3),
        "retrain_macro_f1_original": float(primary["macro_f1_original"]),
        "retrain_macro_f1_cleaned": float(primary["macro_f1_cleaned_training"]),
        "retrain_diff": float(primary["difference"]),
        "retrain_ci": [float(primary["ci_low"]), float(primary["ci_high"])],
        "retrain_table": retrain.round(4),
        "retrain_removed_per_fold": removed.to_dict(orient="records"),
        "retrain_removed_share": float(removed["n_removed"].sum() / removed["n_train"].sum()),
    }
    return res


def register(res: ReliabilityResults) -> None:
    for key, value in res.facts.items():
        registry.record(f"rq3rel.{key}", value, SOURCE, PRODUCER)


def save_tables(res: ReliabilityResults) -> None:
    style.save_table(res.mappings.round(4), "rq3rel_mappings")
    for name, table in res.crosstabs.items():
        style.save_table(table, f"rq3rel_crosstab_{name}", index=True)
    pair_rows = pd.DataFrame(
        [{"pairs": name, **values} for name, values in res.pairs.items()]
    ).round(4)
    style.save_table(pair_rows, "rq3rel_editor_pairs")
    style.save_table(res.pair_domains, "rq3rel_editor_pairs_differing_domains")
    style.save_table(res.cl.noise_by_domain.round(4), "rq3rel_noise_by_domain")
    style.save_table(
        (res.cl.joint * res.cl.facts["n"]).round(1), "rq3rel_estimated_joint", index=True
    )
    style.save_table(res.confusion_pairs.round(3), "rq3rel_top_confusions")
    style.save_table(res.symmetric.round(2), "rq3rel_confusion_by_pair")
    style.save_table(res.cl.ranked.round(4), "rq3rel_flagged_incidents")
    style.save_table(res.retrain.round(4), "rq3rel_retrain_without_flagged")


def make_figures(res: ReliabilityResults) -> None:
    snap = str(load_config()["snapshot_date"])
    figs = {
        "fig_rq3_5_label_reliability": plots.plot_label_reliability(
            res.mappings, res.pairs, res.cl.noise_by_domain
        ),
        "fig_rq3_6_label_confusion": plots.plot_label_confusion(res.cl.joint, res.cl.facts["n"]),
    }
    for name, fig in figs.items():
        style.save_figure(fig, name, snap)


def write_all(res: ReliabilityResults) -> None:
    register(res)
    save_tables(res)
    make_figures(res)


def flagged_examples(res: ReliabilityResults, titles: pd.Series, n: int = 8) -> pd.DataFrame:
    """The most confidently flagged incidents with their titles (for the report and the dashboard)."""
    top = res.cl.ranked.head(n).copy()
    top["title"] = top["incident_id"].map(titles)
    return top[["incident_id", "title", "given_label", "suggested_label", "self_confidence"]]
