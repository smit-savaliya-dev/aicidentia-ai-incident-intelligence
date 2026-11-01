"""RQ1a experiment: fit LDA, NMF and BERTopic over a grid and seeds, score them, and choose a model.

The selection rule is fixed here, in code, before any results are inspected:

1. A configuration is *eligible* if its mean topic count lies in `topics.eligible_topics` (an
   interpretable number of themes for a human to label).
2. Within each contender (LDA, NMF, BERTopic per embedding model), the best eligible configuration
   is the one with the best summed rank of mean C_v and mean NPMI coherence (ties: diversity,
   then stability).
3. Across contenders, the winner has the lowest mean rank over four metrics: C_v, NPMI, topic
   diversity and seed stability (ties: C_v).
4. Sensitivity: the winner is re-derived under coherence-only, C_v-only, NPMI-only and
   without-stability rules, and reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aicidentia import cache
from aicidentia.config import load_config
from aicidentia.features.text import Corpus, embed_sentences, topic_settings
from aicidentia.models import topics as M

logger = logging.getLogger(__name__)
CODE_VERSION = "1"
METRICS = ("c_v", "c_npmi", "diversity", "stability")


@dataclass(frozen=True)
class Config:
    """One model configuration: family, display label, and parameters."""

    family: str  # 'LDA' | 'NMF' | 'BERTopic'
    label: str
    contender: str  # what competes in the cross-family comparison
    param: int  # k for LDA/NMF, min_cluster_size for BERTopic
    embedding_key: str | None = None


@dataclass
class ExperimentResult:
    """Everything the experiment produced."""

    runs: pd.DataFrame  # one row per (config, seed)
    summary: pd.DataFrame  # one row per config
    fits: dict[tuple[str, int], M.TopicFit] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)


@dataclass
class Selection:
    """Outcome of the pre-specified selection rule."""

    best_per_contender: pd.DataFrame
    ranks: pd.DataFrame
    winner: str
    winner_config: str
    sensitivity: dict[str, str]


def embedding_label(key: str) -> str:
    """'sentence_embedding_small' -> 'MiniLM'; 'sentence_embedding_large' -> 'mpnet' (from the model names)."""
    name = load_config()["models"][key].lower()
    return "MiniLM" if "minilm" in name else "mpnet" if "mpnet" in name else key


def build_configs() -> list[Config]:
    """The grid: LDA and NMF over k; BERTopic over embedding model and min_cluster_size."""
    cfg = topic_settings()
    configs = []
    for family in ("LDA", "NMF"):
        configs += [Config(family, f"{family} k={k}", family, k) for k in cfg["k_grid"]]
    for key in cfg["bertopic"]["embedding_models"]:
        tag = embedding_label(key)
        configs += [
            Config("BERTopic", f"BERTopic[{tag}] mcs={m}", f"BERTopic[{tag}]", m, key)
            for m in cfg["bertopic"]["min_cluster_size"]
        ]
    return configs


def seeds_for_run() -> list[int]:
    """base seed .. base seed + n_seeds - 1."""
    base = int(load_config()["random_seed"])
    return [base + i for i in range(int(topic_settings()["n_seeds"]))]


def run_one(corpus: Corpus, config: Config, seed: int, embeddings: dict[str, np.ndarray]) -> dict:
    """Fit one configuration with one seed and score it (cached on corpus + params + seed)."""
    top_n = topic_settings()["top_n"]

    def compute() -> dict:
        if config.family == "LDA":
            fit = M.fit_lda(corpus, config.param, seed)
        elif config.family == "NMF":
            fit = M.fit_nmf(corpus, config.param, seed)
        else:
            fit = M.fit_bertopic(
                corpus, embeddings[config.embedding_key], seed, config.param, config.embedding_key
            )
        return {
            "fit": fit,
            "c_v": M.topic_coherence(fit.terms, corpus, "c_v"),
            "c_npmi": M.topic_coherence(fit.terms, corpus, "c_npmi"),
            "diversity": M.topic_diversity(fit.terms, top_n),
        }

    parts = (corpus.fingerprint, config.family, config.param, config.embedding_key, seed, top_n)
    return cache.cached("topicfit", parts, compute, version=CODE_VERSION)


def run_experiment(corpus: Corpus) -> ExperimentResult:
    """Run the whole grid over all seeds and summarise it."""
    cfg = load_config()
    embeddings = {
        key: embed_sentences(corpus.texts, cfg["models"][key])
        for key in topic_settings()["bertopic"]["embedding_models"]
    }
    configs, seeds = build_configs(), seeds_for_run()
    rows, fits = [], {}
    for i, config in enumerate(configs, 1):
        logger.info("[%d/%d] %s", i, len(configs), config.label)
        for seed in seeds:
            out = run_one(corpus, config, seed, embeddings)
            fits[(config.label, seed)] = out["fit"]
            rows.append(
                {
                    "label": config.label,
                    "family": config.family,
                    "contender": config.contender,
                    "param": config.param,
                    "seed": seed,
                    "n_topics": out["fit"].n_topics,
                    "c_v": out["c_v"],
                    "c_npmi": out["c_npmi"],
                    "diversity": out["diversity"],
                    "outlier_share_before": out["fit"].extra.get("outlier_share_before", np.nan),
                }
            )
    runs = pd.DataFrame(rows)
    summary = summarise(runs, fits, configs, seeds)
    return ExperimentResult(runs, summary, fits, seeds)


def summarise(
    runs: pd.DataFrame,
    fits: dict[tuple[str, int], M.TopicFit],
    configs: list[Config],
    seeds: list[int],
) -> pd.DataFrame:
    """Mean and standard deviation of every metric over seeds, plus seed stability, per configuration."""
    lo, hi = topic_settings()["eligible_topics"]
    rows = []
    for config in configs:
        sub = runs[runs["label"] == config.label]
        row = {
            "label": config.label,
            "family": config.family,
            "contender": config.contender,
            "param": config.param,
        }
        row["n_topics_mean"] = sub["n_topics"].mean()
        row["n_topics_sd"] = sub["n_topics"].std(ddof=1)
        for metric in ("c_v", "c_npmi", "diversity", "outlier_share_before"):
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_sd"] = sub[metric].std(ddof=1)
        row["stability"] = M.stability([fits[(config.label, s)] for s in seeds])
        row["eligible"] = bool(lo <= row["n_topics_mean"] <= hi)
        rows.append(row)
    return pd.DataFrame(rows)


def _rank(series: pd.Series) -> pd.Series:
    """Rank where 1 = best (highest value); ties share the average rank."""
    return series.rank(ascending=False, method="average")


def select_model(summary: pd.DataFrame) -> Selection:
    """Apply the pre-specified selection rule (see module docstring)."""
    eligible = summary[summary["eligible"]].copy()
    best_rows = []
    for _, group in eligible.groupby("contender", sort=False):
        group = group.copy()
        group["within_score"] = _rank(group["c_v_mean"]) + _rank(group["c_npmi_mean"])
        best = group.sort_values(
            ["within_score", "diversity_mean", "stability"], ascending=[True, False, False]
        ).iloc[0]
        best_rows.append(best)
    best_df = pd.DataFrame(best_rows).reset_index(drop=True)
    cols = {
        "c_v": "c_v_mean",
        "c_npmi": "c_npmi_mean",
        "diversity": "diversity_mean",
        "stability": "stability",
    }
    ranks = best_df[["contender", "label"]].copy()
    for name, col in cols.items():
        ranks[f"rank_{name}"] = _rank(best_df[col])
    ranks["mean_rank"] = ranks[[f"rank_{m}" for m in METRICS]].mean(axis=1)
    ranks = ranks.sort_values(["mean_rank", "rank_c_v"]).reset_index(drop=True)

    def winner_under(metrics: tuple[str, ...]) -> str:
        score = ranks[[f"rank_{m}" for m in metrics]].mean(axis=1)
        order = ranks.assign(score=score).sort_values(["score", "rank_c_v"])
        return str(order.iloc[0]["contender"])

    sensitivity = {
        "all four metrics (the rule)": winner_under(METRICS),
        "coherence only (C_v + NPMI)": winner_under(("c_v", "c_npmi")),
        "C_v only": winner_under(("c_v",)),
        "NPMI only": winner_under(("c_npmi",)),
        "without stability": winner_under(("c_v", "c_npmi", "diversity")),
    }
    winner = str(ranks.iloc[0]["contender"])
    winner_config = str(ranks.iloc[0]["label"])
    return Selection(best_df, ranks, winner, winner_config, sensitivity)
