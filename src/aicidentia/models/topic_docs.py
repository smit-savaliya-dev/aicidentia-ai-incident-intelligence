"""Build docs/rq1_topic_models.md from the Phase 3 results."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from aicidentia.config import load_config
from aicidentia.data.audit_docs import n, pct
from aicidentia.features.text import Corpus, stopword_set, topic_settings
from aicidentia.models import topic_experiment as E
from aicidentia.models import topics as M
from aicidentia.viz.style import to_markdown

logger = logging.getLogger(__name__)


def _pm(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def _figure(name: str, caption: str, snapshot: str) -> str:
    return f"![{caption}](../reports/figures/{name}.png)\n\n*Figure: {caption}. Source: AIID snapshot {snapshot}.*\n"


def _weight_sentence(fit: M.TopicFit) -> str:
    """Median dominant-topic weight and share of incidents below 0.5 (LDA/NMF only)."""
    if fit.weights is None:
        return ""
    dominant = fit.weights[np.arange(len(fit.assignments)), np.clip(fit.assignments, 0, None)]
    return (
        f"For the chosen model the median dominant-topic weight is {np.median(dominant):.2f} and "
        f"{pct(float((dominant < 0.5).mean()), 0)} of incidents have a dominant weight below 0.5, i.e. they sit between themes. "
    )


def _compare(a: pd.Series, b: pd.Series, a_name: str, b_name: str) -> list[dict]:
    """Metric-by-metric comparison of two contenders' best configurations (a minus b)."""
    rows = []
    for metric, label in (("c_v", "C_v"), ("c_npmi", "NPMI"), ("diversity", "Topic diversity")):
        diff = a[f"{metric}_mean"] - b[f"{metric}_mean"]
        spread = max(a[f"{metric}_sd"], b[f"{metric}_sd"])
        reading = (
            f"{a_name} higher"
            if diff > spread
            else f"{b_name} higher"
            if diff < -spread
            else "within seed noise"
        )
        rows.append(
            {
                "a": a_name,
                "b": b_name,
                "metric": label,
                "a_value": round(a[f"{metric}_mean"], 3),
                "b_value": round(b[f"{metric}_mean"], 3),
                "difference": round(diff, 3),
                "seed_sd (max)": round(spread, 3),
                "reading": reading,
            }
        )
    diff = a["stability"] - b["stability"]
    rows.append(
        {
            "a": a_name,
            "b": b_name,
            "metric": "Seed stability",
            "a_value": round(a["stability"], 3),
            "b_value": round(b["stability"], 3),
            "difference": round(diff, 3),
            "seed_sd (max)": np.nan,
            "reading": f"{a_name} higher" if diff > 0 else f"{b_name} higher",
        }
    )
    return rows


def _baseline_comparison(best: pd.DataFrame, winner: str) -> tuple[str, pd.DataFrame]:
    """Winner against every other contender, and the best BERTopic against the classical models.

    A difference smaller than the larger of the two seed standard deviations is called 'within seed
    noise'. Reported honestly whichever way it goes.
    """
    by = best.set_index("contender")
    rows: list[dict] = []
    for other in by.index:
        if other != winner:
            rows += _compare(by.loc[winner], by.loc[other], winner, other)
    berts = [c for c in by.index if c.startswith("BERTopic")]
    parts = []
    if rows:
        table = pd.DataFrame(rows)
        for other in [c for c in by.index if c != winner]:
            sub = table[table["b"] == other]
            parts.append(
                f"{winner} is higher than {other} on {(sub['reading'] == f'{winner} higher').sum()} of {len(sub)} metrics and lower on {(sub['reading'] == f'{other} higher').sum()}"
            )
    sentence = "; ".join(parts) + "." if parts else "Only one contender was eligible."
    if berts and winner not in berts:
        top = max(berts, key=lambda c: by.loc[c, "c_v_mean"])
        extra = []
        for base in ("LDA", "NMF"):
            if base in by.index:
                sub = pd.DataFrame(_compare(by.loc[top], by.loc[base], top, base))
                extra.append(
                    f"the best advanced contender ({top}) beats {base} on {(sub['reading'] == f'{top} higher').sum()} of {len(sub)} metrics and falls behind on {(sub['reading'] == f'{base} higher').sum()}"
                )
                rows += _compare(by.loc[top], by.loc[base], top, base)
        sentence += " In other words, " + "; ".join(extra) + "."
    return sentence, pd.DataFrame(rows)


def write_topic_doc(
    corpus: Corpus,
    result: E.ExperimentResult,
    selection: E.Selection,
    table: pd.DataFrame,
    fit: M.TopicFit,
    label_status: str,
    path: Path,
) -> None:
    """Write docs/rq1_topic_models.md."""
    cfg = topic_settings()
    snap = str(load_config()["snapshot_date"])
    best = selection.best_per_contender
    win = best.set_index("label").loc[selection.winner_config]
    lengths = np.array([len(t) for t in corpus.tokens])
    sentence, versus = _baseline_comparison(best, selection.winner)
    agree = sorted({v for v in selection.sensitivity.values()})
    summary = result.summary
    bert = summary[summary["family"] == "BERTopic"]

    L: list[str] = []
    add = L.append
    add("# RQ1a: latent risk themes (Phase 3)\n")
    add(
        f"_Auto-generated by `aicidentia.models.topic_outputs` (AIID snapshot **{snap}**); re-run `scripts/run_phase3_topics.py` to refresh. "
        "Numbers are registered in `reports/results.json` under `rq1a.*`. Fitted models are cached in `data/interim/`._\n"
    )
    add("## Summary\n")
    add(
        f"- **Selected model: {selection.winner_config}** ({int(round(win['n_topics_mean']))} topics, seed {load_config()['random_seed']} used for the topic table). "
        f"C_v {_pm(win['c_v_mean'], win['c_v_sd'])}, NPMI {_pm(win['c_npmi_mean'], win['c_npmi_sd'])}, topic diversity {_pm(win['diversity_mean'], win['diversity_sd'])}, "
        f"seed stability {win['stability']:.3f} (mean ± sd over {len(result.seeds)} seeds).\n"
        f"- Under the alternative rules (coherence only, C_v only, NPMI only, without stability) the winner is: {', '.join(f'**{a}**' for a in agree)}. "
        + (
            "The choice is therefore robust to the weighting of the metrics.\n"
            if len(agree) == 1
            else "The choice is **sensitive to how the metrics are weighted**; see section 2.\n"
        )
        + f"- {sentence}\n"
        f"- {len(result.summary)} configurations x {len(result.seeds)} seeds = {len(result.runs)} fits, all scored on one shared tokenised corpus.\n"
        f"- **Topic labels are pending you**: `data/manual/topic_labels.csv` has the top terms, five representative incident titles and an automatic placeholder label per topic, "
        "with a blank `final_label` column for you to confirm.\n"
    )
    add("## 1. Method\n")
    add(
        f"- **Corpus.** {n(len(corpus.docs))} incidents, modelled on `text` (title + description). Lower-case, alphabetic tokens of 3+ letters, "
        f"scikit-learn English stopwords plus {len(stopword_set()) - len(stopword_set([]))} domain stopwords (hedging and boilerplate such as *allegedly*, *reportedly*, *system*, *ai*), "
        f"WordNet lemmatisation (noun then verb), then a document-frequency filter (kept: in at least {cfg['min_df']} incidents and at most {pct(cfg['max_df'], 0)}). "
        f"Result: vocabulary of {n(len(corpus.vocabulary))} words, {lengths.mean():.1f} tokens per incident on average (minimum {lengths.min()}).\n"
        f"- **LDA** (gensim): batch variational Bayes, learned alpha and eta, {cfg['lda']['passes']} passes; k in {cfg['k_grid']}.\n"
        f"- **NMF** (scikit-learn): TF-IDF matrix over the same vocabulary, random initialisation (so seeds differ), same k grid.\n"
        f"- **BERTopic**: sentence embeddings of the untouched text (cached; models: "
        f"{', '.join(load_config()['models'][k] for k in cfg['bertopic']['embedding_models'])}), seeded UMAP "
        f"({cfg['bertopic']['umap']['n_components']} dimensions, {cfg['bertopic']['umap']['n_neighbors']} neighbours, cosine), scikit-learn HDBSCAN "
        f"(min_cluster_size in {cfg['bertopic']['min_cluster_size']}, min_samples {cfg['bertopic']['min_samples']}), c-TF-IDF over the shared vocabulary, "
        f"and outlier reduction by embedding similarity (every incident ends up in a topic). The number of topics is not set but emerges from min_cluster_size, so the grid was chosen "
        "from topic counts only (never from metric values) to cover roughly 8-40 topics.\n"
        f"- **Metrics.** *C_v* and *NPMI* coherence from gensim `CoherenceModel` on the shared tokenised corpus (top {cfg['top_n']} words per topic; sliding-window co-occurrence); "
        "*topic diversity* (Dieng et al., 2020) = unique words / total words in the topics' top words; *stability* = mean pairwise similarity between the topic sets of different seeds, "
        "where topics are matched one-to-one by Jaccard overlap of their top words and the summed Jaccard is divided by the larger topic count (so different topic counts are penalised).\n"
        f"- **Seeds.** {len(result.seeds)} seeds ({result.seeds[0]} to {result.seeds[-1]}); every number is mean ± sd across them.\n"
    )
    add("### Selection rule (fixed before looking at results)\n")
    lo, hi = cfg["eligible_topics"]
    add(
        f"1. A configuration is eligible if its mean number of topics is between {lo} and {hi} (a number of themes a person can label and trend).\n"
        "2. Within each contender (LDA, NMF, and BERTopic per embedding model) the best eligible configuration has the best summed rank of mean C_v and mean NPMI (ties: diversity, then stability).\n"
        "3. Across contenders the winner has the lowest mean rank over C_v, NPMI, topic diversity and seed stability.\n"
        "4. The winner is re-derived under four alternative weightings as a sensitivity check.\n"
    )
    add("## 2. Results\n")
    add("### Best eligible configuration per contender\n")
    show = best[
        [
            "contender",
            "label",
            "n_topics_mean",
            "c_v_mean",
            "c_v_sd",
            "c_npmi_mean",
            "c_npmi_sd",
            "diversity_mean",
            "diversity_sd",
            "stability",
            "outlier_share_before_mean",
        ]
    ].round(3)
    add(to_markdown(show))
    family = summary[(summary["contender"] == selection.winner) & summary["eligible"]]
    steadiest = family.sort_values("stability", ascending=False).iloc[0]
    if steadiest["label"] != selection.winner_config:
        add(
            f"\n*Note:* within {selection.winner} the rule chose **{selection.winner_config}** (stability {win['stability']:.3f}), "
            f"but **{steadiest['label']}** is the most seed-stable eligible configuration "
            f"(stability {steadiest['stability']:.3f}, C_v {steadiest['c_v_mean']:.3f}). "
            "If reproducibility of the topic set matters more to you than the coherence rank, that is the alternative.\n"
        )
    add("\n### Cross-contender ranks (1 = best)\n")
    add(to_markdown(selection.ranks.round(2)))
    add("\n### Sensitivity of the winner to the weighting\n")
    add(
        to_markdown(
            pd.DataFrame(
                {
                    "rule": list(selection.sensitivity),
                    "winner": list(selection.sensitivity.values()),
                }
            )
        )
    )
    add("")
    add("### Pairwise comparison: selected model and best BERTopic against the others\n")
    add(f"{sentence}\n")
    if len(versus):
        add(to_markdown(versus))
        add("")
    add("### BERTopic: topic count depends strongly on min_cluster_size\n")
    add(
        "HDBSCAN's `min_cluster_size` controls how many clusters survive, and the relationship is not smooth: a small step can collapse the solution to a handful of topics. "
        "Share of incidents HDBSCAN left as outliers before reduction is shown alongside.\n"
    )
    add(
        to_markdown(
            bert[
                [
                    "label",
                    "n_topics_mean",
                    "n_topics_sd",
                    "outlier_share_before_mean",
                    "c_v_mean",
                    "c_npmi_mean",
                    "eligible",
                ]
            ].round(3)
        )
    )
    add("")
    add("### Full grid (mean over seeds)\n")
    grid = summary[
        [
            "label",
            "n_topics_mean",
            "c_v_mean",
            "c_v_sd",
            "c_npmi_mean",
            "c_npmi_sd",
            "diversity_mean",
            "stability",
            "eligible",
        ]
    ].round(3)
    add(to_markdown(grid))
    add("")
    add(
        _figure(
            "fig_rq1_1_coherence_by_config",
            "Coherence against number of topics for every configuration (shaded: eligible range)",
            snap,
        )
    )
    add(_figure("fig_rq1_2_model_comparison", "Best eligible configuration per contender", snap))
    add("## 3. Chosen model: topics\n")
    add(
        f"`{selection.winner_config}`, base seed {load_config()['random_seed']}. Sizes are numbers of incidents assigned (each incident gets one dominant topic). "
        "The *suggested label* is only the first three terms, an automatic placeholder, not an interpretation.\n"
    )
    shown = table[["topic_id", "n_incidents", "suggested_label", "top_10_terms"]].copy()
    shown["example_titles"] = [
        " || ".join(t.split(" || ")[:2]) for t in table["representative_titles"]
    ]
    add(to_markdown(shown))
    add("")
    add(_figure("fig_rq1_3_topic_sizes", "Incidents per topic (chosen model)", snap))
    add("## 4. Your action: confirm the topic labels\n")
    status_text = {
        "created": "was created",
        "unchanged": "already existed for this exact topic solution and was left untouched",
        "replaced": "was replaced (the previous file had no answers)",
        "proposed": "was NOT overwritten because it contains your answers for a different topic solution; the new template is `data/manual/topic_labels.proposed.csv`",
    }[label_status]
    add(
        f"`data/manual/topic_labels.csv` {status_text}. For each topic it lists the top 10 terms, five representative incident titles with ids, and an automatic placeholder label. "
        "**Fill `final_label`** (a short human name for the theme; add `notes` if a topic is incoherent or should be merged with another). "
        "Phase 4 will use your confirmed labels; the file carries a `model_key` so that your answers are never silently applied to a different topic solution.\n"
    )
    add("## 5. Limitations\n")
    add(
        "- **Coherence is a proxy for interpretability, not a measurement of it.** C_v and NPMI correlate with, but do not replace, human judgement; your label review is the real test, and a topic can score well and still be uninformative.\n"
        "- **Short texts.** Incidents are title + description (about 34 tokens after filtering), which limits all three families; co-occurrence statistics are sparse. Full report text could change the ranking (Phase 1 recommended it only as an ablation).\n"
        "- **Preprocessing choices matter.** The domain stopword list and frequency filter are judgement calls applied equally to all models; different choices would change absolute scores.\n"
        "- **BERTopic is not tuned to a topic count.** Its configurations are compared at whatever number of topics HDBSCAN yields, so contenders differ in k even inside the eligible range; coherence and diversity are not independent of k.\n"
        "- **Hard assignment.** Each incident is given one dominant topic; incidents about several themes are forced into one. "
        + _weight_sentence(fit)
        + "LDA/NMF weights are kept in the fits but the saved assignment is dominant-topic only.\n"
        "- **Reporting bias.** Themes describe what AIID editors chose to record, not the prevalence of harms (see the datasheet).\n"
        f"- **One snapshot, one corpus.** Topics are fitted on all {n(len(corpus.docs))} incidents (including pre-2015 and the partial {snap[:4]}); trend tests in Phase 4 use complete years only.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    logger.info("Wrote %s", path)
