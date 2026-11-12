"""Cross-question story outputs (runs after Phases 4-7): ChatGPT shift, risk priority matrix, synthesis and
docs/key_findings.md. Registry namespace `story.*`; tables `story_*`; figures `fig_story_*`."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from aicidentia import registry
from aicidentia.config import PROJECT_ROOT, load_config
from aicidentia.data import clean
from aicidentia.models import story as S
from aicidentia.viz import plots, style

logger = logging.getLogger(__name__)
PRODUCER = "aicidentia.models.story_outputs.run_story"
SOURCE = "data/processed/incidents_clean.parquet"
TABLES = PROJECT_ROOT / "reports" / "tables"


@dataclass
class StoryResults:
    chatgpt_themes: dict[str, Any]
    chatgpt_domains: dict[str, Any]
    matrix: pd.DataFrame
    matrix_facts: dict[str, Any]
    synthesis: pd.DataFrame
    window_start: int
    facts: dict[str, Any] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)


def theme_frame() -> pd.DataFrame:
    from aicidentia.models.themes import load_theme_frame

    return load_theme_frame().data[["incident_id", "topic_id", "theme"]]


def run_story() -> StoryResults:
    inc = clean.load_processed("incidents_clean")
    themes = theme_frame()
    start = int(registry.get("audit.year_range_recommendation")["start"])
    window = inc[inc["year"] >= start].merge(themes, on="incident_id")
    period = S.period_of(window["date"])
    t = S.chi_square_shift(window["theme"], period)
    labelled = window[window["mit_risk_domain"].notna()]
    d = S.chi_square_shift(labelled["mit_risk_domain"], S.period_of(labelled["date"]))
    after = window[period == "after"]
    d["mit_coverage_before"] = float(window.loc[period == "before", "has_mit"].mean())
    d["mit_coverage_after"] = float(after["has_mit"].mean())
    trends = pd.read_csv(TABLES / "rq1b_trends_themes.csv")
    matrix, mfacts = S.priority_matrix(inc, themes, trends)
    emerging = list(registry.get("rq1b.emerging_themes"))
    centrality = pd.read_csv(TABLES / "rq2_centrality.csv")
    synth = S.synthesis(
        themes,
        inc,
        clean.load_processed("incident_entities"),
        clean.load_processed("entity_communities"),
        centrality,
        emerging,
    )
    res = StoryResults(t, d, matrix, mfacts, synth, start)
    res.facts = _facts(res)
    res.findings = key_findings(res)
    return res


def _shift_facts(r: dict[str, Any]) -> dict[str, Any]:
    keep = ["chi2", "dof", "p_value", "cramers_v", "n", "n_before", "n_after", "min_expected"]
    out = {k: r[k] for k in keep}
    out["table"] = r["table"].round(4)
    for k in ("mit_coverage_before", "mit_coverage_after"):
        if k in r:
            out[k] = r[k]
    return out


def _facts(res: StoryResults) -> dict[str, Any]:
    dom = res.chatgpt_domains["table"].set_index("group")
    mal = dom.loc["Malicious Actors & Misuse"]
    return {
        "chatgpt_cutoff": S.CHATGPT_DATE.strftime("%Y-%m-%d"),
        "chatgpt_window_start": res.window_start,
        "chatgpt_themes": _shift_facts(res.chatgpt_themes),
        "chatgpt_domains": _shift_facts(res.chatgpt_domains),
        "chatgpt_malicious_share_before": float(mal["share_before"]),
        "chatgpt_malicious_share_after": float(mal["share_after"]),
        "matrix": res.matrix.round(4),
        **{f"matrix_{k}": v for k, v in res.matrix_facts.items()},
        "synthesis": res.synthesis.round(4),
    }


# --------------------------------------------------------------------------------------
# Key findings (one sentence, number, results.json key, limitation, dashboard page)
# --------------------------------------------------------------------------------------


def key_findings(res: StoryResults) -> list[dict[str, Any]]:
    get = registry.get
    f = res.facts
    ct, cd = f["chatgpt_themes"], f["chatgpt_domains"]
    rep = get("rq4.repeat")
    dep = get("rq4.deployed")
    found = rep["found_in_top5"][dep]
    sim = get("rq3rel.pairs_similar")
    kap = get("rq3rel.mapping_kappa")
    emerging = list(get("rq1b.emerging_themes"))
    big = res.synthesis[res.synthesis["community"] == res.synthesis["community"].mode().iloc[0]]
    orgs = big["alleged_central_organisations"].iloc[0].split("; ")[:2]
    return [
        {
            "title": "AI harm shifted from failures to misuse after ChatGPT",
            "sentence": (
                f"Malicious Actors & Misuse rose from {f['chatgpt_malicious_share_before']:.0%} to "
                f"{f['chatgpt_malicious_share_after']:.0%} of MIT-labelled incidents after 30 Nov 2022, the largest "
                f"shift in the risk-domain mix (Cramer's V {cd['cramers_v']:.2f}; themes V {ct['cramers_v']:.2f})."
            ),
            "number": f"{f['chatgpt_malicious_share_before']:.0%} → {f['chatgpt_malicious_share_after']:.0%}",
            "key": "story.chatgpt_malicious_share_after; story.chatgpt_domains",
            "limitation": (
                "Reported incidents only; MIT coverage is lower after the cut-off "
                f"({cd['mit_coverage_after']:.0%} vs {cd['mit_coverage_before']:.0%}) and AIID's editorial intake changed over the same years."
            ),
            "page": "Themes and Trends",
        },
        {
            "title": f"{len(emerging)} of {get('rq1b.n_themes')} risk themes are emerging",
            "sentence": (
                "Deepfake videos and hoaxes, hallucinated legal citations, deepfake impersonation scams, AI nude "
                "images in schools and AI election disinformation all grow significantly as a share of incidents "
                "(Mann-Kendall with Benjamini-Hochberg, 2015-2025)."
            ),
            "number": f"{len(emerging)} emerging themes",
            "key": "rq1b.emerging_themes; rq1b.themes_share_trends",
            "limitation": "Eleven complete years give little power; shares describe the database, not the real-world frequency of harm.",
            "page": "Themes and Trends",
        },
        {
            "title": "One in six recent incidents repeats a known failure, and search finds the precedent",
            "sentence": (
                f"{rep['share_with_earlier_precedent']:.0%} of incidents since {rep['from_year']} have an editor-linked earlier "
                f"precedent, and searching only earlier incidents puts it in the top 5 for {found['hit@5']:.0%} of them; "
                f"at depth 3 our search scores MRR {get('rq4.mrr3')[dep]:.2f} against {get('rq4.aiid_nlp_rr3'):.3f} for AIID's built-in suggestions."
            ),
            "number": f"{found['hit@5']:.0%} found in top 5",
            "key": "rq4.repeat; rq4.mrr3; rq4.aiid_nlp_rr3",
            "limitation": "Editors link only some incidents, so the share with a precedent is a lower bound and the search is scored on linked incidents only.",
            "page": "Precedent Finder",
        },
        {
            "title": "The AI-generated labels are usable but about 1 in 7 is doubtful",
            "sentence": (
                f"Editor-linked similar incidents share an MIT domain {sim['same_domain_rate']:.0%} of the time against "
                f"{sim['baseline_rate']:.0%} by chance, and the discrimination label agrees substantially with human CSETv1 "
                f"annotation (kappa {kap['discrimination_vs_protected_characteristic']:.2f}), yet confident learning "
                f"estimates {get('rq3rel.cl_noise_rate'):.0%} of labels are wrong."
            ),
            "number": f"{get('rq3rel.cl_noise_rate'):.0%} doubtful labels",
            "key": "rq3rel.pairs_similar; rq3rel.mapping_kappa; rq3rel.cl_noise_rate",
            "limitation": "Human fields map only partially onto MIT domains and cover pre-2024 incidents; confident learning inherits the classifier's own errors.",
            "page": "Label Reliability",
        },
        {
            "title": "Deepfake, scam and election harms concentrate around the big platforms",
            "sentence": (
                f"{len(big)} of the {len(emerging)} emerging themes concentrate in network community {int(big['community'].iloc[0])}, "
                f"where {big['share_in_community'].min():.0%}-{big['share_in_community'].max():.0%} of their incidents involve its entities "
                f"(about {big['lift'].mean():.1f} times the rate for all incidents), "
                f"with {' and '.join(orgs)} as the most central allegedly involved organisations."
            ),
            "number": f"{big['share_in_community'].min():.0%}-{big['share_in_community'].max():.0%} in one community",
            "key": "story.synthesis; rq2.community_profile",
            "limitation": "Organisations are alleged deployers or developers, centrality reflects how often they are reported (not blame), and communities depend on a few hub entities.",
            "page": "Risk Network",
        },
    ]


def write_key_findings(findings: list[dict[str, Any]], path: Any = None) -> None:
    path = path or PROJECT_ROOT / "docs" / "key_findings.md"
    snap = load_config()["snapshot_date"]
    lines = [
        "# Key findings\n",
        f"_Generated by `aicidentia.models.story_outputs` from `reports/results.json` (AIID snapshot {snap}). "
        "Every number can be traced through its results.json key._\n",
    ]
    for i, f in enumerate(findings, 1):
        lines += [
            f"## {i}. {f['title']}\n",
            f"{f['sentence']}\n",
            f"- **Number:** {f['number']}",
            f"- **results.json:** `{'`, `'.join(k.strip() for k in f['key'].split(';'))}`",
            f"- **Limitation:** {f['limitation']}",
            f"- **Dashboard page:** {f['page']}\n",
        ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------------------
# Artefacts
# --------------------------------------------------------------------------------------


def register(res: StoryResults) -> None:
    for key, value in res.facts.items():
        registry.record(f"story.{key}", value, SOURCE, PRODUCER)
    registry.record("story.key_findings", res.findings, "docs/key_findings.md", PRODUCER)


def save_tables(res: StoryResults) -> None:
    style.save_table(res.chatgpt_themes["table"].round(4), "story_chatgpt_themes")
    style.save_table(res.chatgpt_domains["table"].round(4), "story_chatgpt_domains")
    style.save_table(res.matrix.round(4), "story_risk_matrix")
    style.save_table(res.synthesis.round(4), "story_synthesis")


def make_figures(res: StoryResults) -> None:
    snap = str(load_config()["snapshot_date"])
    cutoff = S.CHATGPT_DATE.strftime("%d %b %Y")
    figs = {
        "fig_story_1_chatgpt_shift": plots.plot_chatgpt_slopes(
            res.chatgpt_themes["table"], res.chatgpt_domains["table"], cutoff
        ),
        "fig_story_2_risk_matrix": plots.plot_risk_matrix(
            res.matrix, res.matrix_facts["severity_overall"], res.matrix_facts["min_cset_per_theme"]
        ),
    }
    for name, fig in figs.items():
        style.save_figure(fig, name, snap)


def write_all(res: StoryResults) -> None:
    register(res)
    save_tables(res)
    make_figures(res)
    write_key_findings(res.findings)
