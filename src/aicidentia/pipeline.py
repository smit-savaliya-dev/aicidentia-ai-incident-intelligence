"""Re-run every phase that depends on the manual inputs, in dependency order (`scripts/refresh_all.py`).

Order and dependencies
----------------------
phase2 (applies entity_review.csv and duplicate_review.csv)
  -> phase3 (topic models; cached, so it only refits if phase2 changed the incidents)
       -> phase4 (trends; applies topic_labels.csv)
  -> phase5 (network)
  -> phase6 (classification; scores the Colab DistilBERT predictions if present and valid)
  -> phase7 (retrieval)
-> dashboard (headless check that every page renders)

A failed step blocks the steps that depend on it; the others still run. Nothing here edits a manual file.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from aicidentia.config import PROJECT_ROOT

logger = logging.getLogger(__name__)
SUMMARY_PATH = PROJECT_ROOT / "reports" / "refresh_summary.md"


class StepSkipped(Exception):
    """Raised by a step that has nothing to do (not an error)."""


class StepBlocked(Exception):
    """Raised by a step that cannot run until you act (e.g. stale topic labels)."""


@dataclass
class Step:
    name: str
    label: str
    run: Callable[[], str]
    requires: tuple[str, ...] = ()


@dataclass
class StepResult:
    name: str
    label: str
    status: str  # ok | skipped | blocked | failed | not run
    seconds: float = 0.0
    note: str = ""


@dataclass
class RefreshOptions:
    dashboard_check: bool = True
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# The steps (same calls as the per-phase scripts)
# --------------------------------------------------------------------------------------


def run_phase2() -> str:
    from aicidentia.data.prepare import build_processed
    from aicidentia.data.prepare_outputs import write_all

    res = build_processed()
    write_all(res)
    f = getattr(res, "facts", {}) or {}
    ent, dup = f.get("entity_review", {}), f.get("near_duplicate_review", {})
    return (
        f"{f.get('incidents_clean', '?')} incidents; entity review {ent.get('answered', 0)}/{ent.get('rows', 0)} answered; "
        f"duplicate review {dup.get('answered', 0)}/{dup.get('rows', 0)} answered, {f.get('user_merges_applied', 0)} merges applied"
    )


def run_phase3() -> str:
    from aicidentia.data import clean
    from aicidentia.features.text import build_corpus
    from aicidentia.models import topic_experiment as E
    from aicidentia.models import topic_outputs

    incidents = clean.load_processed("incidents_clean")
    corpus = build_corpus(incidents["incident_id"], incidents["text"])
    result = E.run_experiment(corpus)
    selection = E.select_model(result.summary)
    out = topic_outputs.write_all(corpus, result, selection)
    note = f"{selection.winner_config}; topic_labels.csv {out['label_status']}"
    if out["label_status"] == "proposed":
        note += " (the topic solution changed: your labels no longer apply, see topic_labels.proposed.csv)"
    return note


def run_phase4() -> str:
    from aicidentia.models import trend_outputs
    from aicidentia.models.themes import StaleLabelsError

    try:
        res = trend_outputs.run_trends()
    except StaleLabelsError as exc:
        raise StepBlocked(f"topic labels are stale: {exc}") from exc
    trend_outputs.write_all(res)
    f = res.facts
    state = "PROVISIONAL (labels not all confirmed)" if f["provisional"] else "labels confirmed"
    return f"{state}; emerging themes: {len(f['emerging_themes'])}; forecasting justified: {f['forecast_justified']}"


def run_phase5() -> str:
    from aicidentia.models import network_outputs

    res = network_outputs.run_network()
    network_outputs.write_all(res)
    return f"{res.chosen_method}, {res.facts['n_communities']} communities, modularity {res.facts['chosen_modularity']:.3f}"


def run_phase6() -> str:
    from aicidentia.models import classify_outputs

    res = classify_outputs.run_classification()
    classify_outputs.write_all(res)
    f = res.facts
    colab = (
        "DistilBERT scored"
        if res.colab_present
        else (
            "Colab predictions NOT used: " + f["colab_problem"]
            if f.get("colab_problem")
            else "DistilBERT pending (no Colab file)"
        )
    )
    return f"deployed {res.deployable} ({f['calibration_method']}); {colab}"


def run_phase7() -> str:
    from aicidentia.models import retrieval_outputs

    res = retrieval_outputs.run_retrieval()
    retrieval_outputs.write_all(res)
    return f"best {res.facts['best_method']} (MRR {res.facts['best_mrr']:.3f}); deployed {res.deployed}"


def check_dashboard() -> str:
    """Render every page headlessly; fail if any page raises."""
    from streamlit.testing.v1 import AppTest

    pages = [
        PROJECT_ROOT / "app" / "Home.py",
        *sorted((PROJECT_ROOT / "app" / "pages").glob("*.py")),
    ]
    problems = []
    for page in pages:
        at = AppTest.from_file(str(page), default_timeout=300).run()
        if at.exception:
            problems.append(f"{page.name}: {at.exception[0].value[:200]}")
    if problems:
        raise RuntimeError("; ".join(problems))
    return f"{len(pages)} pages render"


def steps(options: RefreshOptions | None = None) -> list[Step]:
    options = options or RefreshOptions()
    out = [
        Step("phase2", "Phase 2 data preparation (entity + duplicate review)", run_phase2),
        Step(
            "phase3",
            "Phase 3 topic models (refits only if the data changed)",
            run_phase3,
            ("phase2",),
        ),
        Step("phase4", "Phase 4 trends (topic labels)", run_phase4, ("phase3",)),
        Step("phase5", "Phase 5 network", run_phase5, ("phase2",)),
        Step("phase6", "Phase 6 classification (+ Colab predictions)", run_phase6, ("phase2",)),
        Step("phase7", "Phase 7 retrieval", run_phase7, ("phase2",)),
    ]
    if options.dashboard_check:
        out.append(
            Step(
                "dashboard",
                "Dashboard check (all pages render)",
                check_dashboard,
                ("phase4", "phase5", "phase6", "phase7"),
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------


def run_steps(selected: list[Step], all_names: list[str] | None = None) -> list[StepResult]:
    """Run `selected` in order. A step whose requirement failed or was blocked in THIS run is blocked;
    requirements that were not selected are assumed to be up to date."""
    done: dict[str, str] = {}
    results = []
    for step in selected:
        bad = [r for r in step.requires if done.get(r) in ("failed", "blocked")]
        if bad:
            results.append(
                StepResult(step.name, step.label, "blocked", note=f"needs {', '.join(bad)}")
            )
            done[step.name] = "blocked"
            continue
        start = time.perf_counter()
        try:
            note, status = step.run(), "ok"
        except StepSkipped as exc:
            note, status = str(exc), "skipped"
        except StepBlocked as exc:
            note, status = str(exc), "blocked"
        except Exception as exc:  # report and carry on with independent steps
            logger.exception("Step %s failed", step.name)
            note, status = f"{type(exc).__name__}: {exc}"[:400], "failed"
        results.append(StepResult(step.name, step.label, status, time.perf_counter() - start, note))
        done[step.name] = status
    return results


def select(
    all_steps: list[Step], only: list[str] | None = None, start: str | None = None
) -> list[Step]:
    names = [s.name for s in all_steps]
    for name in (only or []) + ([start] if start else []):
        if name not in names:
            raise ValueError(f"unknown step {name!r}; choose from {names}")
    if only:
        return [s for s in all_steps if s.name in only]
    if start:
        return all_steps[names.index(start) :]
    return all_steps


def summary_table(results: list[StepResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"step": r.name, "status": r.status, "seconds": round(r.seconds, 1), "note": r.note}
            for r in results
        ]
    )


def write_summary(results: list[StepResult], status_table: pd.DataFrame, path=SUMMARY_PATH) -> None:
    from aicidentia.viz.style import to_markdown

    stamp = time.strftime("%Y-%m-%d %H:%M")
    text = (
        f"# Last refresh ({stamp})\n\n_Written by `scripts/refresh_all.py`._\n\n## Manual inputs\n\n{to_markdown(status_table)}\n\n"
        f"## Steps\n\n{to_markdown(summary_table(results))}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
