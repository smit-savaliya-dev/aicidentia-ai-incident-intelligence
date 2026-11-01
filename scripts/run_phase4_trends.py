"""Phase 4: theme and MIT-domain trends, reporting-bias sensitivity and a forecasting evaluation.

Usage:  uv run python scripts/run_phase4_trends.py
Uses your confirmed labels in data/manual/topic_labels.csv (blank labels give PROVISIONAL placeholder
theme names; identical final labels merge topics). Re-run after you label or merge topics.
Outputs: reports/results.json (rq1b.*), reports/tables/rq1b_*, reports/figures/fig_rq1b_*, docs/rq1_trends.md
"""

from __future__ import annotations

import time

from aicidentia.config import configure_logging
from aicidentia.models import trend_outputs


def main() -> None:
    configure_logging()
    start = time.perf_counter()
    results = trend_outputs.run_trends()
    trend_outputs.write_all(results)
    state = (
        "PROVISIONAL (topic labels not all confirmed)"
        if results.facts["provisional"]
        else "labels confirmed"
    )
    print(f"Emerging themes: {results.facts['emerging_themes']} | {state}")
    print(f"Phase 4 finished in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
