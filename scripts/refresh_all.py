"""Refresh every result that depends on your manual files, with one command.

Usage:
    uv run python scripts/refresh_all.py                 # run everything (rules, Phases 2-7 + dashboard check)
    uv run python scripts/refresh_all.py --check         # only show the status of your manual files
    uv run python scripts/refresh_all.py --only phase4 phase6
    uv run python scripts/refresh_all.py --from phase4   # skip Phases 2-3 (manual files for them unchanged)

Steps: rules (D2/D3) -> phase2 (entity + duplicate review) -> phase3 (topics) -> phase4 (trends, topic labels) -> phase5 (network)
-> phase6 (classification, DistilBERT predictions, label reliability) -> phase7 (retrieval) -> dashboard.
A failed or blocked step blocks only the steps that depend on it. Writes reports/refresh_summary.md.
Exit code: 0 if no step failed or was blocked, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
import time

from aicidentia import pipeline
from aicidentia.config import configure_logging
from aicidentia.dashboard.status import manual_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check", action="store_true", help="only print the status of the manual files"
    )
    parser.add_argument("--only", nargs="+", metavar="STEP", help="run only these steps")
    parser.add_argument("--from", dest="start", metavar="STEP", help="start at this step")
    parser.add_argument(
        "--no-dashboard-check", action="store_true", help="skip the headless page check"
    )
    args = parser.parse_args(argv)

    configure_logging()
    status = manual_status()
    print(
        "\nManual inputs\n"
        + status[["input", "done", "total", "state"]].to_string(index=False)
        + "\n"
    )
    if args.check:
        return 0

    options = pipeline.RefreshOptions(dashboard_check=not args.no_dashboard_check)
    selected = pipeline.select(pipeline.steps(options), args.only, args.start)
    start = time.perf_counter()
    results = pipeline.run_steps(selected)
    pipeline.write_summary(results, manual_status())
    table = pipeline.summary_table(results)
    print("\nRefresh summary\n" + table.to_string(index=False, max_colwidth=110))
    print(
        f"\nFinished in {time.perf_counter() - start:.0f}s; details in reports/refresh_summary.md. "
        "In a running dashboard, press 'Reload data after a refresh'."
    )
    return 1 if any(r.status in ("failed", "blocked") for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
