"""Phase 1: run the data audit and write registry entries, tables, figures and docs.

Usage:  uv run python scripts/run_phase1_audit.py
Outputs: reports/results.json (audit.*), reports/tables/audit_*, reports/figures/fig_audit_*,
         docs/data_audit.md, docs/datasheet.md
"""

from __future__ import annotations

import time

from aicidentia.config import configure_logging
from aicidentia.data.audit import run_audit
from aicidentia.data.audit_outputs import write_all


def main() -> None:
    configure_logging()
    start = time.perf_counter()
    results = run_audit()
    write_all(results)
    print(f"Phase 1 audit finished in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
