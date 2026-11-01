"""Phase 2: build the clean tables, manual-review templates, figures and docs.

Usage:  uv run python scripts/run_phase2_prepare.py
Re-run after editing config/entity_aliases.yaml, config/entity_types.yaml, or the manual files
data/manual/entity_review.csv and data/manual/duplicate_review.csv.
Outputs: data/processed/*.parquet, reports/results.json (prep.*), reports/tables/prep_*,
         reports/figures/fig_eda_*, docs/data_preparation.md
"""

from __future__ import annotations

import time

from aicidentia.config import configure_logging
from aicidentia.data.prepare import build_processed
from aicidentia.data.prepare_outputs import write_all


def main() -> None:
    configure_logging()
    start = time.perf_counter()
    results = build_processed()
    write_all(results)
    print(f"Phase 2 preparation finished in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
