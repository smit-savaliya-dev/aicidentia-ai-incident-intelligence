"""Phase 7: similar-incident retrieval (RQ4) evaluated against editor-marked similar incidents.

Usage:  uv run python scripts/run_phase7_retrieval.py
Outputs: reports/results.json (rq4.*), reports/tables/rq4_*, reports/figures/fig_rq4_*, docs/rq4_similarity.md
"""

from __future__ import annotations

import time

from aicidentia.config import configure_logging
from aicidentia.models import retrieval_outputs


def main() -> None:
    configure_logging()
    start = time.perf_counter()
    results = retrieval_outputs.run_retrieval()
    retrieval_outputs.write_all(results)
    print(
        f"Best: {results.facts['best_method']} (MRR {results.facts['best_mrr']:.3f}) | deployed: {results.deployed}"
    )
    print(f"Phase 7 finished in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
