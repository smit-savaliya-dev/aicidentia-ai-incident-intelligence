"""Phase 5: incident network - centrality, communities, temporal evolution and dashboard export.

Usage:  uv run python scripts/run_phase5_network.py
Outputs: reports/results.json (rq2.*), reports/tables/rq2_*, reports/figures/fig_rq2_*,
         reports/network_graph.json (dashboard), data/processed/entity_communities.parquet, docs/rq2_network.md
"""

from __future__ import annotations

import time

from aicidentia.config import configure_logging
from aicidentia.models import network_outputs


def main() -> None:
    configure_logging()
    start = time.perf_counter()
    results = network_outputs.run_network()
    network_outputs.write_all(results)
    print(
        f"Communities: {results.chosen_method}, {results.facts['n_communities']} (modularity {results.facts['chosen_modularity']:.3f})"
    )
    print(f"Phase 5 finished in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
