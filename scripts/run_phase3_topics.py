"""Phase 3: fit and compare LDA, NMF and BERTopic; write tables, figures, assignments, label template and docs.

Usage:  uv run python scripts/run_phase3_topics.py
Fits are cached in data/interim/, so re-running is fast unless the corpus, parameters or seeds change.
Outputs: data/processed/incident_topics.parquet, data/manual/topic_labels.csv (blank final_label for you),
         reports/results.json (rq1a.*), reports/tables/rq1a_*, reports/figures/fig_rq1_*, docs/rq1_topic_models.md
"""

from __future__ import annotations

import time

from aicidentia.config import configure_logging
from aicidentia.data import clean
from aicidentia.features.text import build_corpus
from aicidentia.models import topic_experiment as E
from aicidentia.models import topic_outputs


def main() -> None:
    configure_logging()
    start = time.perf_counter()
    incidents = clean.load_processed("incidents_clean")
    corpus = build_corpus(incidents["incident_id"], incidents["text"])
    result = E.run_experiment(corpus)
    selection = E.select_model(result.summary)
    out = topic_outputs.write_all(corpus, result, selection)
    print(f"Selected {selection.winner_config}; topic_labels.csv {out['label_status']}")
    print(f"Phase 3 finished in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
