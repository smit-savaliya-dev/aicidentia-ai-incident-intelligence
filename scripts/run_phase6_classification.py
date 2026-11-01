"""Phase 6: classify incidents into MIT risk domains, evaluate, calibrate, save the model, create the audit sample.

Usage:  uv run python scripts/run_phase6_classification.py
Re-run after copying the Colab output to reports/colab/rq3_distilbert_predictions.csv to add DistilBERT.
Outputs: reports/results.json (rq3.*), reports/tables/rq3_*, reports/figures/fig_rq3_*, docs/rq3_classification.md,
         docs/label_guidelines.md, data/manual/label_audit_sample.csv (blank answers for you; never overwritten),
         reports/colab/rq3_colab_input.csv, models/rq3_domain_classifier.joblib, data/processed/rq3_domain_predictions.parquet
"""

from __future__ import annotations

import time

from aicidentia.config import configure_logging
from aicidentia.models import classify_outputs


def main() -> None:
    configure_logging()
    start = time.perf_counter()
    results = classify_outputs.run_classification()
    classify_outputs.write_all(results)
    cv = results.metrics[results.metrics["setting"] == "cv"].sort_values(
        "macro_f1", ascending=False
    )
    print(cv[["model", "macro_f1", "ci_low", "ci_high"]].round(3).to_string(index=False))
    print(
        f"Deployable: {results.deployable} | audit sample: {results.facts['audit_sample_status']} | DistilBERT: {'scored' if results.colab_present else 'pending Colab'}"
    )
    print(f"Phase 6 finished in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
