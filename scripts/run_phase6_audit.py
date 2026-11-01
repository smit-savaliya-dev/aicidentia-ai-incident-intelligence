"""Phase 6 (second step): analyse YOUR blind annotations of the 150-incident audit sample.

Usage:  uv run python scripts/run_phase6_audit.py
Does nothing (and says so) until data/manual/label_audit_sample.csv contains your labels.
Outputs: reports/results.json (rq3audit.*), reports/tables/rq3audit_*, reports/figures/fig_rq3audit_*, docs/rq3_label_audit.md
"""

from __future__ import annotations

from aicidentia.config import configure_logging
from aicidentia.models import classify as C
from aicidentia.models import label_audit_outputs as out


def main() -> None:
    configure_logging()
    result = out.run_audit_analysis()
    if result is None:
        print(
            "No annotations yet: fill my_label and my_confidence in data/manual/label_audit_sample.csv "
            "(or run `uv run streamlit run scripts/label_audit_app.py`), then re-run this script."
        )
        return
    out.write_all(result, C.load_dataset(with_embeddings=False).classes)
    print(
        f"Annotated: {result['n_annotated']} | kappa {result['kappa_raw']:.3f} ({result['kappa_raw_band']}) | "
        f"design-weighted kappa {result['kappa_weighted']:.3f} ({result['kappa_weighted_band']})"
    )


if __name__ == "__main__":
    main()
