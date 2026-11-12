"""Answer the duplicate and entity review files by rule (docs/decisions.md, D2 and D3).

Usage:  uv run python scripts/apply_rules.py
Rewrites data/manual/duplicate_review.csv and data/manual/entity_review.csv (decision / is_named_person and notes)
and records the counts under `rules.*` in reports/results.json. Re-run Phase 2 afterwards (refresh_all.py does).
"""

from __future__ import annotations

from aicidentia import registry
from aicidentia.config import configure_logging
from aicidentia.data import rules


def main() -> None:
    configure_logging()
    facts = rules.apply_all()
    for group, values in facts.items():
        for key, value in values.items():
            registry.record(
                f"rules.{group}.{key}", value, "data/manual", "aicidentia.data.rules.apply_all"
            )
    for group, values in facts.items():
        print(group, values)


if __name__ == "__main__":
    main()
