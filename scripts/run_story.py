"""Cross-question story analyses (after Phases 4-7): what changed after ChatGPT, the risk priority matrix,
themes x network communities x risk domains, and docs/key_findings.md.

Usage:  uv run python scripts/run_story.py
Outputs: reports/results.json (story.*), reports/tables/story_*, reports/figures/fig_story_*, docs/key_findings.md
"""

from __future__ import annotations

from aicidentia.config import configure_logging
from aicidentia.pipeline import run_story


def main() -> None:
    configure_logging()
    print(run_story())


if __name__ == "__main__":
    main()
