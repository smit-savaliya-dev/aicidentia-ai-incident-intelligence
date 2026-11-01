"""Export uv.lock to requirements.txt (pip fallback), including the PyTorch CUDA index.

`uv export` does not emit the explicit PyTorch index, so `torch==X+cu132` would not resolve
with pip. This script prepends the matching --extra-index-url line.

Usage:  uv run python scripts/export_requirements.py
"""

from __future__ import annotations

import subprocess
import tomllib

from aicidentia.config import PROJECT_ROOT


def torch_index_url() -> str:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as fh:
        indexes = tomllib.load(fh)["tool"]["uv"]["index"]
    return next(i["url"] for i in indexes if i["name"].startswith("pytorch"))


def main() -> None:
    out = PROJECT_ROOT / "requirements.txt"
    exported = subprocess.run(
        ["uv", "export", "--all-extras", "--no-hashes", "--no-emit-project", "--no-header"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    header = (
        "# Exported from uv.lock by scripts/export_requirements.py - do not edit by hand.\n"
        "# Install with: pip install -r requirements.txt && pip install -e . --no-deps\n"
        f"--extra-index-url {torch_index_url()}\n"
    )
    out.write_text(header + exported, encoding="utf-8", newline="\n")
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
