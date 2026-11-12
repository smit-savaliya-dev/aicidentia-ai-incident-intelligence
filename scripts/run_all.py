"""Regenerate every result from data/raw with one command, and print a timing summary.

Usage:  uv run python scripts/run_all.py                    # everything (about 5 minutes with warm caches)
        uv run python scripts/run_all.py --no-dashboard-check
        uv run python scripts/run_all.py --clear-cache      # also delete data/interim (recompute embeddings,
                                                            # topic fits and CV runs; much slower)

Order: verify the raw files against docs/data_manifest.md (SHA-256) -> Phase 1 audit -> rules D2/D3 ->
Phases 2-7 -> story (key findings) -> headless dashboard check. Every step is deterministic (fixed seeds, fixed
folds, figures without save dates), so a rerun leaves git unchanged; the script reports whether it did.
DistilBERT is not retrained (GPU kernels are not bit-for-bit reproducible): Phase 6 scores the saved predictions in
reports/colab/rq3_distilbert_predictions.csv (retrain with scripts/train_distilbert_local.py).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import time

from aicidentia import pipeline
from aicidentia.config import PROJECT_ROOT, configure_logging, load_config

MANIFEST = PROJECT_ROOT / "docs" / "data_manifest.md"


def verify_raw() -> str:
    """Every file listed in the manifest exists in data/raw with the recorded SHA-256."""
    raw = PROJECT_ROOT / load_config()["paths"]["raw"]
    text = MANIFEST.read_text(encoding="utf-8")
    excel = re.findall(r"\| `([^`]+\.xlsx)` \| \w+ \| \d* \| `([0-9a-f]{64})`", text)
    dump_dir = re.search(r"## Extracted snapshot directory `([^`]+)/`", text)
    dump = re.findall(r"\| `([^`]+)` \| \d+ \| `([0-9a-f]{64})` \|", text)
    files = [(raw / name, sha) for name, sha in excel]
    if dump_dir:
        files += [(raw / dump_dir.group(1) / name, sha) for name, sha in dump]
    bad = []
    for path, sha in files:
        if not path.exists():
            bad.append(f"missing {path.relative_to(PROJECT_ROOT)}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != sha:
            bad.append(f"changed {path.relative_to(PROJECT_ROOT)}")
    if bad:
        raise RuntimeError("raw data differ from docs/data_manifest.md: " + "; ".join(bad[:5]))
    return f"{len(files)} raw files match docs/data_manifest.md"


def run_phase1() -> str:
    from aicidentia.data.audit import run_audit
    from aicidentia.data.audit_outputs import write_all

    write_all(run_audit())
    return "data audit, datasheet, audit tables and figures"


def git_state() -> dict[str, str]:
    """path -> content fingerprint for every file git reports as changed or untracked (content, not just status),
    so a rerun that rewrites an already-modified file differently is detected."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    state = {}
    for line in out.stdout.splitlines():
        path = line[3:].strip().strip('"')
        target = PROJECT_ROOT / path
        state[path] = (
            hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else line[:2]
        )
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--no-dashboard-check", action="store_true")
    parser.add_argument("--clear-cache", action="store_true", help="delete data/interim first")
    args = parser.parse_args(argv)
    configure_logging()
    if args.clear_cache:
        interim = PROJECT_ROOT / "data" / "interim"
        for item in interim.iterdir():
            if item.name != ".gitkeep":
                shutil.rmtree(item) if item.is_dir() else item.unlink()
    before = git_state()
    steps = [
        pipeline.Step("verify_raw", "Verify raw files (SHA-256)", verify_raw),
        pipeline.Step("phase1", "Phase 1 data audit", run_phase1, ("verify_raw",)),
    ]
    rest = pipeline.steps(pipeline.RefreshOptions(dashboard_check=not args.no_dashboard_check))
    rest[0].requires = ("phase1",)
    start = time.perf_counter()
    results = pipeline.run_steps(steps + rest)
    total = time.perf_counter() - start
    table = pipeline.summary_table(results)
    print("\nrun_all summary\n" + table.to_string(index=False, max_colwidth=100))
    print(f"\nTotal {total / 60:.1f} min ({total:.0f} s)")
    after = git_state()
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    if not changed:
        print("git: unchanged by this run (outputs reproduced exactly)")
    else:
        print(f"git: {len(changed)} paths changed by this run:")
        for path in changed[:30]:
            print("  " + path)
    failed = any(r.status in ("failed", "blocked") for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
