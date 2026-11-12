"""Fine-tune DistilBERT on the local NVIDIA GPU with exactly the Colab protocol (docs/decisions.md, D5).

Usage:  uv run python scripts/train_distilbert_local.py            # full run (5 folds + time split)
        uv run python scripts/train_distilbert_local.py --smoke    # tiny end-to-end check

It imports the training code from scripts/colab_distilbert_code.py, the same file the Colab notebook embeds, so
folds, seeds, hyperparameters and the output format are identical. Input: reports/colab/rq3_colab_input.csv
(written by Phase 6). Output: reports/colab/rq3_distilbert_predictions.csv, where Phase 6 expects it, plus a
small run log (reports/colab/rq3_distilbert_run.json: device, versions, minutes). The model is loaded from the
local Hugging Face cache (HF_HOME in .env) and is never downloaded. Re-run Phase 6 afterwards.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time

from aicidentia.config import PROJECT_ROOT, configure_logging

COLAB_DIR = PROJECT_ROOT / "reports" / "colab"
INPUT = COLAB_DIR / "rq3_colab_input.csv"
OUTPUT = COLAB_DIR / "rq3_distilbert_predictions.csv"
RUN_LOG = COLAB_DIR / "rq3_distilbert_run.json"


def load_protocol():
    spec = importlib.util.spec_from_file_location(
        "colab_distilbert_code", PROJECT_ROOT / "scripts" / "colab_distilbert_code.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="a few incidents, one fold, one epoch")
    args = parser.parse_args(argv)
    configure_logging()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use the local cache only
    import torch
    import transformers

    if not torch.cuda.is_available() and not args.smoke:
        print("No CUDA GPU found; use the Colab notebook instead.", file=sys.stderr)
        return 1
    protocol = load_protocol()
    output = COLAB_DIR / "rq3_distilbert_smoke.csv" if args.smoke else OUTPUT
    started = time.time()
    protocol.run(str(INPUT), str(output), smoke=args.smoke)
    if not args.smoke:
        RUN_LOG.write_text(
            json.dumps(
                {
                    "where": "local GPU (scripts/train_distilbert_local.py)",
                    "device": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "transformers": transformers.__version__,
                    "python": platform.python_version(),
                    "minutes": round((time.time() - started) / 60, 1),
                    "protocol": {
                        "model": protocol.MODEL_NAME,
                        "max_len": protocol.MAX_LEN,
                        "epochs": protocol.EPOCHS,
                        "batch_size": protocol.BATCH_SIZE,
                        "learning_rate": protocol.LEARNING_RATE,
                        "weight_decay": protocol.WEIGHT_DECAY,
                        "warmup_fraction": protocol.WARMUP_FRACTION,
                        "base_seed": protocol.BASE_SEED,
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
