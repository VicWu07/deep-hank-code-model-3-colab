"""Run full training + diagnostics in one command.

Usage:
  .venv/bin/python scripts/run_full_training.py
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import jax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deep_hank.config import default_config
from deep_hank.diagnostics import run_all_visual_diagnostics, write_report
from deep_hank.train import train


def main() -> None:
    cfg = default_config()
    outdir = Path("full_runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "train_runtime.jsonl"
    ckpt_dir = outdir / "checkpoints"

    print(f"[full] Output directory: {outdir}")
    with log_path.open("w", encoding="utf-8", buffering=1) as f:

        def log_fn(payload: dict) -> None:
            f.write(json.dumps(payload) + "\n")

        model, history = train(cfg, log_fn=log_fn, checkpoint_dir=ckpt_dir, resume=False)

    key = jax.random.PRNGKey(int(cfg["seed"]) + 321)
    stats = run_all_visual_diagnostics(model, history, cfg, outdir, key)
    write_report(outdir, cfg, stats)
    print(f"[full] Completed. Diagnostics at: {outdir}")


if __name__ == "__main__":
    main()
