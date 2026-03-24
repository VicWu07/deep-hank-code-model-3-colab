"""Run training with structured runtime logging.

Usage:
  .venv/bin/python scripts/run_train.py
  .venv/bin/python scripts/run_train.py --quick
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deep_hank.config import default_config
from deep_hank.train import train


def _quick_overrides(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["batch_size"] = 24
    cfg["phase0_steps"] = 20
    cfg["phase1_steps"] = 20
    cfg["phase2_steps"] = 20
    cfg["phase3_steps"] = 4
    cfg["sim_path_steps"] = 24
    cfg["sim_refresh_every"] = 10
    cfg["sim_mix_frac"] = 0.3
    cfg["w_bisect_steps"] = 8
    cfg["w_newton_steps"] = 1
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep HANK training runner with JSONL logs.")
    parser.add_argument("--quick", action="store_true", help="Use short smoke-test training config.")
    parser.add_argument("--log-every", type=int, default=10, help="Emit one training log line every N steps.")
    parser.add_argument("--debug-metrics", action="store_true", help="Include additional phase-3 debug metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config()
    if args.quick:
        cfg = _quick_overrides(cfg)

    cfg["enable_step_logging"] = True
    cfg["log_every_steps"] = int(args.log_every)
    cfg["enable_debug_metrics"] = bool(args.debug_metrics)
    cfg["debug_metrics_batch_cap"] = 8

    outdir = Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "train_runtime.jsonl"
    summary_path = outdir / "run_summary.json"

    print(f"[train] Output directory: {outdir}")
    print(f"[train] Runtime log: {log_path}")
    print("[train] Starting training...")

    with log_path.open("w", encoding="utf-8", buffering=1) as f:

        def log_fn(payload: dict) -> None:
            f.write(json.dumps(payload) + "\n")

        _model, history = train(cfg, log_fn=log_fn)

    summary = {
        "steps": len(history),
        "final_phase": history[-1]["phase"] if history else None,
        "final_step": history[-1]["step"] if history else None,
        "final_loss": history[-1]["loss"] if history else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[train] Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()

