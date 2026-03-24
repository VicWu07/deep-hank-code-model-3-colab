"""Run quick visual diagnostics for Deep HANK.

Usage:
  .venv/bin/python scripts/run_diagnostics.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import json

import jax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deep_hank.config import default_config
from deep_hank.diagnostics import run_all_visual_diagnostics, write_report
from deep_hank.train import train


def quick_cfg() -> dict:
    cfg = default_config()
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
    cfg["diag_heatmap_a_points"] = 12
    cfg["diag_heatmap_z_points"] = 10
    cfg["diag_heatmap_include_distribution"] = False
    cfg["diag_wage_samples"] = 24
    cfg["diag_sim_steps"] = 28
    cfg["enable_step_logging"] = True
    cfg["log_every_steps"] = 10
    cfg["enable_debug_metrics"] = True
    cfg["debug_metrics_batch_cap"] = 4
    return cfg


def main() -> None:
    cfg = quick_cfg()
    outdir = Path("diagnostics") / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "train_runtime.log"

    print(f"[diag] Output directory: {outdir}")
    print(f"[diag] Runtime log: {log_path}")
    print("[diag] Starting quick train + diagnostics...")

    with log_path.open("w", encoding="utf-8", buffering=1) as f:
        def log_fn(payload: dict) -> None:
            f.write(json.dumps(payload) + "\n")

        model, history = train(cfg, log_fn=log_fn)

    key = jax.random.PRNGKey(cfg["seed"] + 123)
    stats = run_all_visual_diagnostics(model, history, cfg, outdir, key)
    write_report(outdir, cfg, stats)
    print(f"[diag] Diagnostics saved to: {outdir}")


if __name__ == "__main__":
    main()

