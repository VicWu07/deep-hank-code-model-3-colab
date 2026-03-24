"""Autopilot runner: train -> diagnostics -> pass/fail -> optional remediation.

Usage:
  .venv/bin/python scripts/run_autopilot.py
  .venv/bin/python scripts/run_autopilot.py --quick --max-retries 1
"""

from __future__ import annotations

import argparse
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


def _quick_overrides(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["batch_size"] = 24
    cfg["phase0_steps"] = 20
    cfg["phase1_steps"] = 20
    cfg["phase2_steps"] = 20
    cfg["phase3_steps"] = 10
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
    cfg["diagnostic_every"] = 10
    cfg["checkpoint_every"] = 10
    cfg["active_candidate_size"] = 128
    cfg["active_sampling_every"] = 50
    cfg["active_topk_frac"] = 0.1
    cfg["active_mix_frac"] = 0.2
    return cfg


def _default_remediation(cfg: dict, manifest: dict) -> tuple[dict, list[str]]:
    out = dict(cfg)
    checks = manifest.get("checks", {})
    changes: list[str] = []

    if (not checks.get("loss_residual_mse", True)) or (not checks.get("loss_shape", True)):
        if not out.get("enable_phase1_curriculum", False):
            out["enable_phase1_curriculum"] = True
            changes.append("enable_phase1_curriculum=true")
        if not out.get("enable_phase2_range_curriculum", False):
            out["enable_phase2_range_curriculum"] = True
            changes.append("enable_phase2_range_curriculum=true")

    if not checks.get("residual_heatmap_max", True):
        if not out.get("enable_active_sampling", False):
            out["enable_active_sampling"] = True
            changes.append("enable_active_sampling=true")

    if (not checks.get("wage_clearing_p95", True)) or (not checks.get("wage_clearing_max", True)):
        out["w_bisect_steps"] = int(out.get("w_bisect_steps", 40)) + 8
        out["w_newton_steps"] = max(2, int(out.get("w_newton_steps", 2)))
        changes.append(f"w_bisect_steps={out['w_bisect_steps']}")
        changes.append(f"w_newton_steps={out['w_newton_steps']}")

    return out, changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep HANK autopilot run with machine-decidable remediation.")
    parser.add_argument("--quick", action="store_true", help="Use quick smoke-test configuration.")
    parser.add_argument("--max-retries", type=int, default=1, help="Retries after initial attempt when checks fail.")
    parser.add_argument("--output-root", default="autopilot_runs", help="Base output directory.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint in attempt directory if present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_outdir = Path(args.output_root) / run_id
    root_outdir.mkdir(parents=True, exist_ok=True)

    cfg = default_config()
    if args.quick:
        cfg = _quick_overrides(cfg)
    cfg["enable_step_logging"] = True
    cfg["log_every_steps"] = 20
    cfg["enable_debug_metrics"] = True
    cfg["debug_metrics_batch_cap"] = 8

    attempts = max(0, int(args.max_retries)) + 1
    summary = {"run_id": run_id, "attempts": []}
    final_pass = False

    for attempt in range(attempts):
        attempt_dir = root_outdir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        log_path = attempt_dir / "train_runtime.jsonl"
        ckpt_dir = attempt_dir / "checkpoints"
        print(f"[autopilot] attempt={attempt} outdir={attempt_dir}")

        with log_path.open("w", encoding="utf-8", buffering=1) as f:

            def log_fn(payload: dict) -> None:
                f.write(json.dumps(payload) + "\n")

            model, history = train(
                cfg,
                log_fn=log_fn,
                checkpoint_dir=ckpt_dir,
                resume=bool(args.resume),
            )

        key = jax.random.PRNGKey(int(cfg["seed"]) + 123 + attempt)
        stats = run_all_visual_diagnostics(model, history, cfg, attempt_dir, key)
        write_report(attempt_dir, cfg, stats)

        manifest_path = attempt_dir / "diagnostics.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        attempt_item = {
            "attempt": attempt,
            "pass": bool(manifest.get("pass", False)),
            "checks": manifest.get("checks", {}),
            "summary": manifest.get("summary", {}),
            "config_snapshot": {
                "enable_phase1_curriculum": bool(cfg.get("enable_phase1_curriculum", False)),
                "enable_phase2_range_curriculum": bool(cfg.get("enable_phase2_range_curriculum", False)),
                "enable_active_sampling": bool(cfg.get("enable_active_sampling", False)),
                "w_bisect_steps": int(cfg.get("w_bisect_steps", 40)),
                "w_newton_steps": int(cfg.get("w_newton_steps", 2)),
            },
        }
        summary["attempts"].append(attempt_item)
        (root_outdir / "autopilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if attempt_item["pass"]:
            final_pass = True
            print(f"[autopilot] PASS on attempt {attempt}")
            break

        if attempt == attempts - 1:
            print(f"[autopilot] FAIL after {attempts} attempts")
            break

        cfg, changes = _default_remediation(cfg, manifest)
        remediation_payload = {"attempt": attempt, "changes": changes}
        (attempt_dir / "remediation.json").write_text(json.dumps(remediation_payload, indent=2), encoding="utf-8")
        print(f"[autopilot] remediation changes: {changes}")

    status = {"run_id": run_id, "pass": final_pass, "attempts_used": len(summary["attempts"])}
    (root_outdir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"[autopilot] status: {status}")


if __name__ == "__main__":
    main()
