"""Visual runtime diagnostics for Deep HANK (quick mode)."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .economics import excess_savings, solve_w
from .model import w_forward
from .residual import pde_residual_single
from .sampling import sample_batch
from .simulation import simulate_path


def _to_np(x):
    return np.asarray(x)


def _to_float(x) -> float:
    return float(np.asarray(x))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _diff_summary(y: np.ndarray, x: np.ndarray, tol: float = 1.0e-10) -> dict:
    dx = np.diff(x)
    dy = np.diff(y)
    slope = np.divide(dy, np.maximum(dx, tol))
    return {
        "min_slope": float(slope.min()) if slope.size else float("nan"),
        "monotonicity_violations": int(np.sum(slope < -tol)),
    }


def _bool(x: bool) -> bool:
    return bool(x)


def plot_loss_curves(history: list[dict], outdir: Path) -> dict:
    steps = np.array([h["step"] for h in history], dtype=float)
    warm = np.array([h.get("warmstart_mse", np.nan) for h in history], dtype=float)
    rmse = np.array([h.get("residual_mse", np.nan) for h in history], dtype=float)
    shp = np.array([h.get("shape_loss", np.nan) for h in history], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(steps, warm, label="warmstart_mse")
    ax.plot(steps, rmse, label="residual_mse")
    ax.plot(steps, shp, label="shape_loss")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Metric (log)")
    ax.set_title("Training curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_curves.png", dpi=150)
    plt.close(fig)
    payload = {
        "steps": steps.tolist(),
        "warmstart_mse": warm.tolist(),
        "residual_mse": rmse.tolist(),
        "shape_loss": shp.tolist(),
        "final": {
            "warmstart_mse": float(warm[-1]) if warm.size else float("nan"),
            "residual_mse": float(rmse[-1]) if rmse.size else float("nan"),
            "shape_loss": float(shp[-1]) if shp.size else float("nan"),
        },
    }
    _write_json(outdir / "loss_curves.json", payload)
    return payload


def _representative_phi(key: jax.Array, cfg: dict):
    cfg_eval = dict(cfg)
    cfg_eval["batch_size"] = 1
    batch = sample_batch(key, cfg_eval, phase=3, boundary_frac=cfg_eval.get("boundary_frac_phase3", 0.3))
    return batch["a_all"][0], batch["n_all"][0]


def residual_heatmaps(model, cfg: dict, outdir: Path, key: jax.Array) -> dict:
    a_pts = int(cfg.get("diag_heatmap_a_points", 24))
    z_pts = int(cfg.get("diag_heatmap_z_points", 17))
    a_grid = jnp.linspace(cfg["a_bar"], cfg["a_max"], a_pts)
    z_grid = jnp.linspace(cfg["z_min"], cfg["z_max"], z_pts)
    pi0 = jnp.array(0.0, dtype=jnp.float32)
    a_full, n_full = _representative_phi(key, cfg)
    a_others = a_full[1:]
    n_others = n_full[1:]

    # Wage solved per z for representative cross-section.
    w_by_z = jnp.stack([solve_w(model, z, pi0, a_full, n_full, cfg) for z in z_grid])
    a_grid_np = _to_np(a_grid)
    z_grid_np = _to_np(z_grid)
    heatmaps = {}
    summary = {}

    for state_name, n_val, fname in [
        ("n_low", cfg["n_low"], "residual_heatmap_nlow.png"),
        ("n_high", cfg["n_high"], "residual_heatmap_nhigh.png"),
    ]:
        heat = np.zeros((len(z_grid), len(a_grid)))
        for iz, z in enumerate(z_grid):
            w = w_by_z[iz]
            for ia, a in enumerate(a_grid):
                r = pde_residual_single(
                    model,
                    a_i=a,
                    n_i=jnp.array(n_val, dtype=jnp.float32),
                    z=z,
                    pi=pi0,
                    w=w,
                    a_others=a_others,
                    n_others=n_others,
                    cfg=cfg,
                    include_aggregate=True,
                    include_distribution=bool(cfg.get("diag_heatmap_include_distribution", False)),
                )
                heat[iz, ia] = float(abs(r))
        log10_heat = np.log10(np.maximum(heat, 1e-12))
        heatmaps[state_name] = {
            "log10_abs_R": log10_heat.tolist(),
        }
        summary[f"{state_name}_max_log10_abs_R"] = float(np.max(log10_heat))
        summary[f"{state_name}_mean_log10_abs_R"] = float(np.mean(log10_heat))
        summary[f"{state_name}_p95_log10_abs_R"] = float(np.quantile(log10_heat, 0.95))

        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        im = ax.imshow(
            log10_heat,
            origin="lower",
            aspect="auto",
            extent=[float(a_grid[0]), float(a_grid[-1]), float(z_grid[0]), float(z_grid[-1])],
        )
        ax.set_xlabel("a")
        ax.set_ylabel("z")
        ax.set_title(f"log10 |Residual|, n={n_val:.2f}")
        fig.colorbar(im, ax=ax, label="log10 |R|")
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=150)
        plt.close(fig)
    payload = {
        "a_grid": a_grid_np.tolist(),
        "z_grid": z_grid_np.tolist(),
        "n_low": heatmaps["n_low"],
        "n_high": heatmaps["n_high"],
        "summary": summary,
    }
    _write_json(outdir / "residual_heatmaps.json", payload)
    return payload


def wage_clearing_diagnostics(model, cfg: dict, outdir: Path, key: jax.Array) -> dict:
    cfg_eval = dict(cfg)
    cfg_eval["batch_size"] = int(cfg.get("diag_wage_samples", 48))
    batch = sample_batch(key, cfg_eval, phase=3, boundary_frac=cfg_eval.get("boundary_frac_phase3", 0.3))
    errs = []
    for i in range(cfg_eval["batch_size"]):
        z = batch["z"][i]
        pi = batch["pi"][i]
        a_all = batch["a_all"][i]
        n_all = batch["n_all"][i]
        w = solve_w(model, z, pi, a_all, n_all, cfg)
        f = excess_savings(model, w, z, pi, a_all, n_all, cfg)
        errs.append(float(abs(f)))
    errs = np.asarray(errs)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.hist(errs, bins=30)
    ax.set_xlabel("|F(w)|")
    ax.set_ylabel("Count")
    ax.set_title("Wage-clearing error distribution")
    fig.tight_layout()
    fig.savefig(outdir / "w_clearing.png", dpi=150)
    plt.close(fig)
    payload = {
        "n_samples": int(cfg_eval["batch_size"]),
        "abs_errors": errs.tolist(),
        "mean_abs": float(errs.mean()),
        "p95_abs": float(np.quantile(errs, 0.95)),
        "max_abs": float(errs.max()),
    }
    _write_json(outdir / "wage_clearing.json", payload)
    return {
        "w_clear_mean_abs": float(errs.mean()),
        "w_clear_p95_abs": float(np.quantile(errs, 0.95)),
        "w_clear_max_abs": float(errs.max()),
        "wage_clearing_payload": payload,
    }


def policy_slices(model, cfg: dict, outdir: Path, key: jax.Array) -> dict:
    a_grid = jnp.linspace(cfg["a_bar"], cfg["a_max"], 200)
    pi0 = jnp.array(0.0, dtype=jnp.float32)
    z_vals = [cfg["z_bar"], cfg["z_min"], cfg["z_max"]]
    labels = ["z_bar", "z_min", "z_max"]
    n_specs = [(cfg["n_low"], "n_low", "-"), (cfg["n_high"], "n_high", "--")]

    a_full, n_full = _representative_phi(key, cfg)
    a_others = a_full[1:]
    n_others = n_full[1:]
    a_grid_np = _to_np(a_grid)
    W_data: dict[str, dict[str, list[float]]] = {lab: {} for lab in labels}
    c_data: dict[str, dict[str, list[float]]] = {lab: {} for lab in labels}
    W_min_slope = []
    c_min_slope = []
    W_violations = 0
    c_violations = 0

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for z, zlab in zip(z_vals, labels):
        w = solve_w(model, jnp.array(z), pi0, a_full, n_full, cfg)
        for n_val, nlab, ls in n_specs:
            W = np.array(
                [
                    float(
                        w_forward(
                            model,
                            jnp.array(a, dtype=jnp.float32),
                            jnp.array(n_val, dtype=jnp.float32),
                            jnp.array(z, dtype=jnp.float32),
                            pi0,
                            w,
                            a_others,
                            n_others,
                            cfg,
                        )
                    )
                    for a in a_grid
                ]
            )
            c = np.power(np.maximum(W, 1e-12), -1.0 / cfg["gamma"])
            axes[0].plot(a_grid_np, W, ls=ls, label=f"W {nlab}, {zlab}")
            axes[1].plot(a_grid_np, c, ls=ls, label=f"c {nlab}, {zlab}")
            W_data[zlab][nlab] = W.tolist()
            c_data[zlab][nlab] = c.tolist()
            w_stats = _diff_summary(W, a_grid_np)
            c_stats = _diff_summary(c, a_grid_np)
            W_min_slope.append(w_stats["min_slope"])
            c_min_slope.append(c_stats["min_slope"])
            W_violations += w_stats["monotonicity_violations"]
            c_violations += c_stats["monotonicity_violations"]

    axes[0].set_ylabel("W(a)")
    axes[1].set_ylabel("c(a)")
    axes[1].set_xlabel("a")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "policy_slices.png", dpi=150)
    plt.close(fig)
    payload = {
        "a_grid": a_grid_np.tolist(),
        "W": W_data,
        "c": c_data,
        "summary": {
            "min_dW_da_over_all_slices": float(np.min(W_min_slope)) if W_min_slope else float("nan"),
            "min_dc_da_over_all_slices": float(np.min(c_min_slope)) if c_min_slope else float("nan"),
            "monotonicity_violations_W": int(W_violations),
            "monotonicity_violations_c": int(c_violations),
        },
    }
    _write_json(outdir / "policy_slices.json", payload)
    return payload


def sim_paths_and_hists(model, cfg: dict, outdir: Path, key: jax.Array) -> dict:
    path = simulate_path(key, model, cfg, steps=int(cfg.get("diag_sim_steps", 80)), dt=cfg.get("dt_sim", 0.05))
    z = _to_np(path["z"])
    pi = _to_np(path["pi"])
    w = _to_np(path["w"])
    a_all = _to_np(path["a_all"])
    n_all = _to_np(path["n_all"])

    mean_a = a_all.mean(axis=1)
    std_a = a_all.std(axis=1)
    share_high = (np.isclose(n_all, cfg["n_high"], atol=1e-6)).mean(axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(z, label="z")
    axes[0].plot(pi, label="pi")
    axes[0].plot(np.arange(1, len(w) + 1), w, label="w")
    axes[0].legend()
    axes[0].set_title("Aggregate paths")

    axes[1].plot(mean_a, label="mean(a)")
    axes[1].plot(std_a, label="std(a)")
    axes[1].legend()
    axes[1].set_title("Asset moments")

    axes[2].plot(share_high, label="share n_high")
    axes[2].legend()
    axes[2].set_title("Employment share")
    axes[2].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(outdir / "sim_paths.png", dpi=150)
    plt.close(fig)

    a_last = a_all[-1]
    n_last = n_all[-1]
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.hist(a_last, bins=30, alpha=0.5, label="all")
    ax.hist(a_last[np.isclose(n_last, cfg["n_low"], atol=1e-6)], bins=20, alpha=0.6, label="n_low")
    ax.hist(a_last[np.isclose(n_last, cfg["n_high"], atol=1e-6)], bins=20, alpha=0.6, label="n_high")
    ax.set_title("Assets at final simulated period")
    ax.set_xlabel("a")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "asset_hist.png", dpi=150)
    plt.close(fig)

    has_nan_or_inf = _bool(
        (not np.isfinite(z).all())
        or (not np.isfinite(pi).all())
        or (not np.isfinite(w).all())
        or (not np.isfinite(a_all).all())
        or (not np.isfinite(n_all).all())
    )
    share_range_ok = _bool(np.all((share_high >= 0.0) & (share_high <= 1.0)))
    tgrid = np.arange(z.shape[0], dtype=float) * float(cfg.get("dt_sim", 0.05))
    payload = {
        "steps": int(cfg.get("diag_sim_steps", 80)),
        "time": tgrid.tolist(),
        "z": z.tolist(),
        "pi": pi.tolist(),
        "w": w.tolist(),
        "mean_a": mean_a.tolist(),
        "std_a": std_a.tolist(),
        "share_high": share_high.tolist(),
        "final": {
            "mean_a_abs": float(abs(mean_a[-1])),
            "std_a": float(std_a[-1]),
            "share_high": float(share_high[-1]),
        },
        "summary": {
            "has_nan_or_inf": has_nan_or_inf,
            "share_high_in_valid_range": share_range_ok,
            "max_abs_mean_a": float(np.max(np.abs(mean_a))),
        },
    }
    _write_json(outdir / "sim_paths.json", payload)

    return {
        "sim_mean_a_abs_final": float(abs(mean_a[-1])),
        "sim_std_a_final": float(std_a[-1]),
        "sim_share_high_final": float(share_high[-1]),
        "sim_payload": payload,
    }


def _thresholds(cfg: dict) -> dict:
    return {
        "residual_mse_max": float(cfg.get("diag_pass_residual_mse_max", 1.0e-3)),
        "shape_loss_max": float(cfg.get("diag_pass_shape_loss_max", 1.0e-2)),
        "residual_log10_abs_R_max": float(cfg.get("diag_pass_residual_log10_abs_R_max", -1.5)),
        "w_clear_p95_abs_max": float(cfg.get("diag_pass_w_clear_p95_abs_max", 1.0e-4)),
        "w_clear_max_abs_max": float(cfg.get("diag_pass_w_clear_max_abs_max", 5.0e-4)),
        "policy_max_monotonicity_violations": int(cfg.get("diag_pass_policy_max_monotonicity_violations", 0)),
        "sim_max_abs_mean_a": float(cfg.get("diag_pass_sim_max_abs_mean_a", 0.5)),
        "sim_share_high_min": float(cfg.get("diag_pass_sim_share_high_min", 0.01)),
        "sim_share_high_max": float(cfg.get("diag_pass_sim_share_high_max", 0.99)),
    }


def _evaluate_checks(payload: dict, thresholds: dict) -> dict:
    loss_final = payload["loss_curves"]["final"]
    heat_summary = payload["residual_heatmaps"]["summary"]
    w_summary = payload["wage_clearing"]
    p_summary = payload["policy_slices"]["summary"]
    s_final = payload["sim_paths"]["final"]
    s_summary = payload["sim_paths"]["summary"]
    checks = {
        "loss_residual_mse": _bool(loss_final["residual_mse"] <= thresholds["residual_mse_max"]),
        "loss_shape": _bool(loss_final["shape_loss"] <= thresholds["shape_loss_max"]),
        "residual_heatmap_max": _bool(
            max(heat_summary["n_low_max_log10_abs_R"], heat_summary["n_high_max_log10_abs_R"])
            <= thresholds["residual_log10_abs_R_max"]
        ),
        "wage_clearing_p95": _bool(w_summary["p95_abs"] <= thresholds["w_clear_p95_abs_max"]),
        "wage_clearing_max": _bool(w_summary["max_abs"] <= thresholds["w_clear_max_abs_max"]),
        "policy_monotonic_W": _bool(
            p_summary["monotonicity_violations_W"] <= thresholds["policy_max_monotonicity_violations"]
        ),
        "policy_monotonic_c": _bool(
            p_summary["monotonicity_violations_c"] <= thresholds["policy_max_monotonicity_violations"]
        ),
        "sim_no_nan_or_inf": _bool(not s_summary["has_nan_or_inf"]),
        "sim_share_high_bounds": _bool(
            thresholds["sim_share_high_min"] <= s_final["share_high"] <= thresholds["sim_share_high_max"]
        ),
        "sim_max_abs_mean_a": _bool(s_summary["max_abs_mean_a"] <= thresholds["sim_max_abs_mean_a"]),
    }
    return checks


def write_interpretation_guide(outdir: Path) -> None:
    lines = [
        "# Diagnostics Interpretation Guide",
        "",
        "## loss_curves.png",
        "- Good: residual_mse and shape_loss trend down over training steps.",
        "- Bad: flat or rising curves indicate weak progress or instability.",
        "- Implication: fix training quality first before trusting downstream diagnostics.",
        "",
        "## residual_heatmap_nlow.png / residual_heatmap_nhigh.png",
        "- Good: most cells are cool with low log10(|R|).",
        "- Bad: hot regions indicate local HJB residual violations in (a, z).",
        "- Implication: target those regions with more training or different sampling.",
        "",
        "## w_clearing.png",
        "- Good: histogram mass concentrated near zero.",
        "- Bad: long right tail means wage-clearing errors in sampled states.",
        "- Implication: equilibrium price map may be inconsistent in parts of state space.",
        "",
        "## policy_slices.png",
        "- Good: W(a) and c(a) are smooth and generally increasing in a.",
        "- Bad: non-monotone kinks or decreasing segments signal shape issues.",
        "- Implication: revisit shape regularization and optimization settings.",
        "",
        "## sim_paths.png / asset_hist.png",
        "- Good: bounded aggregate paths, sensible moments, and plausible final asset distribution.",
        "- Bad: exploding series, drift, or boundary-mass pileups imply dynamic instability.",
        "- Implication: simulation diagnostics reflect whether learned policy generalizes in time.",
        "",
        "Machine readers should use JSON artifacts and diagnostics.json only; do not parse image files.",
        "",
        "<!-- amp-managed -->",
    ]
    (outdir / "INTERPRETATION.md").write_text("\n".join(lines), encoding="utf-8")


def write_diagnostics_manifest(outdir: Path, cfg: dict, payload: dict) -> dict:
    thresholds = _thresholds(cfg)
    checks = _evaluate_checks(payload, thresholds)
    summary = {
        "residual_mse_final": _to_float(payload["loss_curves"]["final"]["residual_mse"]),
        "shape_loss_final": _to_float(payload["loss_curves"]["final"]["shape_loss"]),
        "residual_heatmap_max_log10_abs_R": float(
            max(
                payload["residual_heatmaps"]["summary"]["n_low_max_log10_abs_R"],
                payload["residual_heatmaps"]["summary"]["n_high_max_log10_abs_R"],
            )
        ),
        "w_clear_p95_abs": _to_float(payload["wage_clearing"]["p95_abs"]),
        "w_clear_max_abs": _to_float(payload["wage_clearing"]["max_abs"]),
        "policy_slices_monotonic": _bool(
            payload["policy_slices"]["summary"]["monotonicity_violations_W"] == 0
            and payload["policy_slices"]["summary"]["monotonicity_violations_c"] == 0
        ),
        "sim_has_nan_or_inf": _bool(payload["sim_paths"]["summary"]["has_nan_or_inf"]),
    }
    manifest = {
        "run_id": outdir.name,
        "config_snapshot": {
            "batch_size": int(cfg["batch_size"]),
            "phase_steps": [
                int(cfg["phase0_steps"]),
                int(cfg["phase1_steps"]),
                int(cfg["phase2_steps"]),
                int(cfg["phase3_steps"]),
            ],
            "dt_sim": float(cfg["dt_sim"]),
            "diag_heatmap_a_points": int(cfg.get("diag_heatmap_a_points", 24)),
            "diag_heatmap_z_points": int(cfg.get("diag_heatmap_z_points", 17)),
            "diag_wage_samples": int(cfg.get("diag_wage_samples", 48)),
            "diag_sim_steps": int(cfg.get("diag_sim_steps", 80)),
        },
        "artifacts": {
            "loss_curves": "loss_curves.json",
            "residual_heatmaps": "residual_heatmaps.json",
            "wage_clearing": "wage_clearing.json",
            "policy_slices": "policy_slices.json",
            "sim_paths": "sim_paths.json",
        },
        "thresholds": thresholds,
        "checks": checks,
        "summary": summary,
        "pass": _bool(all(checks.values())),
    }
    _write_json(outdir / "diagnostics.json", manifest)
    return manifest


def run_all_visual_diagnostics(model, history: list[dict], cfg: dict, outdir: Path, key: jax.Array) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    stats = {}
    loss_payload = plot_loss_curves(history, outdir)

    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    heatmap_payload = residual_heatmaps(model, cfg, outdir, k1)
    wage_stats = wage_clearing_diagnostics(model, cfg, outdir, k2)
    policy_payload = policy_slices(model, cfg, outdir, k3)
    sim_stats = sim_paths_and_hists(model, cfg, outdir, k4)

    dual_payload = {
        "loss_curves": loss_payload,
        "residual_heatmaps": heatmap_payload,
        "wage_clearing": wage_stats["wage_clearing_payload"],
        "policy_slices": policy_payload,
        "sim_paths": sim_stats["sim_payload"],
    }
    manifest = write_diagnostics_manifest(outdir, cfg, dual_payload)
    write_interpretation_guide(outdir)

    stats.update(
        {
            "loss_residual_mse_final": _to_float(loss_payload["final"]["residual_mse"]),
            "loss_shape_loss_final": _to_float(loss_payload["final"]["shape_loss"]),
            "residual_heatmap_nlow_max_log10_abs_R": _to_float(heatmap_payload["summary"]["n_low_max_log10_abs_R"]),
            "residual_heatmap_nhigh_max_log10_abs_R": _to_float(heatmap_payload["summary"]["n_high_max_log10_abs_R"]),
            "policy_min_dW_da": _to_float(policy_payload["summary"]["min_dW_da_over_all_slices"]),
            "policy_min_dc_da": _to_float(policy_payload["summary"]["min_dc_da_over_all_slices"]),
            "policy_monotonicity_violations_W": _to_float(policy_payload["summary"]["monotonicity_violations_W"]),
            "policy_monotonicity_violations_c": _to_float(policy_payload["summary"]["monotonicity_violations_c"]),
            "diagnostics_pass": 1.0 if manifest["pass"] else 0.0,
        }
    )
    stats.update({k: v for k, v in wage_stats.items() if k != "wage_clearing_payload"})
    stats.update({k: v for k, v in sim_stats.items() if k != "sim_payload"})
    return stats


def write_report(outdir: Path, cfg: dict, stats: dict) -> None:
    lines = [
        "# Deep HANK Quick Diagnostics Report",
        "",
        "## Runtime config",
        f"- batch_size: {cfg['batch_size']}",
        f"- phase steps: ({cfg['phase0_steps']}, {cfg['phase1_steps']}, {cfg['phase2_steps']}, {cfg['phase3_steps']})",
        f"- dt_sim: {cfg['dt_sim']}",
        "",
        "## Scalar checks",
    ]
    for k, v in stats.items():
        lines.append(f"- {k}: {v:.6e}")
    lines.append("")
    lines.append("<!-- amp-managed -->")
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")

