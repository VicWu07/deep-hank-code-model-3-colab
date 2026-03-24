from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ks_hank_hw.approximators import EMINN
from ks_hank_hw.config import RunConfig
from ks_hank_hw.residuals import HANKMRSOptimizedExactItoResidual
from ks_hank_hw.samplers import AdaptiveIntervalSampler


def _summary(values: np.ndarray):
    abs_v = np.abs(values)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mean_abs": float(np.mean(abs_v)),
        "p90_abs": float(np.percentile(abs_v, 90.0)),
        "p99_abs": float(np.percentile(abs_v, 99.0)),
        "max_abs": float(np.max(abs_v)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _quantiles(values: np.ndarray, qs=(1, 5, 50, 95, 99)):
    out = {}
    for q in qs:
        out[f"q{q}"] = float(np.percentile(values, q))
    return out


def _forward_no_grad_chunked(model, x_flat: torch.Tensor, agg_flat: torch.Tensor, chunk_size: int = 32768):
    outputs = []
    total = int(x_flat.shape[0])
    with torch.no_grad():
        for lo in range(0, total, chunk_size):
            hi = min(lo + chunk_size, total)
            outputs.append(model(x_flat[lo:hi], agg_flat[lo:hi])[:, 0])
    return torch.cat(outputs, dim=0)


def _compute_residuals_chunked(residual_op, model, x_np: np.ndarray, agg_np: np.ndarray, device, batch_size: int = 256):
    vals = []
    n = int(x_np.shape[0])
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        x_t = torch.tensor(x_np[lo:hi], dtype=torch.float32, device=device)
        agg_t = torch.tensor(agg_np[lo:hi], dtype=torch.float32, device=device)
        with torch.enable_grad():
            r = residual_op(model, x_t, agg_t)
        vals.append(r.detach().cpu().numpy().reshape(-1))
    return np.concatenate(vals, axis=0)


def _compute_outputs_chunked(model, x_np: np.ndarray, agg_np: np.ndarray, device, batch_size: int = 4096):
    vals = []
    n = int(x_np.shape[0])
    with torch.no_grad():
        for lo in range(0, n, batch_size):
            hi = min(lo + batch_size, n)
            x_t = torch.tensor(x_np[lo:hi], dtype=torch.float32, device=device)
            agg_t = torch.tensor(agg_np[lo:hi], dtype=torch.float32, device=device)
            vals.append(model(x_t, agg_t)[:, 0].detach().cpu().numpy())
    return np.concatenate(vals, axis=0)


def _compute_derivs_chunked(model, x_np: np.ndarray, agg_np: np.ndarray, device, batch_size: int = 128):
    dvda0 = []
    dvdz = []
    n = int(x_np.shape[0])
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        x_t = torch.tensor(x_np[lo:hi], dtype=torch.float32, device=device, requires_grad=True)
        agg_t = torch.tensor(agg_np[lo:hi], dtype=torch.float32, device=device, requires_grad=True)
        v = model(x_t, agg_t)[:, 0:1]
        dV_dx = torch.autograd.grad(v, x_t, grad_outputs=torch.ones_like(v), create_graph=False, retain_graph=True)[0]
        dV_dagg = torch.autograd.grad(v, agg_t, grad_outputs=torch.ones_like(v), create_graph=False)[0]
        dvda0.append(dV_dx[:, 0].detach().cpu().numpy())
        dvdz.append(dV_dagg[:, 0].detach().cpu().numpy())
    return np.concatenate(dvda0, axis=0), np.concatenate(dvdz, axis=0)


def _compute_drift_gap_batch(model, cfg: RunConfig, x_t: torch.Tensor, agg_t: torch.Tensor):
    """
    Policy-implied aggregate drift gap for each row:
    gap = mean_j s_j, where s_j is per-capita saving drift for agent j.
    """
    eco = HANKMRSOptimizedExactItoResidual(cfg.model).eco
    bsize = int(x_t.shape[0])
    n_pop = int(cfg.model.n_pop)
    if n_pop <= 1:
        raise NotImplementedError("Need n_pop > 1.")

    # Focal swaps (k into slot 0).
    x_all = x_t.unsqueeze(1).repeat(1, n_pop, 1)
    for k in range(n_pop):
        if k == 0:
            continue
        x_all[:, k, 0], x_all[:, k, k] = x_t[:, k].clone(), x_t[:, 0].clone()
        x_all[:, k, n_pop], x_all[:, k, n_pop + k] = x_t[:, n_pop + k].clone(), x_t[:, n_pop].clone()

    x_all_flat = x_all.view(bsize * n_pop, 2 * n_pop)
    agg_all_flat = agg_t.unsqueeze(1).repeat(1, n_pop, 1).view(bsize * n_pop, 2)
    v_all_flat = _forward_no_grad_chunked(model, x_all_flat, agg_all_flat)
    v_all = v_all_flat.view(bsize, n_pop)
    c_all = torch.clamp(v_all, min=cfg.model.c_floor) ** (-1.0 / cfg.model.gam)

    # Exact leave-one-out avg marginal utility.
    x_excl = x_all.unsqueeze(2).repeat(1, 1, n_pop - 1, 1)
    for m in range(1, n_pop):
        a0 = x_excl[:, :, m - 1, 0].clone()
        am = x_excl[:, :, m - 1, m].clone()
        x_excl[:, :, m - 1, 0] = am
        x_excl[:, :, m - 1, m] = a0

        y0 = x_excl[:, :, m - 1, n_pop].clone()
        ym = x_excl[:, :, m - 1, n_pop + m].clone()
        x_excl[:, :, m - 1, n_pop] = ym
        x_excl[:, :, m - 1, n_pop + m] = y0

    x_excl_flat = x_excl.reshape(bsize * n_pop * (n_pop - 1), 2 * n_pop)
    agg_excl_flat = agg_t.unsqueeze(1).unsqueeze(2).repeat(1, n_pop, n_pop - 1, 1).reshape(
        bsize * n_pop * (n_pop - 1), 2
    )
    v_excl_flat = _forward_no_grad_chunked(model, x_excl_flat, agg_excl_flat)
    c_excl = torch.clamp(v_excl_flat.view(bsize, n_pop, n_pop - 1), min=cfg.model.c_floor) ** (-1.0 / cfg.model.gam)
    avg_marg_u_excl = eco.du(c_excl).mean(dim=2)

    y_other = x_all[:, :, n_pop + 1 : 2 * n_pop]
    n_star_excl = eco.labor_from_indicator(y_other).mean(dim=2)
    w_all = eco.mrs_wage(n_star_excl, avg_marg_u_excl)
    r_all = eco.real_rate(agg_t[:, 1]).unsqueeze(1).repeat(1, n_pop)
    z_rep = agg_t[:, 0].unsqueeze(1).repeat(1, n_pop)
    pi_rep = agg_t[:, 1].unsqueeze(1).repeat(1, n_pop)
    pi_share_all = eco.transfer(z_rep, pi_rep, w_all, n_star_excl)

    a_all = x_t[:, :n_pop]
    y_all = x_t[:, n_pop:]
    n_all = eco.labor_from_indicator(y_all)
    s_all = r_all * a_all + w_all * n_all - c_all + pi_share_all
    gap = torch.mean(s_all, dim=1)
    return gap.detach().cpu().numpy().reshape(-1)


def _compute_drift_gap_chunked(model, cfg: RunConfig, x_np: np.ndarray, agg_np: np.ndarray, device, batch_size: int = 256):
    vals = []
    n = int(x_np.shape[0])
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        x_t = torch.tensor(x_np[lo:hi], dtype=torch.float32, device=device)
        agg_t = torch.tensor(agg_np[lo:hi], dtype=torch.float32, device=device)
        vals.append(_compute_drift_gap_batch(model, cfg, x_t, agg_t))
    return np.concatenate(vals, axis=0)


def _make_heatmap(values: np.ndarray, a: np.ndarray, z: np.ndarray, a_edges: np.ndarray, z_edges: np.ndarray):
    n_a = int(a_edges.size - 1)
    n_z = int(z_edges.size - 1)
    out = np.full((n_z, n_a), np.nan, dtype=np.float64)
    for iz in range(n_z):
        z_mask = (z >= z_edges[iz]) & (z < z_edges[iz + 1] if iz < n_z - 1 else z <= z_edges[iz + 1])
        for ia in range(n_a):
            a_mask = (a >= a_edges[ia]) & (a < a_edges[ia + 1] if ia < n_a - 1 else a <= a_edges[ia + 1])
            mask = z_mask & a_mask
            if np.any(mask):
                out[iz, ia] = float(np.mean(values[mask]))
    return out


def _permute_others_per_row(x_np: np.ndarray, n_pop: int, rng: np.random.Generator):
    x_sw = x_np.copy()
    for i in range(x_np.shape[0]):
        perm = rng.permutation(n_pop - 1) + 1
        x_sw[i, 1:n_pop] = x_np[i, perm]
        x_sw[i, n_pop + 1 : 2 * n_pop] = x_np[i, n_pop + perm]
    return x_sw


def main():
    cfg = RunConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(cfg.train.seed)

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "mrs_optimized_exact_outputs"
    diag_dir = output_dir / "diagnostics_first6"
    diag_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoints" / "final.pt"

    model = EMINN(
        input_dim=2 * cfg.model.n_pop + 2,
        width=cfg.net.width,
        layers=cfg.net.layers,
    ).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    residual_op = HANKMRSOptimizedExactItoResidual(cfg.model)
    sampler = AdaptiveIntervalSampler(cfg.model, cfg.sampling)

    # One holdout sample (no bootstrap economies), as requested.
    n_holdout = 4096
    x_np, agg_np = sampler.sample_main(n_holdout)
    y_i = np.round(x_np[:, cfg.model.n_pop]).astype(np.int64)
    a_i = x_np[:, 0]
    z_vals = agg_np[:, 0]

    # 1) Holdout residual summary (R at states).
    residuals = _compute_residuals_chunked(residual_op, model, x_np, agg_np, device, batch_size=256)
    residual_abs = np.abs(residuals)

    diag1 = {
        "all": _summary(residuals),
        "n_low": _summary(residuals[y_i == 0]) if np.any(y_i == 0) else None,
        "n_high": _summary(residuals[y_i == 1]) if np.any(y_i == 1) else None,
    }

    # 2) Residual heatmap by region (asset x z), split by own employment.
    a_edges = np.linspace(cfg.model.a_min, cfg.model.a_max, 21)
    z_edges = np.linspace(cfg.model.z_min, cfg.model.z_max, 21)
    heat_low = _make_heatmap(residual_abs[y_i == 0], a_i[y_i == 0], z_vals[y_i == 0], a_edges, z_edges)
    heat_high = _make_heatmap(residual_abs[y_i == 1], a_i[y_i == 1], z_vals[y_i == 1], a_edges, z_edges)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    vmin = np.nanmin([np.nanmin(heat_low), np.nanmin(heat_high)])
    vmax = np.nanmax([np.nanmax(heat_low), np.nanmax(heat_high)])
    im0 = axes[0].imshow(
        heat_low,
        origin="lower",
        aspect="auto",
        extent=[cfg.model.a_min, cfg.model.a_max, cfg.model.z_min, cfg.model.z_max],
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    axes[0].set_title("mean |R| heatmap (own n_low)")
    axes[0].set_xlabel("own a")
    axes[0].set_ylabel("z")
    axes[1].imshow(
        heat_high,
        origin="lower",
        aspect="auto",
        extent=[cfg.model.a_min, cfg.model.a_max, cfg.model.z_min, cfg.model.z_max],
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    axes[1].set_title("mean |R| heatmap (own n_high)")
    axes[1].set_xlabel("own a")
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label("mean |R|")
    fig.suptitle("Diagnostic 2: Residual Heatmaps by State Region")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(diag_dir / "diag2_residual_heatmap_absR.png", dpi=180)
    plt.close(fig)

    # 3) Aggregate drift-saving gap (policy-implied, not static state-clearing).
    drift_gap = _compute_drift_gap_chunked(model, cfg, x_np, agg_np, device, batch_size=256)
    diag3 = {
        "all": _summary(drift_gap),
        "n_low": _summary(drift_gap[y_i == 0]) if np.any(y_i == 0) else None,
        "n_high": _summary(drift_gap[y_i == 1]) if np.any(y_i == 1) else None,
    }

    # 4) Policy admissibility checks.
    w_pred = _compute_outputs_chunked(model, x_np, agg_np, device, batch_size=4096)
    c_pred = np.maximum(w_pred, cfg.model.c_floor) ** (-1.0 / cfg.model.gam)
    diag4 = {
        "share_w_nonpositive": float(np.mean(w_pred <= 0.0)),
        "share_w_below_floor": float(np.mean(w_pred <= cfg.model.c_floor)),
        "share_w_finite": float(np.mean(np.isfinite(w_pred))),
        "share_c_finite": float(np.mean(np.isfinite(c_pred))),
        "w_quantiles": _quantiles(w_pred),
        "c_quantiles": _quantiles(c_pred),
    }

    # 5) Shape diagnostics on a subset.
    n_shape = 2048
    idx_shape = rng.choice(np.arange(n_holdout), size=n_shape, replace=False)
    dvda0, dvdz = _compute_derivs_chunked(model, x_np[idx_shape], agg_np[idx_shape], device, batch_size=128)
    diag5 = {
        "mean_dvda0": float(np.mean(dvda0)),
        "mean_dvdz": float(np.mean(dvdz)),
        "share_dvda0_positive": float(np.mean(dvda0 > 0.0)),
        "share_dvdz_positive": float(np.mean(dvdz > 0.0)),
        "dvda0_quantiles": _quantiles(dvda0),
        "dvdz_quantiles": _quantiles(dvdz),
    }

    # 6) Symmetry/permutation consistency (permute only "other agents").
    n_perm = 2048
    idx_perm = rng.choice(np.arange(n_holdout), size=n_perm, replace=False)
    x_base = x_np[idx_perm]
    agg_base = agg_np[idx_perm]
    x_perm = _permute_others_per_row(x_base, cfg.model.n_pop, rng=rng)
    w_base = _compute_outputs_chunked(model, x_base, agg_base, device, batch_size=4096)
    w_perm = _compute_outputs_chunked(model, x_perm, agg_base, device, batch_size=4096)
    perm_diff = w_perm - w_base
    diag6 = {
        "delta_w_summary": _summary(perm_diff),
        "delta_w_abs_quantiles": _quantiles(np.abs(perm_diff)),
    }

    # Compact table for the six diagnostics.
    table_rows = [
        {"diagnostic": "1_residual_holdout", "mean_abs": diag1["all"]["mean_abs"], "p99_abs": diag1["all"]["p99_abs"]},
        {"diagnostic": "3_drift_gap", "mean_abs": diag3["all"]["mean_abs"], "p99_abs": diag3["all"]["p99_abs"]},
        {"diagnostic": "4_admissibility_w<=0_share", "mean_abs": diag4["share_w_nonpositive"], "p99_abs": np.nan},
        {"diagnostic": "4_admissibility_w<=floor_share", "mean_abs": diag4["share_w_below_floor"], "p99_abs": np.nan},
        {"diagnostic": "5_shape_dvda_pos_share", "mean_abs": diag5["share_dvda0_positive"], "p99_abs": np.nan},
        {"diagnostic": "5_shape_dvdz_pos_share", "mean_abs": diag5["share_dvdz_positive"], "p99_abs": np.nan},
        {"diagnostic": "6_perm_delta_w_mean_abs", "mean_abs": diag6["delta_w_summary"]["mean_abs"], "p99_abs": diag6["delta_w_summary"]["p99_abs"]},
    ]

    payload_out = {
        "checkpoint": str(ckpt_path),
        "sample": {"n_holdout": int(n_holdout), "n_shape": int(n_shape), "n_perm": int(n_perm)},
        "diag1_residual_holdout": diag1,
        "diag2_residual_heatmap": {
            "a_edges": a_edges.tolist(),
            "z_edges": z_edges.tolist(),
            "heat_low": heat_low.tolist(),
            "heat_high": heat_high.tolist(),
            "plot": str(diag_dir / "diag2_residual_heatmap_absR.png"),
        },
        "diag3_aggregate_drift_gap": diag3,
        "diag4_policy_admissibility": diag4,
        "diag5_shape": diag5,
        "diag6_permutation_consistency": diag6,
    }

    with (diag_dir / "diagnostics_first6.json").open("w", encoding="utf-8") as f:
        json.dump(payload_out, f, indent=2)

    with (diag_dir / "diagnostics_first6_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["diagnostic", "mean_abs", "p99_abs"])
        writer.writeheader()
        writer.writerows(table_rows)

    print("Saved first-6 diagnostics to:", diag_dir)
    print("JSON:", diag_dir / "diagnostics_first6.json")
    print("Table:", diag_dir / "diagnostics_first6_table.csv")
    print("Heatmap:", diag_dir / "diag2_residual_heatmap_absR.png")


if __name__ == "__main__":
    main()
