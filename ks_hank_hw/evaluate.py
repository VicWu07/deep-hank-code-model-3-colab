from __future__ import annotations

import numpy as np
import torch

from .config import ModelConfig


def _default_device(device=None):
    return device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))


def _normalized_probs(weights: np.ndarray):
    probs = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    total = float(np.sum(probs))
    if total <= 0.0:
        return np.ones_like(probs) / max(probs.size, 1)
    return probs / total


def _draw_fd_stationary_panel(fd_state: dict, n_pop: int, seed: int | None = None, rng=None):
    rng = rng or np.random.default_rng(seed)
    a_grid = np.asarray(fd_state["a_grid"], dtype=np.float64)
    g = np.asarray(fd_state["g"], dtype=np.float64)
    g = np.maximum(g, 0.0)
    if g.ndim != 2 or g.shape[0] != 2:
        raise ValueError("Expected fd_state['g'] with shape (2, n_assets).")

    n_other = max(n_pop - 1, 0)
    n_low = n_other // 2
    n_high = n_other - n_low
    low_probs = _normalized_probs(g[0])
    high_probs = _normalized_probs(g[1])

    low_idx = rng.choice(a_grid.size, size=n_low, replace=True, p=low_probs) if n_low > 0 else np.zeros(0, dtype=int)
    high_idx = (
        rng.choice(a_grid.size, size=n_high, replace=True, p=high_probs) if n_high > 0 else np.zeros(0, dtype=int)
    )
    a_low = a_grid[low_idx]
    a_high = a_grid[high_idx]
    a_other = np.concatenate([a_low, a_high]).astype(np.float64)
    if a_other.size > 0:
        a_other = a_other - np.mean(a_other)
    y_other = np.concatenate([np.zeros(n_low, dtype=np.float32), np.ones(n_high, dtype=np.float32)])
    return {
        "a_other": a_other.astype(np.float32),
        "y_other": y_other,
        "n_low": int(n_low),
        "n_high": int(n_high),
    }


def build_fd_slice(
    a_grid: np.ndarray,
    z: float,
    pi: float,
    n_pop: int,
    own_employment: float,
    panel: dict,
):
    n = len(a_grid)
    x = np.zeros((n, 2 * n_pop), dtype=np.float32)
    x[:, 0] = a_grid.astype(np.float32)
    if n_pop > 1:
        x[:, 1:n_pop] = panel["a_other"][None, :]
        x[:, n_pop + 1 : 2 * n_pop] = panel["y_other"][None, :]
    x[:, n_pop] = float(own_employment)
    agg = np.zeros((n, 2), dtype=np.float32)
    agg[:, 0] = float(z)
    agg[:, 1] = float(pi)
    return x, agg


def predict_dv_and_c(model, x: np.ndarray, agg: np.ndarray, gam: float, device=None):
    device = _default_device(device)
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    at = torch.tensor(agg, dtype=torch.float32, device=device)
    with torch.no_grad():
        dv = model(xt, at).detach().cpu().numpy().reshape(-1)
    c = np.maximum(dv, 1e-8) ** (-1.0 / gam)
    return dv, c


def error_stats(pred: np.ndarray, ref: np.ndarray):
    err = pred - ref
    abs_err = np.abs(err)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "mae": float(np.mean(abs_err)),
        "max_abs": float(np.max(abs_err)),
        "p90_abs": float(np.percentile(abs_err, 90.0)),
    }


def compare_fd_steady_anchor(
    model,
    fd_anchor: dict,
    cfg: ModelConfig,
    device=None,
    seed: int = 777,
):
    a_grid = np.asarray(fd_anchor["a_grid"], dtype=np.float64)
    fd_c = np.asarray(fd_anchor["c_fd"], dtype=np.float64)
    fd_dv = np.asarray(fd_anchor["dv_fd"], dtype=np.float64)
    panel = _draw_fd_stationary_panel(fd_anchor, n_pop=cfg.n_pop, seed=seed)

    nn_c = np.zeros_like(fd_c)
    nn_dv = np.zeros_like(fd_dv)
    stats = {}

    for y_state in (0, 1):
        x, agg = build_fd_slice(
            a_grid=a_grid,
            z=float(fd_anchor["z"]),
            pi=float(fd_anchor["pi"]),
            n_pop=cfg.n_pop,
            own_employment=float(y_state),
            panel=panel,
        )
        dv_pred, c_pred = predict_dv_and_c(model, x, agg, gam=cfg.gam, device=device)
        nn_c[y_state, :] = c_pred
        nn_dv[y_state, :] = dv_pred
        stats[f"c_state{y_state}"] = error_stats(c_pred, fd_c[y_state, :])
        stats[f"dv_state{y_state}"] = error_stats(dv_pred, fd_dv[y_state, :])

    return {
        "a_grid": a_grid,
        "fd_c": fd_c,
        "fd_dv": fd_dv,
        "nn_c": nn_c,
        "nn_dv": nn_dv,
        "error_stats": stats,
        "panel": {
            "a_other": panel["a_other"].tolist(),
            "y_other": panel["y_other"].tolist(),
            "n_low": panel["n_low"],
            "n_high": panel["n_high"],
        },
    }


def compare_fd_z_slices(
    model,
    fd_slices: dict,
    cfg: ModelConfig,
    device=None,
    seed: int = 777,
):
    rng = np.random.default_rng(seed)
    panel_ref = _draw_fd_stationary_panel(fd_slices["z_low"], n_pop=cfg.n_pop, rng=rng)
    out = {}

    for key in ("z_low", "z_high"):
        state = fd_slices[key]
        a_grid = np.asarray(state["a_grid"], dtype=np.float64)
        fd_c = np.asarray(state["c_fd"], dtype=np.float64)
        fd_dv = np.asarray(state["dv_fd"], dtype=np.float64)
        nn_c = np.zeros_like(fd_c)
        nn_dv = np.zeros_like(fd_dv)
        stats = {}
        for y_state in (0, 1):
            x, agg = build_fd_slice(
                a_grid=a_grid,
                z=float(state["z"]),
                pi=float(state["pi"]),
                n_pop=cfg.n_pop,
                own_employment=float(y_state),
                panel=panel_ref,
            )
            dv_pred, c_pred = predict_dv_and_c(model, x, agg, gam=cfg.gam, device=device)
            nn_c[y_state, :] = c_pred
            nn_dv[y_state, :] = dv_pred
            stats[f"c_state{y_state}"] = error_stats(c_pred, fd_c[y_state, :])
            stats[f"dv_state{y_state}"] = error_stats(dv_pred, fd_dv[y_state, :])

        out[key] = {
            "a_grid": a_grid,
            "fd_c": fd_c,
            "fd_dv": fd_dv,
            "nn_c": nn_c,
            "nn_dv": nn_dv,
            "error_stats": stats,
            "z": float(state["z"]),
            "pi": float(state["pi"]),
        }
    return out
