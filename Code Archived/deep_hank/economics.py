"""Economic primitives and equilibrium price maps."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from .model import w_forward


def compute_r(pi: jnp.ndarray, cfg: dict) -> jnp.ndarray:
    return cfg["r_star"] + (cfg["phi_pi"] - 1.0) * pi


def compute_N_star(n_all: jnp.ndarray, _: int | None = None) -> jnp.ndarray:
    return jnp.mean(n_all)


def compute_mu_z(z: jnp.ndarray, cfg: dict) -> jnp.ndarray:
    return cfg["eta_z"] * (cfg["z_bar"] - z)


def compute_Pi_transfer(w: jnp.ndarray, z: jnp.ndarray, pi: jnp.ndarray, N_star: jnp.ndarray, cfg: dict) -> jnp.ndarray:
    # Keep this aligned with the writeup's transfer convention.
    return (1.0 - w / jnp.exp(z) + 0.5 * cfg["psi"] * pi**2) * jnp.exp(z) * N_star


def mu_pi(z: jnp.ndarray, pi: jnp.ndarray, w: jnp.ndarray, cfg: dict) -> jnp.ndarray:
    mu_z = compute_mu_z(z, cfg)
    return (
        cfg["r_star"] * pi
        + (cfg["phi_pi"] - 1.0) * pi**2
        + (cfg["epsilon"] / cfg["psi"]) * (((cfg["epsilon"] - 1.0) / cfg["epsilon"]) - w / jnp.exp(z))
        - (mu_z - 0.5 * cfg["sigma_z"] ** 2) * pi
    )


def savings(a_i: jnp.ndarray, n_i: jnp.ndarray, c_i: jnp.ndarray, r: jnp.ndarray, w: jnp.ndarray, pi_share: jnp.ndarray) -> jnp.ndarray:
    return r * a_i + w * n_i - c_i + pi_share


def penalty_deriv(a: jnp.ndarray, cfg: dict) -> jnp.ndarray:
    return jnp.where(a <= cfg["a_lb"], -cfg["kappa"] * (a - cfg["a_lb"]), 0.0)


def lambda_of_n(n: jnp.ndarray, cfg: dict) -> jnp.ndarray:
    # Match exactly one of the two states.
    is_low = jnp.isclose(n, cfg["n_low"], atol=1.0e-6)
    return jnp.where(is_low, cfg["lambda_1"], cfg["lambda_2"])


def flip_n(n: jnp.ndarray, cfg: dict) -> jnp.ndarray:
    is_low = jnp.isclose(n, cfg["n_low"], atol=1.0e-6)
    return jnp.where(is_low, cfg["n_high"], cfg["n_low"])


def split_agent_view(a_all: jnp.ndarray, n_all: jnp.ndarray, idx: int) -> Tuple[jnp.ndarray, ...]:
    a_i = a_all[idx]
    n_i = n_all[idx]
    a_others = jnp.concatenate([a_all[:idx], a_all[idx + 1 :]])
    n_others = jnp.concatenate([n_all[:idx], n_all[idx + 1 :]])
    return a_i, n_i, a_others, n_others


def evaluate_W_for_agent(model, idx: int, a_all: jnp.ndarray, n_all: jnp.ndarray, z: jnp.ndarray, pi: jnp.ndarray, w: jnp.ndarray, cfg: dict) -> jnp.ndarray:
    a_i, n_i, a_others, n_others = split_agent_view(a_all, n_all, idx)
    return w_forward(model, a_i, n_i, z, pi, w, a_others, n_others, cfg)


def evaluate_all_policies(model, a_all: jnp.ndarray, n_all: jnp.ndarray, z: jnp.ndarray, pi: jnp.ndarray, w: jnp.ndarray, cfg: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
    N = int(cfg["N"])
    W_all = jnp.stack([evaluate_W_for_agent(model, i, a_all, n_all, z, pi, w, cfg) for i in range(N)])
    c_all = jnp.power(jnp.maximum(W_all, 1.0e-12), -1.0 / cfg["gamma"])
    return W_all, c_all


def excess_savings(model, w: jnp.ndarray, z: jnp.ndarray, pi: jnp.ndarray, a_all: jnp.ndarray, n_all: jnp.ndarray, cfg: dict) -> jnp.ndarray:
    r = compute_r(pi, cfg)
    N_star = compute_N_star(n_all, cfg["N"])
    Pi = compute_Pi_transfer(w, z, pi, N_star, cfg)
    _, c_all = evaluate_all_policies(model, a_all, n_all, z, pi, w, cfg)
    # Empirical-measure convention: each (equal-weight) agent receives the per-capita transfer Pi.
    # Market clearing should therefore use the empirical average of savings.
    s_all = savings(a_all, n_all, c_all, r, w, Pi)
    return jnp.mean(s_all)


def _bisect(model, z: jnp.ndarray, pi: jnp.ndarray, a_all: jnp.ndarray, n_all: jnp.ndarray, cfg: dict, lo: float, hi: float) -> jnp.ndarray:
    f_lo = float(excess_savings(model, jnp.array(lo), z, pi, a_all, n_all, cfg))
    f_hi = float(excess_savings(model, jnp.array(hi), z, pi, a_all, n_all, cfg))
    if f_lo == 0.0:
        return jnp.array(lo)
    if f_hi == 0.0:
        return jnp.array(hi)
    # Expand bracket if needed.
    tries = 0
    while f_lo * f_hi > 0 and tries < 12:
        width = hi - lo
        lo -= width
        hi += width
        f_lo = float(excess_savings(model, jnp.array(lo), z, pi, a_all, n_all, cfg))
        f_hi = float(excess_savings(model, jnp.array(hi), z, pi, a_all, n_all, cfg))
        tries += 1

    mid = 0.5 * (lo + hi)
    for _ in range(int(cfg["w_bisect_steps"])):
        mid = 0.5 * (lo + hi)
        f_mid = float(excess_savings(model, jnp.array(mid), z, pi, a_all, n_all, cfg))
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return jnp.array(mid)


def _newton_refine(model, z: jnp.ndarray, pi: jnp.ndarray, a_all: jnp.ndarray, n_all: jnp.ndarray, cfg: dict, w0: jnp.ndarray) -> jnp.ndarray:
    w = w0
    g = jax.grad(lambda ww: excess_savings(model, ww, z, pi, a_all, n_all, cfg))
    for _ in range(int(cfg["w_newton_steps"])):
        f = excess_savings(model, w, z, pi, a_all, n_all, cfg)
        fp = g(w)
        step = jnp.where(jnp.abs(fp) > cfg["w_newton_eps"], f / fp, 0.0)
        w = w - cfg["w_newton_damp"] * step
    return w


def solve_w(model, z: jnp.ndarray, pi: jnp.ndarray, a_all: jnp.ndarray, n_all: jnp.ndarray, cfg: dict) -> jnp.ndarray:
    w_bi = _bisect(model, z, pi, a_all, n_all, cfg, cfg["w_bracket_min"], cfg["w_bracket_max"])
    return _newton_refine(model, z, pi, a_all, n_all, cfg, w_bi)

