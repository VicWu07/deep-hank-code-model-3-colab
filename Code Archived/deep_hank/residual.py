"""Master-equation residual and losses."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .economics import (
    compute_Pi_transfer,
    compute_mu_z,
    compute_r,
    flip_n,
    lambda_of_n,
    mu_pi,
    penalty_deriv,
    savings,
)
from .model import w_forward


def _agent_view_from_full(a_full: jnp.ndarray, n_full: jnp.ndarray, idx: int):
    a_i = a_full[idx]
    n_i = n_full[idx]
    a_others = jnp.concatenate([a_full[:idx], a_full[idx + 1 :]])
    n_others = jnp.concatenate([n_full[:idx], n_full[idx + 1 :]])
    return a_i, n_i, a_others, n_others


def _W_scalar(model, cfg, a_i, n_i, z, pi, w, a_others, n_others):
    return w_forward(model, a_i, n_i, z, pi, w, a_others, n_others, cfg)


def pde_residual_single(
    model,
    a_i: jnp.ndarray,
    n_i: jnp.ndarray,
    z: jnp.ndarray,
    pi: jnp.ndarray,
    w: jnp.ndarray,
    a_others: jnp.ndarray,
    n_others: jnp.ndarray,
    cfg: dict,
    include_aggregate: bool = True,
    include_distribution: bool = True,
) -> jnp.ndarray:
    # Price-taking detail: treat solved equilibrium price as given in derivatives.
    w_fixed = jax.lax.stop_gradient(w)
    N = int(cfg["N"])

    W_i = _W_scalar(model, cfg, a_i, n_i, z, pi, w_fixed, a_others, n_others)
    c_i = jnp.power(jnp.maximum(W_i, 1.0e-12), -1.0 / cfg["gamma"])

    a_full = jnp.concatenate([jnp.array([a_i]), a_others])
    n_full = jnp.concatenate([jnp.array([n_i]), n_others])
    r = compute_r(pi, cfg)
    N_star = jnp.mean(n_full)
    Pi = compute_Pi_transfer(w_fixed, z, pi, N_star, cfg)
    s_i = savings(a_i, n_i, c_i, r, w_fixed, Pi)

    dW_da_i = jax.grad(lambda ai: _W_scalar(model, cfg, ai, n_i, z, pi, w_fixed, a_others, n_others))(a_i)
    W_i_flip = _W_scalar(model, cfg, a_i, flip_n(n_i, cfg), z, pi, w_fixed, a_others, n_others)
    Lx = dW_da_i * s_i + lambda_of_n(n_i, cfg) * (W_i_flip - W_i)

    Lz = 0.0
    Lpi = 0.0
    Lzpi = 0.0
    if include_aggregate:
        dW_dz = jax.grad(lambda zz: _W_scalar(model, cfg, a_i, n_i, zz, pi, w_fixed, a_others, n_others))(z)
        d2W_dz2 = jax.grad(
            lambda zz: jax.grad(lambda zzz: _W_scalar(model, cfg, a_i, n_i, zzz, pi, w_fixed, a_others, n_others))(zz)
        )(z)
        dW_dpi = jax.grad(lambda pp: _W_scalar(model, cfg, a_i, n_i, z, pp, w_fixed, a_others, n_others))(pi)
        d2W_dpi2 = jax.grad(
            lambda pp: jax.grad(lambda ppp: _W_scalar(model, cfg, a_i, n_i, z, ppp, w_fixed, a_others, n_others))(pp)
        )(pi)
        d2W_dzdpi = jax.grad(
            lambda pp: jax.grad(lambda zz: _W_scalar(model, cfg, a_i, n_i, zz, pp, w_fixed, a_others, n_others))(z)
        )(pi)

        sig2 = cfg["sigma_z"] ** 2
        mu_z_val = compute_mu_z(z, cfg)
        mu_pi_val = mu_pi(z, pi, w_fixed, cfg)
        Lz = dW_dz * mu_z_val + 0.5 * sig2 * d2W_dz2
        Lpi = dW_dpi * mu_pi_val + 0.5 * sig2 * pi**2 * d2W_dpi2
        Lzpi = -sig2 * pi * d2W_dzdpi

    Lg = 0.0
    if include_distribution:
        dW_da_others = jax.jacrev(lambda ao: _W_scalar(model, cfg, a_i, n_i, z, pi, w_fixed, ao, n_others))(a_others)

        acc = 0.0
        for k in range(N - 1):
            j = k + 1
            n_others_flip_k = n_others.at[k].set(flip_n(n_others[k], cfg))
            W_i_flip_j = _W_scalar(model, cfg, a_i, n_i, z, pi, w_fixed, a_others, n_others_flip_k)

            a_j, n_j, a_others_j, n_others_j = _agent_view_from_full(a_full, n_full, j)
            W_j = _W_scalar(model, cfg, a_j, n_j, z, pi, w_fixed, a_others_j, n_others_j)
            c_j = jnp.power(jnp.maximum(W_j, 1.0e-12), -1.0 / cfg["gamma"])
            s_j = savings(a_j, n_j, c_j, r, w_fixed, Pi)

            acc = acc + dW_da_others[k] * s_j + lambda_of_n(n_j, cfg) * (W_i_flip_j - W_i)

        # Finite-agent distribution operator (GLMP/EMINN eq. (3.1)) is a sum over j≠i.
        # Any 1/N scaling arises endogenously if the network depends on others through
        # the empirical measure rather than their labeled positions.
        Lg = acc

    return (r - cfg["rho"]) * W_i + penalty_deriv(a_i, cfg) + Lx + Lz + Lpi + Lzpi + Lg


def _shape_penalty_single(model, a_i, n_i, z, pi, w, a_others, n_others, cfg):
    w_fixed = jax.lax.stop_gradient(w)
    dW_da_i = jax.grad(lambda ai: _W_scalar(model, cfg, ai, n_i, z, pi, w_fixed, a_others, n_others))(a_i)
    dW_dz = jax.grad(lambda zz: _W_scalar(model, cfg, a_i, n_i, zz, pi, w_fixed, a_others, n_others))(z)
    da_penalty = jax.nn.relu(dW_da_i - cfg["shape_upper_bound_da"]) ** 2
    dz_penalty = jax.nn.relu(dW_dz) ** 2
    return da_penalty + dz_penalty


def residual_batch(model, batch: dict, w_batch: jnp.ndarray, cfg: dict, include_aggregate: bool = True, include_distribution: bool = True):
    vf = jax.vmap(
        lambda ai, ni, z, pi, w, ao, no: pde_residual_single(
            model, ai, ni, z, pi, w, ao, no, cfg, include_aggregate=include_aggregate, include_distribution=include_distribution
        ),
        in_axes=(0, 0, 0, 0, 0, 0, 0),
    )
    return vf(batch["a_i"], batch["n_i"], batch["z"], batch["pi"], w_batch, batch["a_others"], batch["n_others"])


def shape_loss(model, batch: dict, w_batch: jnp.ndarray, cfg: dict) -> jnp.ndarray:
    vf = jax.vmap(
        lambda ai, ni, z, pi, w, ao, no: _shape_penalty_single(model, ai, ni, z, pi, w, ao, no, cfg),
        in_axes=(0, 0, 0, 0, 0, 0, 0),
    )
    return jnp.mean(vf(batch["a_i"], batch["n_i"], batch["z"], batch["pi"], w_batch, batch["a_others"], batch["n_others"]))


def total_loss(
    model,
    batch: dict,
    w_batch: jnp.ndarray,
    cfg: dict,
    include_aggregate: bool = True,
    include_distribution: bool = True,
) -> tuple[jnp.ndarray, dict]:
    R = residual_batch(model, batch, w_batch, cfg, include_aggregate=include_aggregate, include_distribution=include_distribution)
    Ee = jnp.mean(R**2)
    Es = shape_loss(model, batch, w_batch, cfg)
    loss = cfg["kappa_e"] * Ee + cfg["kappa_s"] * Es
    return loss, {"residual_mse": Ee, "shape_loss": Es}

