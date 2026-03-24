"""Sampling utilities for Deep HANK training."""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp


def _draw_n_states(key: jax.Array, batch_size: int, N: int, cfg: dict) -> jnp.ndarray:
    p_high = cfg["lambda_1"] / (cfg["lambda_1"] + cfg["lambda_2"])
    is_high = jax.random.bernoulli(key, p=p_high, shape=(batch_size, N))
    return jnp.where(is_high, cfg["n_high"], cfg["n_low"]).astype(jnp.float32)


def _project_assets_zero_supply(a_all: jnp.ndarray) -> jnp.ndarray:
    centered = a_all - jnp.mean(a_all, axis=1, keepdims=True)
    return centered


def _enforce_asset_bounds(a_all: jnp.ndarray, cfg: dict) -> jnp.ndarray:
    clipped = jnp.clip(a_all, cfg["a_bar"], cfg["a_max"])
    # Recenter after clipping so stock-clearing remains exact in finite precision.
    return clipped - jnp.mean(clipped, axis=1, keepdims=True)


def _split_focal(a_all: jnp.ndarray, n_all: jnp.ndarray, focal_idx: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
    batch_size, N = a_all.shape
    row = jnp.arange(batch_size)
    a_i = a_all[row, focal_idx]
    n_i = n_all[row, focal_idx]

    mask = jnp.ones((batch_size, N), dtype=bool).at[row, focal_idx].set(False)
    a_others = a_all[mask].reshape(batch_size, N - 1)
    n_others = n_all[mask].reshape(batch_size, N - 1)
    return a_i, n_i, a_others, n_others


def sample_batch(key: jax.Array, cfg: dict, phase: int, boundary_frac: float = 0.2) -> Dict[str, jnp.ndarray]:
    batch_size = int(cfg["batch_size"])
    N = int(cfg["N"])

    k_z, k_pi, k_n, k_a, k_focal, k_mix, k_bound = jax.random.split(key, 7)
    z = jax.random.uniform(k_z, (batch_size,), minval=cfg["z_min"], maxval=cfg["z_max"]).astype(jnp.float32)
    pi = jax.random.uniform(k_pi, (batch_size,), minval=cfg["pi_min"], maxval=cfg["pi_max"]).astype(jnp.float32)

    n_all = _draw_n_states(k_n, batch_size, N, cfg)
    a_all = jax.random.uniform(k_a, (batch_size, N), minval=cfg["a_bar"], maxval=cfg["a_max"]).astype(jnp.float32)
    a_all = _project_assets_zero_supply(a_all)
    a_all = _enforce_asset_bounds(a_all, cfg)

    focal_idx = jax.random.randint(k_focal, (batch_size,), minval=0, maxval=N)
    a_i, n_i, a_others, n_others = _split_focal(a_all, n_all, focal_idx)

    # Boundary oversampling for focal states.
    mix = jax.random.bernoulli(k_mix, boundary_frac, shape=(batch_size,))
    a_i_boundary = jax.random.uniform(k_bound, (batch_size,), minval=cfg["a_bar"], maxval=cfg["a_lb"]).astype(jnp.float32)
    a_i = jnp.where(mix, a_i_boundary, a_i)

    return {
        "z": z,
        "pi": pi,
        "a_all": a_all,
        "n_all": n_all,
        "focal_idx": focal_idx,
        "a_i": a_i,
        "n_i": n_i,
        "a_others": a_others,
        "n_others": n_others,
        "phase": jnp.array(phase),
    }


def sample_warmstart_batch(key: jax.Array, cfg: dict) -> Dict[str, jnp.ndarray]:
    batch_size = int(cfg["batch_size"])
    k_a, k_n = jax.random.split(key, 2)
    a = jax.random.uniform(k_a, (batch_size,), minval=cfg["a_bar"], maxval=cfg["a_max"]).astype(jnp.float32)
    n_is_high = jax.random.bernoulli(k_n, 0.5, shape=(batch_size,))
    n = jnp.where(n_is_high, cfg["n_high"], cfg["n_low"]).astype(jnp.float32)
    return {"a": a, "n": n}

