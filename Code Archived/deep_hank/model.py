"""Neural network model for W = dV/da."""

from __future__ import annotations

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp


def input_dim(cfg: dict) -> int:
    """Input is (a_i, n_i, z, pi, w, a_others, n_others)."""
    return 5 + 2 * (cfg["N"] - 1)


def normalize_scalar_state(a: jnp.ndarray, n: jnp.ndarray, z: jnp.ndarray, pi: jnp.ndarray, cfg: dict) -> Tuple[jnp.ndarray, ...]:
    a_norm = (a - cfg["a_mid"]) / cfg["a_scale"]
    n_norm = (n - cfg["n_mid"]) / cfg["n_scale"]
    z_norm = (z - cfg["z_bar"]) / cfg["z_scale"]
    pi_norm = pi / cfg["pi_scale"]
    return a_norm, n_norm, z_norm, pi_norm


def normalize_others(a_others: jnp.ndarray, n_others: jnp.ndarray, cfg: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
    a_others_norm = (a_others - cfg["a_mid"]) / cfg["a_scale"]
    n_others_norm = (n_others - cfg["n_mid"]) / cfg["n_scale"]
    return a_others_norm, n_others_norm


def build_input(
    a_i: jnp.ndarray,
    n_i: jnp.ndarray,
    z: jnp.ndarray,
    pi: jnp.ndarray,
    w: jnp.ndarray,
    a_others: jnp.ndarray,
    n_others: jnp.ndarray,
    cfg: dict,
) -> jnp.ndarray:
    a_norm, n_norm, z_norm, pi_norm = normalize_scalar_state(a_i, n_i, z, pi, cfg)
    a_others_norm, n_others_norm = normalize_others(a_others, n_others, cfg)
    core = jnp.array([a_norm, n_norm, z_norm, pi_norm, w], dtype=jnp.float32)
    return jnp.concatenate([core, a_others_norm.astype(jnp.float32), n_others_norm.astype(jnp.float32)], axis=0)


class WNetwork(eqx.Module):
    """Simple MLP with softplus output for W > 0."""

    mlp: eqx.nn.MLP
    softplus_eps: float

    def __init__(self, key: jax.Array, cfg: dict):
        self.mlp = eqx.nn.MLP(
            in_size=input_dim(cfg),
            out_size=1,
            width_size=cfg["nn_width"],
            depth=cfg["nn_layers"],
            activation=jax.nn.tanh,
            final_activation=lambda x: x,
            key=key,
        )
        self.softplus_eps = float(cfg["w_softplus_eps"])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        raw = self.mlp(x)
        return jax.nn.softplus(raw.squeeze(-1)) + self.softplus_eps


def init_model(key: jax.Array, cfg: dict) -> WNetwork:
    return WNetwork(key, cfg)


def w_forward(
    model: WNetwork,
    a_i: jnp.ndarray,
    n_i: jnp.ndarray,
    z: jnp.ndarray,
    pi: jnp.ndarray,
    w: jnp.ndarray,
    a_others: jnp.ndarray,
    n_others: jnp.ndarray,
    cfg: dict,
) -> jnp.ndarray:
    x = build_input(a_i, n_i, z, pi, w, a_others, n_others, cfg)
    return model(x)


def w_forward_batch(
    model: WNetwork,
    a_i: jnp.ndarray,
    n_i: jnp.ndarray,
    z: jnp.ndarray,
    pi: jnp.ndarray,
    w: jnp.ndarray,
    a_others: jnp.ndarray,
    n_others: jnp.ndarray,
    cfg: dict,
) -> jnp.ndarray:
    f = jax.vmap(
        lambda ai, ni, zi, pii, wi, ao, no: w_forward(model, ai, ni, zi, pii, wi, ao, no, cfg),
        in_axes=(0, 0, 0, 0, 0, 0, 0),
    )
    return f(a_i, n_i, z, pi, w, a_others, n_others)

