"""Level-1 finite-agent simulation and replay-buffer utilities."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .economics import compute_r, compute_Pi_transfer, evaluate_all_policies, lambda_of_n, mu_pi, solve_w


@dataclass
class ReplayBuffer:
    z: jnp.ndarray
    pi: jnp.ndarray
    a_all: jnp.ndarray
    n_all: jnp.ndarray
    ptr: int
    size: int


def init_replay_buffer(cfg: dict) -> ReplayBuffer:
    m = int(cfg["sim_replay_size"])
    N = int(cfg["N"])
    return ReplayBuffer(
        z=jnp.zeros((m,), dtype=jnp.float32),
        pi=jnp.zeros((m,), dtype=jnp.float32),
        a_all=jnp.zeros((m, N), dtype=jnp.float32),
        n_all=jnp.full((m, N), cfg["n_low"], dtype=jnp.float32),
        ptr=0,
        size=0,
    )


def _write_buffer(buf: ReplayBuffer, z: jnp.ndarray, pi: jnp.ndarray, a_all: jnp.ndarray, n_all: jnp.ndarray) -> ReplayBuffer:
    m = buf.z.shape[0]
    idx = buf.ptr % m
    new = ReplayBuffer(
        z=buf.z.at[idx].set(z),
        pi=buf.pi.at[idx].set(pi),
        a_all=buf.a_all.at[idx].set(a_all),
        n_all=buf.n_all.at[idx].set(n_all),
        ptr=(buf.ptr + 1) % m,
        size=min(buf.size + 1, m),
    )
    return new


def _simulate_one_step(key: jax.Array, model, z: jnp.ndarray, pi: jnp.ndarray, a_all: jnp.ndarray, n_all: jnp.ndarray, cfg: dict):
    dt = cfg["dt_sim"]
    k_eps, k_emp = jax.random.split(key, 2)

    w_t = solve_w(model, z, pi, a_all, n_all, cfg)
    r_t = compute_r(pi, cfg)
    N_star = jnp.mean(n_all)
    Pi_t = compute_Pi_transfer(w_t, z, pi, N_star, cfg)

    W_all, c_all = evaluate_all_policies(model, a_all, n_all, z, pi, w_t, cfg)
    s_all = r_t * a_all + w_t * n_all - c_all + Pi_t / cfg["N"]

    eps = jax.random.normal(k_eps, ())
    z_next = z + cfg["eta_z"] * (cfg["z_bar"] - z) * dt + cfg["sigma_z"] * jnp.sqrt(dt) * eps
    pi_next = pi + mu_pi(z, pi, w_t, cfg) * dt - cfg["sigma_z"] * pi * jnp.sqrt(dt) * eps

    emp_flip = jax.random.bernoulli(k_emp, p=jnp.clip(lambda_of_n(n_all, cfg) * dt, 0.0, 1.0), shape=n_all.shape)
    n_next = jnp.where(emp_flip, jnp.where(jnp.isclose(n_all, cfg["n_low"]), cfg["n_high"], cfg["n_low"]), n_all)

    a_next = a_all + s_all * dt
    a_next = jnp.clip(a_next, cfg["a_bar"], cfg["a_max"])
    a_next = a_next - jnp.mean(a_next)
    return z_next.astype(jnp.float32), pi_next.astype(jnp.float32), a_next.astype(jnp.float32), n_next.astype(jnp.float32), w_t


def refresh_replay_buffer(key: jax.Array, model, cfg: dict, buffer: ReplayBuffer) -> ReplayBuffer:
    """Generate short paths and store visited states."""
    N = int(cfg["N"])
    k0, key = jax.random.split(key)
    z = jax.random.uniform(k0, (), minval=cfg["z_min"], maxval=cfg["z_max"])
    k1, key = jax.random.split(key)
    pi = jax.random.uniform(k1, (), minval=cfg["pi_min"], maxval=cfg["pi_max"])
    k2, key = jax.random.split(key)
    a_all = jax.random.uniform(k2, (N,), minval=cfg["a_bar"], maxval=cfg["a_max"])
    a_all = a_all - jnp.mean(a_all)
    k3, key = jax.random.split(key)
    n_high = jax.random.bernoulli(k3, 0.5, shape=(N,))
    n_all = jnp.where(n_high, cfg["n_high"], cfg["n_low"]).astype(jnp.float32)

    out = buffer
    for _ in range(int(cfg["sim_path_steps"])):
        k_step, key = jax.random.split(key)
        z, pi, a_all, n_all, _ = _simulate_one_step(k_step, model, z, pi, a_all, n_all, cfg)
        out = _write_buffer(out, z, pi, a_all, n_all)
    return out


def sample_from_replay(key: jax.Array, buffer: ReplayBuffer, batch_size: int, cfg: dict) -> dict:
    if buffer.size == 0:
        raise ValueError("Replay buffer is empty. Refresh before sampling.")
    idx = jax.random.randint(key, (batch_size,), minval=0, maxval=buffer.size)
    z = buffer.z[idx]
    pi = buffer.pi[idx]
    a_all = buffer.a_all[idx]
    n_all = buffer.n_all[idx]

    focal_idx = jnp.zeros((batch_size,), dtype=jnp.int32)
    a_i = a_all[:, 0]
    n_i = n_all[:, 0]
    a_others = a_all[:, 1:]
    n_others = n_all[:, 1:]
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
    }


def mix_batches(base_batch: dict, replay_batch: dict, replay_frac: float) -> dict:
    b = base_batch["a_i"].shape[0]
    k = int(replay_frac * b)
    if k <= 0:
        return base_batch
    out = {}
    for key in base_batch:
        if key in replay_batch and getattr(base_batch[key], "ndim", 0) >= 1:
            out[key] = jnp.concatenate([replay_batch[key][:k], base_batch[key][k:]], axis=0)
        else:
            out[key] = base_batch[key]
    return out


def simulate_path(
    key: jax.Array,
    model,
    cfg: dict,
    steps: int = 200,
    dt: float | None = None,
    init_state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
) -> dict:
    """Simulate a finite-agent path and return arrays for diagnostics.

    Returns a dict with keys:
      - z: (T+1,)
      - pi: (T+1,)
      - w: (T,)
      - a_all: (T+1, N)
      - n_all: (T+1, N)
    """
    N = int(cfg["N"])
    cfg_local = dict(cfg)
    if dt is not None:
        cfg_local["dt_sim"] = float(dt)

    if init_state is None:
        k0, k1, k2, k3 = jax.random.split(key, 4)
        z0 = jax.random.uniform(k0, (), minval=cfg_local["z_min"], maxval=cfg_local["z_max"])
        pi0 = jax.random.uniform(k1, (), minval=cfg_local["pi_min"], maxval=cfg_local["pi_max"])
        a0 = jax.random.uniform(k2, (N,), minval=cfg_local["a_bar"], maxval=cfg_local["a_max"])
        a0 = a0 - jnp.mean(a0)
        n_high = jax.random.bernoulli(k3, 0.5, shape=(N,))
        n0 = jnp.where(n_high, cfg_local["n_high"], cfg_local["n_low"]).astype(jnp.float32)
    else:
        z0, pi0, a0, n0 = init_state

    z_hist = [z0.astype(jnp.float32)]
    pi_hist = [pi0.astype(jnp.float32)]
    a_hist = [a0.astype(jnp.float32)]
    n_hist = [n0.astype(jnp.float32)]
    w_hist = []

    z_t, pi_t, a_t, n_t = z_hist[0], pi_hist[0], a_hist[0], n_hist[0]
    for _ in range(int(steps)):
        k_step, key = jax.random.split(key)
        z_t, pi_t, a_t, n_t, w_t = _simulate_one_step(k_step, model, z_t, pi_t, a_t, n_t, cfg_local)
        w_hist.append(w_t.astype(jnp.float32))
        z_hist.append(z_t)
        pi_hist.append(pi_t)
        a_hist.append(a_t)
        n_hist.append(n_t)

    return {
        "z": jnp.stack(z_hist),
        "pi": jnp.stack(pi_hist),
        "w": jnp.stack(w_hist) if w_hist else jnp.zeros((0,), dtype=jnp.float32),
        "a_all": jnp.stack(a_hist),
        "n_all": jnp.stack(n_hist),
    }

