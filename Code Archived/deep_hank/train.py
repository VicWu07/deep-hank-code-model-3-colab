"""Training loop for Deep HANK."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable, Dict, List

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .config import default_config
from .economics import compute_Pi_transfer, excess_savings, solve_w
from .model import init_model, w_forward_batch
from .residual import residual_batch, total_loss
from .sampling import sample_batch, sample_warmstart_batch
from .simulation import ReplayBuffer, init_replay_buffer, mix_batches, refresh_replay_buffer, sample_from_replay


@dataclass
class TrainState:
    model: eqx.Module
    opt_state: optax.OptState
    step: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)
    replay: ReplayBuffer | None = None


def _resolve_checkpoint_dir(checkpoint_dir: str | Path | None) -> Path | None:
    if checkpoint_dir is None:
        return None
    out = Path(checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_checkpoint(
    checkpoint_dir: Path | None,
    state: TrainState,
    progress: dict,
) -> None:
    if checkpoint_dir is None:
        return
    eqx.tree_serialise_leaves(checkpoint_dir / "model.eqx", state.model)
    _write_json(checkpoint_dir / "history.json", {"history": state.history})
    _write_json(
        checkpoint_dir / "state.json",
        {
            "step": int(state.step),
            "phase0_done": int(progress.get("phase0_done", 0)),
            "phase1_done": int(progress.get("phase1_done", 0)),
            "phase2_done": int(progress.get("phase2_done", 0)),
            "phase3_done": int(progress.get("phase3_done", 0)),
        },
    )


def _load_checkpoint(
    cfg: dict,
    checkpoint_dir: Path | None,
) -> tuple[eqx.Module | None, list[dict], dict]:
    empty_progress = {"phase0_done": 0, "phase1_done": 0, "phase2_done": 0, "phase3_done": 0}
    if checkpoint_dir is None:
        return None, [], empty_progress
    model_path = checkpoint_dir / "model.eqx"
    history_path = checkpoint_dir / "history.json"
    state_path = checkpoint_dir / "state.json"
    if not (model_path.exists() and history_path.exists() and state_path.exists()):
        return None, [], empty_progress

    # Build a skeleton model and load leaves.
    key = jax.random.PRNGKey(int(cfg["seed"]) + 99)
    model = init_model(key, cfg)
    model = eqx.tree_deserialise_leaves(model_path, model)

    history_obj = json.loads(history_path.read_text(encoding="utf-8"))
    history = list(history_obj.get("history", []))
    state_obj = json.loads(state_path.read_text(encoding="utf-8"))
    progress = {
        "phase0_done": int(state_obj.get("phase0_done", 0)),
        "phase1_done": int(state_obj.get("phase1_done", 0)),
        "phase2_done": int(state_obj.get("phase2_done", 0)),
        "phase3_done": int(state_obj.get("phase3_done", 0)),
    }
    return model, history, progress


def _cosine_lr(step: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    frac = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + jnp.cos(jnp.pi * frac))


def _make_optimizer(cfg: dict, total_steps: int):
    schedule = optax.inject_hyperparams(optax.adamw)(
        learning_rate=lambda step: _cosine_lr(int(step), total_steps, cfg["lr_max"], cfg["lr_min"]),
        weight_decay=cfg["weight_decay"],
    )
    return optax.chain(optax.clip_by_global_norm(cfg["grad_clip"]), schedule)


def _warmstart_loss(model, batch: dict, cfg: dict):
    b = batch["a"].shape[0]
    z = jnp.full((b,), cfg["z_bar"], dtype=jnp.float32)
    pi = jnp.zeros((b,), dtype=jnp.float32)
    w = jnp.full((b,), cfg["w_ss"], dtype=jnp.float32)
    a_others = jnp.full((b, cfg["N"] - 1), cfg["a_mid"], dtype=jnp.float32)
    n_others = jnp.full((b, cfg["N"] - 1), cfg["n_mid"], dtype=jnp.float32)

    W_pred = w_forward_batch(model, batch["a"], batch["n"], z, pi, w, a_others, n_others, cfg)
    N_star_ss = 0.5 * (cfg["n_low"] + cfg["n_high"])
    Pi_ss = compute_Pi_transfer(cfg["w_ss"], cfg["z_bar"], 0.0, N_star_ss, cfg)
    c_ss = cfg["w_ss"] * batch["n"] + cfg["r_ss"] * batch["a"] + Pi_ss
    c_ss = jnp.maximum(c_ss, 1.0e-6)
    W_tgt = c_ss ** (-cfg["gamma"])
    loss = jnp.mean((W_pred - W_tgt) ** 2)
    return loss, {"warmstart_mse": loss}


def _solve_w_batch(model, batch: dict, cfg: dict) -> jnp.ndarray:
    b = batch["z"].shape[0]
    out = []
    for i in range(b):
        out.append(
            solve_w(
                model,
                batch["z"][i],
                batch["pi"][i],
                batch["a_all"][i],
                batch["n_all"][i],
                cfg,
            )
        )
    return jnp.stack(out)


def _train_step(state: TrainState, optimizer, loss_fn):
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(state.model)
    updates, new_opt_state = optimizer.update(grads, state.opt_state, eqx.filter(state.model, eqx.is_array))
    new_model = eqx.apply_updates(state.model, updates)
    return loss, aux, TrainState(model=new_model, opt_state=new_opt_state, step=state.step + 1, history=state.history, replay=state.replay)


def _append_history(state: TrainState, phase: str, loss: jnp.ndarray, aux: dict):
    item = {"step": float(state.step), "phase": phase, "loss": float(loss)}
    for k, v in aux.items():
        item[k] = float(v)
    state.history.append(item)


def _maybe_log(cfg: dict, log_fn: Callable[[dict], None] | None, state: TrainState, phase: str, loss: jnp.ndarray, aux: dict):
    if not cfg.get("enable_step_logging", False):
        return
    step = int(state.step)
    has_debug_metrics = ("w_mean" in aux) or ("clear_abs_mean_cap" in aux)
    if (step % int(cfg.get("log_every_steps", 50)) != 0) and not has_debug_metrics:
        return
    payload = {"step": step, "phase": phase, "loss": float(loss)}
    for k, v in aux.items():
        payload[k] = float(v)
    if log_fn is None:
        print(payload)
    else:
        log_fn(payload)


def _phase3_debug_metrics(model, batch: dict, w_batch: jnp.ndarray, cfg: dict) -> dict:
    if not cfg.get("enable_debug_metrics", False):
        return {}
    cap = min(int(cfg.get("debug_metrics_batch_cap", 8)), int(batch["z"].shape[0]))
    errs = []
    for i in range(cap):
        f = excess_savings(
            model,
            w_batch[i],
            batch["z"][i],
            batch["pi"][i],
            batch["a_all"][i],
            batch["n_all"][i],
            cfg,
        )
        errs.append(jnp.abs(f))
    e = jnp.stack(errs) if errs else jnp.array([0.0], dtype=jnp.float32)
    return {
        "w_mean": float(jnp.mean(w_batch)),
        "w_std": float(jnp.std(w_batch)),
        "clear_abs_mean_cap": float(jnp.mean(e)),
        "clear_abs_max_cap": float(jnp.max(e)),
    }


def _phase3_eval_metrics(
    model,
    key: jax.Array,
    cfg: dict,
) -> dict:
    cap_bs = max(8, min(int(cfg["batch_size"]), int(cfg.get("debug_metrics_batch_cap", 8)) * 4))
    cfg_eval = dict(cfg)
    cfg_eval["batch_size"] = cap_bs
    batch = sample_batch(key, cfg_eval, phase=3, boundary_frac=cfg_eval["boundary_frac_phase3"])
    w_batch = _solve_w_batch(model, batch, cfg_eval)
    R = residual_batch(model, batch, w_batch, cfg_eval, include_aggregate=True, include_distribution=True)
    abs_r = jnp.abs(R)

    errs = []
    cap = min(int(cfg.get("debug_metrics_batch_cap", 8)), int(batch["z"].shape[0]))
    for i in range(cap):
        f = excess_savings(
            model,
            w_batch[i],
            batch["z"][i],
            batch["pi"][i],
            batch["a_all"][i],
            batch["n_all"][i],
            cfg_eval,
        )
        errs.append(jnp.abs(f))
    e = jnp.stack(errs) if errs else jnp.array([0.0], dtype=jnp.float32)
    return {
        "phase3_residual_abs_mean": float(jnp.mean(abs_r)),
        "phase3_residual_abs_p95": float(jnp.quantile(abs_r, 0.95)),
        "phase3_w_clear_abs_mean": float(jnp.mean(e)),
        "phase3_w_clear_abs_max": float(jnp.max(e)),
    }


def _mix_with_active_samples(
    model,
    key: jax.Array,
    batch: dict,
    cfg: dict,
) -> dict:
    if not bool(cfg.get("enable_active_sampling", False)):
        return batch
    candidate_size = max(int(cfg["batch_size"]), int(cfg.get("active_candidate_size", cfg["batch_size"])))
    cfg_cand = dict(cfg)
    cfg_cand["batch_size"] = candidate_size
    cand = sample_batch(key, cfg_cand, phase=3, boundary_frac=cfg_cand["boundary_frac_phase3"])
    w_cand = _solve_w_batch(model, cand, cfg_cand)
    r_cand = residual_batch(model, cand, w_cand, cfg_cand, include_aggregate=True, include_distribution=True)
    abs_r = jnp.abs(r_cand)
    k = max(1, int(float(cfg.get("active_topk_frac", 0.2)) * int(cfg["batch_size"])))
    k = min(k, int(abs_r.shape[0]))
    top_idx = jnp.argsort(abs_r)[-k:]

    out = dict(batch)
    for key_name, val in batch.items():
        if hasattr(val, "ndim") and val.ndim >= 1 and key_name in cand:
            replacement = cand[key_name][top_idx]
            out[key_name] = jnp.concatenate([replacement, val[k:]], axis=0)
    return out


def _converged(history: list[dict], cfg: dict) -> bool:
    if not bool(cfg.get("stop_on_convergence", False)):
        return False
    window = int(cfg.get("convergence_window", 50))
    if len(history) < window:
        return False
    tail = history[-window:]
    residual_vals = [h.get("residual_mse") for h in tail if "residual_mse" in h]
    clear_vals = [h.get("phase3_w_clear_abs_mean") for h in tail if "phase3_w_clear_abs_mean" in h]
    if not residual_vals or not clear_vals:
        return False
    return (
        float(sum(residual_vals) / len(residual_vals)) <= float(cfg.get("convergence_residual_mse_target", 1.0e-3))
        and float(sum(clear_vals) / len(clear_vals)) <= float(cfg.get("convergence_w_clear_target", 1.0e-4))
    )


def _phase1_loss(
    model,
    batch: dict,
    cfg: dict,
    alpha: float,
    ws_batch: dict | None = None,
) -> tuple[jnp.ndarray, dict]:
    w_batch = jnp.full((cfg["batch_size"],), cfg["w_ss"], dtype=jnp.float32)
    pde_loss, pde_aux = total_loss(
        model, batch, w_batch, cfg, include_aggregate=False, include_distribution=False
    )
    if alpha <= 0.0:
        return pde_loss, pde_aux
    if ws_batch is None:
        ws_batch = sample_warmstart_batch(jax.random.PRNGKey(int(cfg["seed"]) + 17), cfg)
    ws_loss, ws_aux = _warmstart_loss(model, ws_batch, cfg)
    blended = alpha * ws_loss + (1.0 - alpha) * pde_loss
    out_aux = dict(pde_aux)
    out_aux.update({"phase1_alpha": alpha})
    out_aux.update({"warmstart_mse": ws_aux["warmstart_mse"]})
    return blended, out_aux


def train(
    cfg: dict | None = None,
    log_fn: Callable[[dict], None] | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
):
    cfg = default_config() if cfg is None else cfg
    total_steps = cfg["phase0_steps"] + cfg["phase1_steps"] + cfg["phase2_steps"] + cfg["phase3_steps"]

    ckpt_dir = _resolve_checkpoint_dir(checkpoint_dir)
    progress = {"phase0_done": 0, "phase1_done": 0, "phase2_done": 0, "phase3_done": 0}

    key = jax.random.PRNGKey(cfg["seed"])
    model = None
    history = []
    if resume:
        model, history, progress = _load_checkpoint(cfg, ckpt_dir)
    if model is None:
        key, k_model = jax.random.split(key)
        model = init_model(k_model, cfg)
    optimizer = _make_optimizer(cfg, total_steps)
    state = TrainState(
        model=model,
        opt_state=optimizer.init(eqx.filter(model, eqx.is_array)),
        step=len(history),
        history=history,
        replay=init_replay_buffer(cfg),
    )

    # Phase 0: warm-start.
    for _ in range(int(progress["phase0_done"]), int(cfg["phase0_steps"])):
        key, k = jax.random.split(key)
        batch = sample_warmstart_batch(k, cfg)
        loss_fn = lambda m: _warmstart_loss(m, batch, cfg)
        loss, aux, state = _train_step(state, optimizer, loss_fn)
        progress["phase0_done"] += 1
        _append_history(state, "phase0_warmstart", loss, aux)
        _maybe_log(cfg, log_fn, state, "phase0_warmstart", loss, aux)
        if state.step % int(cfg.get("checkpoint_every", 1000)) == 0:
            _write_checkpoint(ckpt_dir, state, progress)

    # Phase 1: steady-state PDE.
    for i in range(int(progress["phase1_done"]), int(cfg["phase1_steps"])):
        key, k, k_ws = jax.random.split(key, 3)
        batch = sample_batch(k, cfg, phase=1, boundary_frac=cfg["boundary_frac_phase1"])
        alpha = 0.0
        ws_batch = None
        if bool(cfg.get("enable_phase1_curriculum", False)):
            steps = max(1, int(cfg.get("phase1_alpha_steps", 1000)))
            alpha = max(0.0, 1.0 - float(i) / float(steps))
            ws_batch = sample_warmstart_batch(k_ws, cfg)
        loss_fn = lambda m: _phase1_loss(m, batch, cfg, alpha, ws_batch=ws_batch)
        loss, aux, state = _train_step(state, optimizer, loss_fn)
        progress["phase1_done"] += 1
        _append_history(state, "phase1_steady", loss, aux)
        _maybe_log(cfg, log_fn, state, "phase1_steady", loss, aux)
        if state.step % int(cfg.get("checkpoint_every", 1000)) == 0:
            _write_checkpoint(ckpt_dir, state, progress)

    # Phase 2: full PDE, fixed w-rule.
    for i in range(int(progress["phase2_done"]), int(cfg["phase2_steps"])):
        key, k = jax.random.split(key)
        cfg_phase2 = cfg
        if bool(cfg.get("enable_phase2_range_curriculum", False)):
            frac0 = float(cfg.get("phase2_range_start_frac", 0.25))
            steps = max(1, int(cfg.get("phase2_alpha_steps", 1000)))
            alpha = min(1.0, float(i) / float(steps))
            z_half = 0.5 * (cfg["z_max"] - cfg["z_min"]) * (frac0 + (1.0 - frac0) * alpha)
            pi_half = 0.5 * (cfg["pi_max"] - cfg["pi_min"]) * (frac0 + (1.0 - frac0) * alpha)
            cfg_phase2 = dict(cfg)
            cfg_phase2["z_min"] = cfg["z_bar"] - z_half
            cfg_phase2["z_max"] = cfg["z_bar"] + z_half
            cfg_phase2["pi_min"] = -pi_half
            cfg_phase2["pi_max"] = pi_half
        batch = sample_batch(k, cfg_phase2, phase=2, boundary_frac=cfg["boundary_frac_phase2"])
        w_batch = jnp.full((cfg["batch_size"],), cfg["w_ss"], dtype=jnp.float32)
        loss_fn = lambda m: total_loss(
            m, batch, w_batch, cfg, include_aggregate=True, include_distribution=True
        )
        loss, aux, state = _train_step(state, optimizer, loss_fn)
        progress["phase2_done"] += 1
        _append_history(state, "phase2_full_fixed_w", loss, aux)
        _maybe_log(cfg, log_fn, state, "phase2_full_fixed_w", loss, aux)
        if state.step % int(cfg.get("checkpoint_every", 1000)) == 0:
            _write_checkpoint(ckpt_dir, state, progress)

    # Phase 3: full PDE + outer loop w-solve + replay mix.
    for t in range(int(progress["phase3_done"]), int(cfg["phase3_steps"])):
        key, k_sample, k_rep = jax.random.split(key, 3)
        batch = sample_batch(k_sample, cfg, phase=3, boundary_frac=cfg["boundary_frac_phase3"])

        if t % int(cfg["sim_refresh_every"]) == 0:
            key, k_refresh = jax.random.split(key)
            state.replay = refresh_replay_buffer(k_refresh, state.model, cfg, state.replay)

        if state.replay is not None and state.replay.size > 0:
            replay_batch = sample_from_replay(k_rep, state.replay, cfg["batch_size"], cfg)
            batch = mix_batches(batch, replay_batch, cfg["sim_mix_frac"])

        if bool(cfg.get("enable_active_sampling", False)) and (t % int(cfg.get("active_sampling_every", 25)) == 0):
            key, k_active = jax.random.split(key)
            batch = _mix_with_active_samples(state.model, k_active, batch, cfg)

        w_batch = _solve_w_batch(state.model, batch, cfg)
        loss_fn = lambda m: total_loss(
            m, batch, w_batch, cfg, include_aggregate=True, include_distribution=True
        )
        loss, aux, state = _train_step(state, optimizer, loss_fn)
        aux.update(_phase3_debug_metrics(state.model, batch, w_batch, cfg))
        progress["phase3_done"] += 1

        if (state.step % int(cfg.get("diagnostic_every", 500))) == 0:
            key, k_eval = jax.random.split(key)
            aux.update(_phase3_eval_metrics(state.model, k_eval, cfg))

        _append_history(state, "phase3_outer_replay", loss, aux)
        _maybe_log(cfg, log_fn, state, "phase3_outer_replay", loss, aux)
        if state.step % int(cfg.get("checkpoint_every", 1000)) == 0:
            _write_checkpoint(ckpt_dir, state, progress)
        if _converged(state.history, cfg):
            break

    _write_checkpoint(ckpt_dir, state, progress)
    return state.model, state.history

