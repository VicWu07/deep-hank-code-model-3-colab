"""Configuration for the Deep HANK JAX implementation."""

from __future__ import annotations

from copy import deepcopy


CALIBRATION = {
    "gamma": 2.1,
    "rho": 0.05,
    "a_bar": -2.0,
    "a_lb": 0.0,
    "a_max": 20.0,
    "kappa": 3.0,
    "n_low": 0.3,
    "n_high": 1.7,
    "lambda_1": 0.4,
    "lambda_2": 0.4,
    "epsilon": 6.0,
    "psi": 1.0,
    "phi_pi": 1.75,
    "r_star": 0.02,
    "eta_z": 0.1,
    "z_bar": 0.0,
    "sigma_z": 0.02,
    "N": 25,
    "z_min": -0.3,
    "z_max": 0.3,
    "pi_min": -0.05,
    "pi_max": 0.05,
    "kappa_e": 100.0,
    "kappa_s": 1.0,
}


MODEL_DEFAULTS = {
    "nn_width": 64,
    "nn_layers": 5,
    "w_softplus_eps": 1.0e-8,
}


TRAINING_DEFAULTS = {
    "seed": 42,
    "batch_size": 2048,
    "lr_max": 1.0e-3,
    "lr_min": 1.0e-6,
    "weight_decay": 1.0e-6,
    "grad_clip": 1.0,
    "phase0_steps": 1500,
    "phase1_steps": 3000,
    "phase2_steps": 5000,
    "phase3_steps": 10000,
    "phase1_alpha_steps": 1000,
    "phase2_alpha_steps": 1000,
    "boundary_frac_phase0": 0.0,
    "boundary_frac_phase1": 0.2,
    "boundary_frac_phase2": 0.3,
    "boundary_frac_phase3": 0.3,
    "steady_w_guess": 1.0,
    "w_bracket_min": -2.0,
    "w_bracket_max": 4.0,
    "w_bisect_steps": 40,
    "w_newton_steps": 2,
    "w_newton_damp": 0.5,
    "w_newton_eps": 1.0e-6,
    "shape_upper_bound_da": 0.0,
    "dt_sim": 0.05,
    "sim_path_steps": 64,
    "sim_refresh_every": 200,
    "sim_replay_size": 4096,
    "sim_mix_frac": 0.3,
    "diagnostic_every": 500,
    "checkpoint_every": 1000,
    # Optional training curricula toggles.
    "enable_phase1_curriculum": False,
    "enable_phase2_range_curriculum": False,
    "phase2_range_start_frac": 0.25,
    # Optional active-sampling controls (phase 3).
    "enable_active_sampling": False,
    "active_sampling_every": 25,
    "active_candidate_size": 1024,
    "active_topk_frac": 0.2,
    "active_mix_frac": 0.3,
    # Optional convergence stopping.
    "stop_on_convergence": False,
    "convergence_window": 50,
    "convergence_residual_mse_target": 1.0e-3,
    "convergence_w_clear_target": 1.0e-4,
    # Runtime logging / debug controls.
    "enable_step_logging": False,
    "log_every_steps": 50,
    "enable_debug_metrics": False,
    "debug_metrics_batch_cap": 8,
    # Diagnostics pass/fail thresholds used by diagnostics.json.
    "diag_pass_residual_mse_max": 1.0e-3,
    "diag_pass_shape_loss_max": 1.0e-2,
    "diag_pass_residual_log10_abs_R_max": -1.5,
    "diag_pass_w_clear_p95_abs_max": 1.0e-4,
    "diag_pass_w_clear_max_abs_max": 5.0e-4,
    "diag_pass_policy_max_monotonicity_violations": 0,
    "diag_pass_sim_max_abs_mean_a": 0.5,
    "diag_pass_sim_share_high_min": 0.01,
    "diag_pass_sim_share_high_max": 0.99,
}


def _derived(cfg: dict) -> dict:
    out = deepcopy(cfg)
    out["n_high"] = 1.0 + (out["lambda_2"] / out["lambda_1"]) * (1.0 - out["n_low"])
    out["r_ss"] = out["r_star"]
    out["w_ss"] = TRAINING_DEFAULTS["steady_w_guess"]
    out["a_mid"] = 0.5 * (out["a_max"] + out["a_bar"])
    out["a_scale"] = 0.5 * (out["a_max"] - out["a_bar"])
    out["n_mid"] = 0.5 * (out["n_high"] + out["n_low"])
    out["n_scale"] = 0.5 * (out["n_high"] - out["n_low"])
    out["z_scale"] = max(abs(out["z_max"] - out["z_min"]), 1.0e-8)
    out["pi_scale"] = max(abs(out["pi_max"] - out["pi_min"]), 1.0e-8)
    return out


def default_config() -> dict:
    """Return a full mutable config dict used by all modules."""
    cfg = {}
    cfg.update(_derived(CALIBRATION))
    cfg.update(MODEL_DEFAULTS)
    cfg.update(TRAINING_DEFAULTS)
    return cfg

