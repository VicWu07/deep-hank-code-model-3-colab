from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Preferences / idiosyncratic risk.
    gam: float = 2.1
    rho: float = 0.05
    lam1: float = 0.4
    lam2: float = 0.4
    n_low: float = 0.3

    # Asset grid / borrowing.
    a_min: float = -2.0
    a_max: float = 20.0
    a_lb: float = 0.0
    kappa: float = 3.0

    # Aggregate states.
    z_bar: float = 0.0
    eta_z: float = 0.1
    sigma_z: float = 0.02
    z_min: float = -0.3
    z_max: float = 0.3
    pi_min: float = -0.05
    pi_max: float = 0.05

    # NK / policy primitives.
    epsilon: float = 6.0
    psi: float = 1.0
    phi_pi: float = 1.75
    r_star: float = 0.02

    # MRS wage closure: w = chi * (N*)^nu / E[u'(c)].
    chi: float = 1.0
    nu: float = 1.0

    # Finite-agent approximation size.
    n_pop: int = 25

    # Out-of-steady-state z-slice diagnostics.
    z_low: float = -0.15
    z_high: float = 0.15

    # Numerical safeguards.
    w_floor: float = 1e-6
    c_floor: float = 1e-8

    @property
    def n_high(self) -> float:
        return 1.0 + self.lam2 / self.lam1 * (1.0 - self.n_low)


@dataclass
class NetConfig:
    width: int = 64
    layers: int = 5


@dataclass
class SamplingConfig:
    num_intervals: int = 2**4
    points_per_interval: int = 2**8
    batch_points_per_interval: int = 2**4
    al_points: int = 2**10
    # Increase low-asset auxiliary weight inside each optimizer step.
    al_batch_points: int = 2**6
    bc_points: int = 2**10
    bc_batch_points: int = 2**4
    add_points: int = 2**4
    al_region_upper_share: float = 0.1
    adaptive_burn_in_epochs: int = 2000
    # Adaptive reallocation starts after this share of total training epochs.
    adaptive_burn_in_share: float = 0.1
    adaptive_update_every_epochs: int = 10
    adaptive_skip_lower_intervals: int = 0
    adaptive_skip_upper_intervals: int = 1
    adaptive_neighbor_span: int = 2
    adaptive_neighbor_decay: float = 0.5
    adaptive_relative_eps: float = 1e-8
    # Keep a small relative-error tie-break while prioritizing absolute residual MSE.
    adaptive_relative_score_weight: float = 0.05


@dataclass
class FDBaselineConfig:
    a_size: int = 800
    delta: float = 100.0
    max_bellman_iter: int = 500
    bellman_tol: float = 1e-6
    max_wage_iter: int = 80
    wage_tol: float = 1e-6
    wage_damp: float = 0.5


@dataclass
class TrainConfig:
    seed: int = 777
    n_batch: int = 2**4
    pretrain_epochs: int = 1000
    train_epochs: int = 5000
    min_train_epochs: int = 1000
    lr_pretrain: float = 1e-3
    lr_train: float = 1e-3
    lr_scheduler: str = "plateau"
    lr_plateau_metric: str = "residual"
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 200
    lr_plateau_threshold: float = 1e-4
    lr_plateau_cooldown: int = 0
    lr_plateau_min: float = 1e-6
    clip_grad: float = 1.0
    weight_residual: float = 1e2
    weight_shape: float = 1.0
    weight_dvdz: float = 10.0
    weight_dvda: float = 1.0
    stop_residual_threshold: float | None = None
    stop_window: int = 100
    checkpoint_every_epochs: int = 1000


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    net: NetConfig = field(default_factory=NetConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    fd: FDBaselineConfig = field(default_factory=FDBaselineConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
