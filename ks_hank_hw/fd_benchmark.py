from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .config import FDBaselineConfig, ModelConfig


def _labor_states(cfg: ModelConfig) -> np.ndarray:
    return np.array([cfg.n_low, cfg.n_high], dtype=np.float64)


class _FDHouseholdBlock:
    def __init__(self, model_cfg: ModelConfig, fd_cfg: FDBaselineConfig):
        if fd_cfg.a_size < 2:
            raise ValueError("FD benchmark requires a_size >= 2.")
        if model_cfg.a_max <= model_cfg.a_min:
            raise ValueError("FD benchmark requires a_max > a_min.")

        self.model_cfg = model_cfg
        self.fd_cfg = fd_cfg
        self.n_vals = _labor_states(model_cfg)
        self.pi_mat = np.array(
            [
                [-model_cfg.lam1, model_cfg.lam1],
                [model_cfg.lam2, -model_cfg.lam2],
            ],
            dtype=np.float64,
        )
        self.a_grid = np.linspace(model_cfg.a_min, model_cfg.a_max, fd_cfg.a_size, dtype=np.float64)
        self.a_size = self.a_grid.size
        self.z_size = self.n_vals.size
        self.state_count = self.a_size * self.z_size
        self.da = float(self.a_grid[1] - self.a_grid[0])
        self.income_transition = sparse.kron(
            self.pi_mat,
            sparse.eye(self.a_size, format="csr"),
            format="csr",
        )
        self.v = np.zeros((self.z_size, self.a_size), dtype=np.float64)
        self.g = np.zeros((self.z_size, self.a_size), dtype=np.float64)
        self.c_policy = np.zeros((self.z_size, self.a_size), dtype=np.float64)
        self.flow_matrix = self.income_transition.copy()
        self.last_bellman_dist = np.inf
        self.last_bellman_converged = False

    def _utility(self, c: np.ndarray) -> np.ndarray:
        if np.isclose(self.model_cfg.gam, 1.0):
            return np.log(c)
        return c ** (1.0 - self.model_cfg.gam) / (1.0 - self.model_cfg.gam)

    def _penalty_value(self) -> np.ndarray:
        return -0.5 * self.model_cfg.kappa * np.maximum(self.model_cfg.a_lb - self.a_grid, 0.0) ** 2

    def _flow_income(self, r: float, w: float, pi_share: float) -> np.ndarray:
        return r * self.a_grid[None, :] + w * self.n_vals[:, None] + pi_share

    def _init_value(self, r: float, w: float, pi_share: float):
        resources = np.maximum(self._flow_income(r, w, pi_share), 1e-10)
        self.v = self._utility(resources) / self.model_cfg.rho

    def solve_bellman(self, r: float, w: float, pi_share: float):
        dist = np.inf
        if not np.isfinite(np.max(self.v)):
            self._init_value(r, w, pi_share)

        for _ in range(self.fd_cfg.max_bellman_iter):
            dv = np.diff(self.v, axis=1) / self.da
            c_fd = np.maximum(dv, 1e-10) ** (-1.0 / self.model_cfg.gam)

            cash_on_hand = self._flow_income(r, w, pi_share)
            saving_forward = np.zeros_like(cash_on_hand)
            saving_backward = np.zeros_like(cash_on_hand)
            saving_forward[:, :-1] = cash_on_hand[:, :-1] - c_fd
            saving_backward[:, 1:] = cash_on_hand[:, 1:] - c_fd

            use_forward = saving_forward > 0.0
            use_backward = saving_backward < 0.0

            consumption = np.maximum(cash_on_hand, 1e-10)
            consumption[:, :-1] = np.where(use_forward[:, :-1], c_fd, consumption[:, :-1])
            consumption[:, 1:] = np.where(use_backward[:, 1:], c_fd, consumption[:, 1:])
            flow_utility = self._utility(consumption) + self._penalty_value()[None, :]

            flow_matrix = self.income_transition.copy()
            main_diag = (
                -saving_forward * use_forward / self.da + saving_backward * use_backward / self.da
            ).reshape(self.state_count)
            flow_matrix += sparse.spdiags(main_diag, 0, self.state_count, self.state_count)

            lower_diag = (-saving_backward * use_backward / self.da).reshape(self.state_count)
            flow_matrix += sparse.spdiags(lower_diag[1:], -1, self.state_count, self.state_count)

            upper_diag = (saving_forward * use_forward / self.da).reshape(self.state_count)
            flow_matrix += sparse.spdiags(
                np.concatenate(([0.0], upper_diag)),
                1,
                self.state_count,
                self.state_count,
            )

            system = (
                sparse.eye(self.state_count, format="csr") * (1.0 / self.fd_cfg.delta + self.model_cfg.rho)
                - flow_matrix
            )
            rhs = flow_utility.reshape(self.state_count) + self.v.reshape(self.state_count) / self.fd_cfg.delta
            v_new = spsolve(system, rhs).reshape(self.z_size, self.a_size)

            if not np.all(np.isfinite(v_new)):
                raise RuntimeError("FD Bellman update produced non-finite values.")

            dist = float(np.max(np.abs(v_new - self.v)))
            self.v = v_new
            self.flow_matrix = flow_matrix.tocsr()
            self.c_policy = consumption
            if dist < self.fd_cfg.bellman_tol:
                self.last_bellman_dist = dist
                self.last_bellman_converged = True
                return

        self.last_bellman_dist = dist
        self.last_bellman_converged = False

    def compute_stationary_distribution(self):
        system = self.flow_matrix.transpose().tolil()
        rhs = np.zeros(self.state_count, dtype=np.float64)
        rhs[0] = 1.0
        system[0, :] = 0.0
        system[0, 0] = 1.0
        g = spsolve(system.tocsc(), rhs).reshape(self.z_size, self.a_size)
        g = np.abs(g)
        mass = float(np.sum(g))
        if mass <= 0.0 or not np.isfinite(mass):
            raise RuntimeError("FD stationary distribution has invalid mass.")
        self.g = g / mass
        mean_a = float(np.sum(self.g * self.a_grid[None, :]))
        return mean_a

    def aggregates(self):
        marg_u = np.maximum(self.c_policy, 1e-10) ** (-self.model_cfg.gam)
        avg_marg_u = float(np.sum(self.g * marg_u))
        n_star = float(np.sum(self.g * self.n_vals[:, None]))
        return avg_marg_u, n_star


def _mrs_fixed_point(
    solver: _FDHouseholdBlock,
    z: float,
    pi: float,
    w_init: float,
):
    cfg = solver.model_cfg
    r = float(cfg.r_star + (cfg.phi_pi - 1.0) * pi)
    w = float(max(w_init, cfg.w_floor))
    diagnostics = {"w_path": [], "mrs_gap_path": [], "asset_mean_path": []}

    for _ in range(solver.fd_cfg.max_wage_iter):
        # Per-capita transfer at this wage guess.
        p_low = cfg.lam1 / (cfg.lam1 + cfg.lam2)
        p_high = 1.0 - p_low
        n_star_guess = p_low * cfg.n_low + p_high * cfg.n_high
        pi_share = float((1.0 - w / np.exp(z) + 0.5 * cfg.psi * pi**2) * np.exp(z) * n_star_guess)

        solver.solve_bellman(r=r, w=w, pi_share=pi_share)
        mean_a = solver.compute_stationary_distribution()
        avg_marg_u, n_star = solver.aggregates()
        w_mrs = float(cfg.chi * (max(n_star, 0.0) ** cfg.nu) / max(avg_marg_u, 1e-12))
        w_mrs = max(w_mrs, cfg.w_floor)

        gap = w_mrs - w
        diagnostics["w_path"].append(float(w))
        diagnostics["mrs_gap_path"].append(float(gap))
        diagnostics["asset_mean_path"].append(float(mean_a))

        w = (1.0 - solver.fd_cfg.wage_damp) * w + solver.fd_cfg.wage_damp * w_mrs
        if abs(gap) < solver.fd_cfg.wage_tol:
            break

    avg_marg_u, n_star = solver.aggregates()
    final_gap = float(cfg.chi * (max(n_star, 0.0) ** cfg.nu) / max(avg_marg_u, 1e-12) - w)
    return {
        "r": float(r),
        "w": float(w),
        "pi_share": float((1.0 - w / np.exp(z) + 0.5 * cfg.psi * pi**2) * np.exp(z) * n_star),
        "n_star": float(n_star),
        "avg_marg_u": float(avg_marg_u),
        "mrs_gap": final_gap,
        "asset_mean": float(np.sum(solver.g * solver.a_grid[None, :])),
        "diagnostics": diagnostics,
    }


def solve_fd_state(model_cfg: ModelConfig, fd_cfg: FDBaselineConfig | None, z: float, pi: float, w_init: float):
    solver = _FDHouseholdBlock(model_cfg=model_cfg, fd_cfg=fd_cfg or FDBaselineConfig())
    fp = _mrs_fixed_point(solver, z=z, pi=pi, w_init=w_init)
    return {
        "a_grid": solver.a_grid.copy(),
        "g": solver.g.copy(),
        "c_fd": solver.c_policy.copy(),
        "dv_fd": np.maximum(solver.c_policy, 1e-10) ** (-model_cfg.gam),
        "r": fp["r"],
        "w": fp["w"],
        "pi_share": fp["pi_share"],
        "n_star": fp["n_star"],
        "avg_marg_u": fp["avg_marg_u"],
        "mrs_gap": fp["mrs_gap"],
        "asset_mean": fp["asset_mean"],
        "z": float(z),
        "pi": float(pi),
        "fd_diagnostics": {
            "bellman_tol": float(solver.fd_cfg.bellman_tol),
            "max_bellman_iter": int(solver.fd_cfg.max_bellman_iter),
            "last_bellman_dist": float(solver.last_bellman_dist),
            "last_bellman_converged": bool(solver.last_bellman_converged),
            "wage_tol": float(solver.fd_cfg.wage_tol),
            "max_wage_iter": int(solver.fd_cfg.max_wage_iter),
            "wage_damp": float(solver.fd_cfg.wage_damp),
            "w_path": fp["diagnostics"]["w_path"],
            "mrs_gap_path": fp["diagnostics"]["mrs_gap_path"],
            "asset_mean_path": fp["diagnostics"]["asset_mean_path"],
        },
    }


def solve_fd_steady_anchor(model_cfg: ModelConfig, fd_cfg: FDBaselineConfig | None = None):
    return solve_fd_state(
        model_cfg=model_cfg,
        fd_cfg=fd_cfg,
        z=model_cfg.z_bar,
        pi=0.0,
        w_init=1.0,
    )


def solve_fd_z_slices(model_cfg: ModelConfig, fd_cfg: FDBaselineConfig | None = None):
    steady = solve_fd_steady_anchor(model_cfg=model_cfg, fd_cfg=fd_cfg)
    w_init = float(steady["w"])
    low = solve_fd_state(model_cfg=model_cfg, fd_cfg=fd_cfg, z=model_cfg.z_low, pi=0.0, w_init=w_init)
    high = solve_fd_state(model_cfg=model_cfg, fd_cfg=fd_cfg, z=model_cfg.z_high, pi=0.0, w_init=w_init)
    return {"z_low": low, "z_high": high}
