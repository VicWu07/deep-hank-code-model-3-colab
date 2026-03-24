import torch

from .config import ModelConfig


class HANKEconomy:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.n_vals = [cfg.n_low, cfg.n_high]

    def u(self, c: torch.Tensor) -> torch.Tensor:
        g = self.cfg.gam
        return c.pow(1.0 - g) / (1.0 - g)

    def du(self, c: torch.Tensor) -> torch.Tensor:
        return c.pow(-self.cfg.gam)

    def labor_from_indicator(self, y_indicator: torch.Tensor) -> torch.Tensor:
        n_low = torch.tensor(self.cfg.n_low, dtype=y_indicator.dtype, device=y_indicator.device)
        n_high = torch.tensor(self.cfg.n_high, dtype=y_indicator.dtype, device=y_indicator.device)
        return n_low + (n_high - n_low) * y_indicator

    def lambda_from_indicator(self, y_indicator: torch.Tensor) -> torch.Tensor:
        lam1 = torch.tensor(self.cfg.lam1, dtype=y_indicator.dtype, device=y_indicator.device)
        lam2 = torch.tensor(self.cfg.lam2, dtype=y_indicator.dtype, device=y_indicator.device)
        return lam1 * (1.0 - y_indicator) + lam2 * y_indicator

    def real_rate(self, pi: torch.Tensor) -> torch.Tensor:
        return self.cfg.r_star + (self.cfg.phi_pi - 1.0) * pi

    def mu_z(self, z: torch.Tensor) -> torch.Tensor:
        return self.cfg.eta_z * (self.cfg.z_bar - z)

    def mu_pi(self, z: torch.Tensor, pi: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        mu_z = self.mu_z(z)
        sig2 = self.cfg.sigma_z**2
        term_nkpc = (self.cfg.epsilon / self.cfg.psi) * (
            (self.cfg.epsilon - 1.0) / self.cfg.epsilon - w / torch.exp(z)
        )
        return (
            self.cfg.r_star * pi
            + (self.cfg.phi_pi - 1.0) * pi**2
            + term_nkpc
            - (mu_z - 0.5 * sig2) * pi
        )

    def mrs_wage(self, n_star: torch.Tensor, avg_marg_u: torch.Tensor) -> torch.Tensor:
        num = self.cfg.chi * torch.clamp(n_star, min=0.0).pow(self.cfg.nu)
        den = torch.clamp(avg_marg_u, min=1e-12)
        return torch.clamp(num / den, min=self.cfg.w_floor)

    def transfer(self, z: torch.Tensor, pi: torch.Tensor, w: torch.Tensor, n_star: torch.Tensor) -> torch.Tensor:
        # Consistent per-capita transfer convention.
        return (1.0 - w / torch.exp(z)) * torch.exp(z) * n_star

    def penalty_grad(self, a_i: torch.Tensor) -> torch.Tensor:
        return self.cfg.kappa * torch.relu(torch.tensor(self.cfg.a_lb, device=a_i.device) - a_i)
