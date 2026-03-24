import torch

from .config import ModelConfig
from .economics import HANKEconomy
from .state import split_state


class HANKMRSResidual:
    """
    Deep-HANK residual in PS4 style, with explicit price taking:
    prices are computed from others' states (phi^{-i}), not own state.
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.eco = HANKEconomy(cfg)

    @staticmethod
    def _grad(y: torch.Tensor, x: torch.Tensor):
        return torch.autograd.grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
        )[0]

    def _compute_prices(self, model, x: torch.Tensor, agg: torch.Tensor):
        """
        Price-taking prices based on others only:
        r(z,pi,phi^{-i}), w(z,pi,phi^{-i}), Pi(z,pi,w,phi^{-i}).
        """
        a, y = split_state(x, self.cfg)
        if self.cfg.n_pop <= 1:
            raise NotImplementedError("Price-taking finite-agent residual needs n_pop > 1.")

        z = agg[:, 0]
        pi = agg[:, 1]
        r = self.eco.real_rate(pi)

        n_other = self.eco.labor_from_indicator(y[:, 1:])
        n_star = torch.mean(n_other, dim=1)

        marg_u_vals = []
        with torch.no_grad():
            for j in range(1, self.cfg.n_pop):
                xj_switch = x.clone()
                ind = (0, j, self.cfg.n_pop, self.cfg.n_pop + j)
                indx = (j, 0, self.cfg.n_pop + j, self.cfg.n_pop)
                xj_switch[:, ind] = x[:, indx].clone()

                wj = torch.clamp(model(xj_switch, agg)[:, 0], min=self.cfg.c_floor)
                cj = wj ** (-1.0 / self.cfg.gam)
                marg_u_vals.append(self.eco.du(cj))

        avg_marg_u = torch.mean(torch.stack(marg_u_vals, dim=1), dim=1)
        w = self.eco.mrs_wage(n_star, avg_marg_u)
        pi_share = self.eco.transfer(z, pi, w, n_star)
        return r.detach(), w.detach(), pi_share.detach()

    def _aggregate_terms(self, model, x: torch.Tensor, agg: torch.Tensor, w: torch.Tensor):
        agg_req = agg.clone().requires_grad_(True)
        v = model(x, agg_req)
        dV_dagg = self._grad(v, agg_req)
        dV_dz = dV_dagg[:, 0:1]
        dV_dpi = dV_dagg[:, 1:2]

        d2_from_z = self._grad(dV_dz, agg_req)
        d2_from_pi = self._grad(dV_dpi, agg_req)
        d2V_dzz = d2_from_z[:, 0:1]
        d2V_dzpi = d2_from_z[:, 1:2]
        d2V_dpipi = d2_from_pi[:, 1:2]

        z = agg_req[:, 0:1]
        pi = agg_req[:, 1:2]
        w_col = w.detach().unsqueeze(1)
        mu_z = self.eco.mu_z(z)
        mu_pi = self.eco.mu_pi(z, pi, w_col)
        sig2 = self.cfg.sigma_z**2
        l_agg = (
            dV_dz * mu_z
            + 0.5 * sig2 * d2V_dzz
            + dV_dpi * mu_pi
            + 0.5 * sig2 * pi**2 * d2V_dpipi
            - sig2 * pi * d2V_dzpi
        )
        return -l_agg

    def __call__(self, model, x: torch.Tensor, agg: torch.Tensor):
        a = x[:, : self.cfg.n_pop].clone().requires_grad_(True)
        agg_req = agg[:, :2].clone()

        def model_a(a_state: torch.Tensor):
            x_temp = x.clone()
            x_temp[:, : self.cfg.n_pop] = a_state
            return model(x_temp, agg_req)

        v = model_a(a)
        dV_da = self._grad(v, a)
        diffeq = torch.zeros(x.shape[0], dtype=v.dtype, device=v.device)

        r, w, pi_share = self._compute_prices(
            model,
            x,
            agg_req,
        )

        ai = x[:, 0]
        yi = x[:, self.cfg.n_pop]
        ni = self.eco.labor_from_indicator(yi)
        dV_dai = dV_da[:, 0]
        wi = torch.clamp(v[:, 0], min=self.cfg.c_floor)
        ci = wi ** (-1.0 / self.cfg.gam)
        si = r * ai + w * ni - ci + pi_share

        x_switch_y = x.clone()
        x_switch_y[:, self.cfg.n_pop] = 1.0 - x[:, self.cfg.n_pop]
        v_switch_y = model(x_switch_y, agg_req)
        lam_i = self.eco.lambda_from_indicator(yi)
        diffeq += self.cfg.rho * v[:, 0] - dV_dai * si - r * v[:, 0] - lam_i * (v_switch_y - v)[:, 0]
        diffeq -= self.eco.penalty_grad(ai)

        for j in range(1, self.cfg.n_pop):
            xj_switch = x.clone()
            ind = (0, j, self.cfg.n_pop, self.cfg.n_pop + j)
            indx = (j, 0, self.cfg.n_pop + j, self.cfg.n_pop)
            xj_switch[:, ind] = x[:, indx].clone()

            rj, wj, pij = self._compute_prices(
                model,
                xj_switch,
                agg_req,
            )

            aj = xj_switch[:, 0]
            yj = xj_switch[:, self.cfg.n_pop]
            nj = self.eco.labor_from_indicator(yj)
            vj = model(xj_switch, agg_req)[:, 0]
            cj = torch.clamp(vj, min=self.cfg.c_floor) ** (-1.0 / self.cfg.gam)
            sj = rj * aj + wj * nj - cj + pij

            x_switch_yj = x.clone()
            x_switch_yj[:, self.cfg.n_pop + j] = 1.0 - x[:, self.cfg.n_pop + j]
            v_switch_yj = model(x_switch_yj, agg_req)
            lam_j = self.eco.lambda_from_indicator(yj)
            diffeq += -dV_da[:, j] * sj - lam_j * (v_switch_yj - v)[:, 0]

        agg_term = self._aggregate_terms(
            model,
            x,
            agg_req,
            w,
        )
        return diffeq.unsqueeze(1) + agg_term


class HANKMRSItoResidual(HANKMRSResidual):
    """
    Ito trick variant of Deep-HANK residual. Uses one-dimensional 
    epsilon-directional derivative to compute continuous drifts and Ito terms.
    """

    def _ito_trick(self, model, x, agg, mu_a, mu_z, mu_pi):
        sig_z = self.cfg.sigma_z
        z_val = agg[:, 0:1]
        pi_val = agg[:, 1:2]
        sig_z_vec = torch.full_like(z_val, sig_z)
        sig_pi_vec = -sig_z * pi_val

        def F(eps):
            x_eps = x.clone()
            x_eps[:, : self.cfg.n_pop] = x_eps[:, : self.cfg.n_pop] + 0.5 * eps * eps * mu_a
            
            agg_eps = agg.clone()
            agg_eps[:, 0:1] = agg[:, 0:1] + eps * sig_z_vec / (2.0**0.5) + 0.5 * eps * eps * mu_z.unsqueeze(1)
            agg_eps[:, 1:2] = agg[:, 1:2] + eps * sig_pi_vec / (2.0**0.5) + 0.5 * eps * eps * mu_pi.unsqueeze(1)
            
            return model(x_eps, agg_eps)

        eps = torch.zeros_like(agg[:, 0:1], requires_grad=True)
        F_eps = F(eps)
        dF = self._grad(F_eps, eps)
        d2F = self._grad(dF, eps)
        # Match HANKMRSResidual sign convention: subtract PDE operator
        return -d2F

    def __call__(self, model, x: torch.Tensor, agg: torch.Tensor):
        agg_req = agg[:, :2].clone()
        v = model(x, agg_req)
        diffeq = torch.zeros(x.shape[0], dtype=v.dtype, device=v.device)

        r, w, pi_share = self._compute_prices(
            model,
            x,
            agg_req,
        )

        ai = x[:, 0]
        yi = x[:, self.cfg.n_pop]
        ni = self.eco.labor_from_indicator(yi)
        wi = torch.clamp(v[:, 0], min=self.cfg.c_floor)
        ci = wi ** (-1.0 / self.cfg.gam)
        si = r * ai + w * ni - ci + pi_share

        x_switch_y = x.clone()
        x_switch_y[:, self.cfg.n_pop] = 1.0 - x[:, self.cfg.n_pop]
        v_switch_y = model(x_switch_y, agg_req)
        lam_i = self.eco.lambda_from_indicator(yi)
        
        # Note: dV_dai * si is handled by Ito trick
        diffeq += self.cfg.rho * v[:, 0] - r * v[:, 0] - lam_i * (v_switch_y - v)[:, 0]
        diffeq -= self.eco.penalty_grad(ai)

        mu_a = torch.zeros((x.shape[0], self.cfg.n_pop), dtype=v.dtype, device=v.device)
        mu_a[:, 0] = si

        for j in range(1, self.cfg.n_pop):
            xj_switch = x.clone()
            ind = (0, j, self.cfg.n_pop, self.cfg.n_pop + j)
            indx = (j, 0, self.cfg.n_pop + j, self.cfg.n_pop)
            xj_switch[:, ind] = x[:, indx].clone()

            rj, wj, pij = self._compute_prices(
                model,
                xj_switch,
                agg_req,
            )

            aj = xj_switch[:, 0]
            yj = xj_switch[:, self.cfg.n_pop]
            nj = self.eco.labor_from_indicator(yj)
            vj = model(xj_switch, agg_req)[:, 0]
            cj = torch.clamp(vj, min=self.cfg.c_floor) ** (-1.0 / self.cfg.gam)
            sj = rj * aj + wj * nj - cj + pij
            
            mu_a[:, j] = sj

            x_switch_yj = x.clone()
            x_switch_yj[:, self.cfg.n_pop + j] = 1.0 - x[:, self.cfg.n_pop + j]
            v_switch_yj = model(x_switch_yj, agg_req)
            lam_j = self.eco.lambda_from_indicator(yj)
            
            # Note: dV_da[:, j] * sj is handled by Ito trick
            diffeq += -lam_j * (v_switch_yj - v)[:, 0]

        z = agg_req[:, 0]
        pi = agg_req[:, 1]
        w_col = w.detach()
        mu_z = self.eco.mu_z(z)
        mu_pi = self.eco.mu_pi(z, pi, w_col)

        ito = self._ito_trick(model, x, agg_req, mu_a, mu_z, mu_pi)
        return diffeq.unsqueeze(1) + ito


class HANKMRSOptimizedItoResidual(HANKMRSItoResidual):
    """
    O(N) batched vectorized evaluation of the PDE residual eliminating nested swapping.
    Computes all prices and values in two massive batched NN calls instead of N loops.
    """

    def __call__(self, model, x: torch.Tensor, agg: torch.Tensor):
        agg_req = agg[:, :2].clone()
        B = x.shape[0]
        N = self.cfg.n_pop

        # 1. Tile states for all focal swaps
        x_all = x.unsqueeze(1).repeat(1, N, 1)
        for k in range(N):
            if k == 0:
                continue
            x_all[:, k, 0], x_all[:, k, k] = x[:, k].clone(), x[:, 0].clone()
            x_all[:, k, N], x_all[:, k, N + k] = x[:, N + k].clone(), x[:, N].clone()
            
        x_all_flat = x_all.view(B * N, 2 * N)
        agg_all_flat = agg_req.unsqueeze(1).repeat(1, N, 1).view(B * N, 2)

        # 2. Evaluate all agents simultaneously
        v_all_flat = model(x_all_flat, agg_all_flat)[:, 0]
        v_all = v_all_flat.view(B, N)
        c_all = torch.clamp(v_all, min=self.cfg.c_floor) ** (-1.0 / self.cfg.gam)

        # 3. Evaluate jump states.
        # Focal-i own employment jump: W_i(..., n_i') - W_i(..., n_i).
        x_i_jump = x.clone()
        x_i_jump[:, N] = 1.0 - x_i_jump[:, N]
        v_i_jump = model(x_i_jump, agg_req)[:, 0]
        jump_i = v_i_jump - v_all[:, 0]

        # Other agents' employment jumps must be evaluated as focal-i value changes:
        # W_i(..., n_j') - W_i(..., n_j), not W_j(..., n_j') - W_j(..., n_j).
        if N > 1:
            x_others_jump = x.unsqueeze(1).repeat(1, N - 1, 1)
            for j in range(1, N):
                x_others_jump[:, j - 1, N + j] = 1.0 - x_others_jump[:, j - 1, N + j]
            x_others_jump_flat = x_others_jump.view(B * (N - 1), 2 * N)
            agg_others_jump_flat = agg_req.unsqueeze(1).repeat(1, N - 1, 1).view(B * (N - 1), 2)
            v_others_jump_flat = model(x_others_jump_flat, agg_others_jump_flat)[:, 0]
            jump_others = v_others_jump_flat.view(B, N - 1) - v_all[:, 0:1]
        else:
            jump_others = None

        # 4. Global prices computation
        marg_u_all = self.eco.du(c_all)
        sum_marg_u = marg_u_all.sum(dim=1, keepdim=True)
        avg_marg_u_excl = (sum_marg_u - marg_u_all) / (N - 1)

        y_all = x[:, N:]
        n_all = self.eco.labor_from_indicator(y_all)
        sum_n = n_all.sum(dim=1, keepdim=True)
        n_star_excl = (sum_n - n_all) / (N - 1)

        w_all = self.eco.mrs_wage(n_star_excl, avg_marg_u_excl)
        r_all = self.eco.real_rate(agg_req[:, 1]).unsqueeze(1).repeat(1, N)

        z_rep = agg_req[:, 0].unsqueeze(1).repeat(1, N)
        pi_rep = agg_req[:, 1].unsqueeze(1).repeat(1, N)
        pi_share_all = self.eco.transfer(z_rep, pi_rep, w_all, n_star_excl)

        # 5. Assemble drifts and diffs
        a_all = x[:, :N]
        s_all = r_all * a_all + w_all * n_all - c_all + pi_share_all
        lam_all = self.eco.lambda_from_indicator(y_all)

        v = v_all[:, 0]
        diffeq = self.cfg.rho * v - r_all[:, 0] * v - lam_all[:, 0] * jump_i
        diffeq -= self.eco.penalty_grad(a_all[:, 0])

        if N > 1:
            diffeq += -torch.sum(lam_all[:, 1:] * jump_others, dim=1)

        # 6. Ito term via directional derivative
        mu_a = s_all
        mu_z = self.eco.mu_z(agg_req[:, 0])
        mu_pi = self.eco.mu_pi(agg_req[:, 0], agg_req[:, 1], w_all[:, 0])

        ito = self._ito_trick(model, x, agg_req, mu_a, mu_z, mu_pi)
        
        return diffeq.unsqueeze(1) + ito


class HANKMRSOptimizedExactItoResidual(HANKMRSItoResidual):
    """
    Batched/vectorized Ito residual with exact leave-one-out price construction.
    Keeps the optimized batched focal and jump evaluation, but avoids the
    sum-all-minus-own shortcut for avg marginal utility.
    """

    @staticmethod
    def _forward_no_grad_chunked(model, x_flat: torch.Tensor, agg_flat: torch.Tensor, chunk_size: int = 32768):
        outputs = []
        total = int(x_flat.shape[0])
        with torch.no_grad():
            for lo in range(0, total, chunk_size):
                hi = min(lo + chunk_size, total)
                outputs.append(model(x_flat[lo:hi], agg_flat[lo:hi])[:, 0])
        return torch.cat(outputs, dim=0)

    def __call__(self, model, x: torch.Tensor, agg: torch.Tensor):
        agg_req = agg[:, :2].clone()
        B = x.shape[0]
        N = self.cfg.n_pop
        if N <= 1:
            raise NotImplementedError("Price-taking finite-agent residual needs n_pop > 1.")

        # 1) Tile states for all focal swaps: slice k puts agent-k into focal slot 0.
        x_all = x.unsqueeze(1).repeat(1, N, 1)
        for k in range(N):
            if k == 0:
                continue
            x_all[:, k, 0], x_all[:, k, k] = x[:, k].clone(), x[:, 0].clone()
            x_all[:, k, N], x_all[:, k, N + k] = x[:, N + k].clone(), x[:, N].clone()

        x_all_flat = x_all.view(B * N, 2 * N)
        agg_all_flat = agg_req.unsqueeze(1).repeat(1, N, 1).view(B * N, 2)

        # 2) Evaluate focal values with gradients (needed for drift through c_j).
        v_all_flat = model(x_all_flat, agg_all_flat)[:, 0]
        v_all = v_all_flat.view(B, N)
        c_all = torch.clamp(v_all, min=self.cfg.c_floor) ** (-1.0 / self.cfg.gam)

        # 3) Evaluate jump states for focal-i value changes.
        x_i_jump = x.clone()
        x_i_jump[:, N] = 1.0 - x_i_jump[:, N]
        v_i_jump = model(x_i_jump, agg_req)[:, 0]
        jump_i = v_i_jump - v_all[:, 0]

        x_others_jump = x.unsqueeze(1).repeat(1, N - 1, 1)
        for j in range(1, N):
            x_others_jump[:, j - 1, N + j] = 1.0 - x_others_jump[:, j - 1, N + j]
        x_others_jump_flat = x_others_jump.view(B * (N - 1), 2 * N)
        agg_others_jump_flat = agg_req.unsqueeze(1).repeat(1, N - 1, 1).view(B * (N - 1), 2)
        v_others_jump_flat = model(x_others_jump_flat, agg_others_jump_flat)[:, 0]
        jump_others = v_others_jump_flat.view(B, N - 1) - v_all[:, 0:1]

        # 4) Exact leave-one-out prices:
        # For each focal-k state x_all[:, k, :], evaluate all swaps (0 <-> m), m=1..N-1.
        x_excl = x_all.unsqueeze(2).repeat(1, 1, N - 1, 1)
        for m in range(1, N):
            a0 = x_excl[:, :, m - 1, 0].clone()
            am = x_excl[:, :, m - 1, m].clone()
            x_excl[:, :, m - 1, 0] = am
            x_excl[:, :, m - 1, m] = a0

            y0 = x_excl[:, :, m - 1, N].clone()
            ym = x_excl[:, :, m - 1, N + m].clone()
            x_excl[:, :, m - 1, N] = ym
            x_excl[:, :, m - 1, N + m] = y0

        x_excl_flat = x_excl.reshape(B * N * (N - 1), 2 * N)
        agg_excl_flat = agg_req.unsqueeze(1).unsqueeze(2).repeat(1, N, N - 1, 1).reshape(B * N * (N - 1), 2)
        v_excl_flat = self._forward_no_grad_chunked(model, x_excl_flat, agg_excl_flat)
        c_excl = torch.clamp(v_excl_flat.view(B, N, N - 1), min=self.cfg.c_floor) ** (-1.0 / self.cfg.gam)
        avg_marg_u_excl = self.eco.du(c_excl).mean(dim=2)

        y_other = x_all[:, :, N + 1 : 2 * N]
        n_star_excl = self.eco.labor_from_indicator(y_other).mean(dim=2)

        w_all = self.eco.mrs_wage(n_star_excl, avg_marg_u_excl).detach()
        r_all = self.eco.real_rate(agg_req[:, 1]).unsqueeze(1).repeat(1, N).detach()

        z_rep = agg_req[:, 0].unsqueeze(1).repeat(1, N)
        pi_rep = agg_req[:, 1].unsqueeze(1).repeat(1, N)
        pi_share_all = self.eco.transfer(z_rep, pi_rep, w_all, n_star_excl).detach()

        # 5) Assemble drifts and discrete jump terms.
        a_all = x[:, :N]
        y_all = x[:, N:]
        n_all = self.eco.labor_from_indicator(y_all)
        s_all = r_all * a_all + w_all * n_all - c_all + pi_share_all
        lam_all = self.eco.lambda_from_indicator(y_all)

        v = v_all[:, 0]
        diffeq = self.cfg.rho * v - r_all[:, 0] * v - lam_all[:, 0] * jump_i
        diffeq -= self.eco.penalty_grad(a_all[:, 0])
        diffeq += -torch.sum(lam_all[:, 1:] * jump_others, dim=1)

        # 6) Ito term via directional derivative.
        mu_a = s_all
        mu_z = self.eco.mu_z(agg_req[:, 0])
        mu_pi = self.eco.mu_pi(agg_req[:, 0], agg_req[:, 1], w_all[:, 0])
        ito = self._ito_trick(model, x, agg_req, mu_a, mu_z, mu_pi)
        return diffeq.unsqueeze(1) + ito
