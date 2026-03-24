# Deep HANK Codebase and Model Reference

This document (1) maps the **entire** `deep_hank/` implementation to the spec [Deep HANK - Code Model 3](Deep HANK - Code Model 3.md) and to the economic model in [assets/main.tex](assets/main.tex), (2) records corrections to the LaTeX model (cross Itô term and interest-rate rule), and (3) gives constructive feedback on whether the implementation can serve the goal of solving the model.

---

## 1. Spec–code correspondence (Deep HANK - Code Model 3.md ↔ deep_hank/)

### 1.1 Calibration and config

| Spec (symbol / table) | Code location | Notes |
|-----------------------|---------------|--------|
| $\gamma$, $\rho$, $\underline{a}$, $a_{lb}$, $a_{max}$, $\kappa$, $n_1/n_2$, $\lambda_1/\lambda_2$ | `config.py`: `CALIBRATION` | Direct mapping; spec’s $a_{lb}$ = `a_lb`, $\underline{a}$ = `a_bar`. |
| $\epsilon$, $\psi$, $\phi_\pi$, $r^*$ | `CALIBRATION` | `phi_pi`, `r_star`. |
| OU: $\eta$, $\bar{z}$, $\sigma_z$ | `eta_z`, `z_bar`, `sigma_z` | |
| $N$, $z$/$\pi$ ranges, $\kappa_e$, $\kappa_s$ | `CALIBRATION` | |
| $n_2 = 1 + (\lambda_2/\lambda_1)(1-n_1)$ | `config.py`: `_derived()` | `n_high` overwritten from formula. |
| Steady-state $w_{ss}$, $r_{ss}$ | `TRAINING_DEFAULTS["steady_w_guess"]`, `r_ss = r_star` | Used in warm-start and Phase 1/2. |

### 1.2 Module map (six core + simulation + diagnostics)

| Spec §B module | File | Purpose |
|---------------|------|---------|
| config | `config.py` | `CALIBRATION`, `MODEL_DEFAULTS`, `TRAINING_DEFAULTS`; `default_config()` applies `_derived()` and returns full dict. |
| model | `model.py` | W-network: `input_dim`, normalization, `build_input`, `WNetwork` (Equinox MLP + softplus), `w_forward`, `w_forward_batch`. |
| economics | `economics.py` | $r$, $N^*$, $\mu_z$, $\Pi$, $\mu_\pi$, savings, penalty derivative, $\lambda(n)$, flip $n$; `split_agent_view`, `evaluate_W_for_agent`, `evaluate_all_policies`; `excess_savings`, `solve_w` (bisection + Newton). |
| sampling | `sampling.py` | `sample_batch`: $(z,\pi)$ uniform, $n$ from stationary, $a$ uniform then **centered** + clip for $\sum a^i=0$; focal $(a_i,n_i)$ with boundary oversampling; `sample_warmstart_batch` for Phase 0. |
| residual | `residual.py` | `pde_residual_single`: all terms (discount, penalty, $L_x$, $L_z$, $L_\pi$, $L_{z\pi}$, $\hat{L}_g$); `residual_batch`, `shape_loss`, `total_loss`. |
| train | `train.py` | Four-phase loop (warm-start, steady PDE, full PDE fixed $w$, outer $w$-solve + replay), checkpointing, optional curriculum/active sampling/convergence stop. |
| — | `simulation.py` | Level-1: `_simulate_one_step`, `refresh_replay_buffer`, `sample_from_replay`, `mix_batches`, `simulate_path`. |
| — | `diagnostics.py` | Plots + JSON (loss curves, residual heatmaps, wage clearing, policy slices, sim paths/hist), `write_diagnostics_manifest`, `write_interpretation_guide`. |

### 1.3 PDE residual (spec §A) vs `residual.py`

The spec gives

$$\mathcal{R}^i = (r - \rho) W^i + \mathbf{1}_{a^i \leq a_{lb}} \partial_a \psi(a^i) + L_x W + L_z W + L_\pi W + L_{z\pi} W + \hat{L}_g W,$$

with $L_x$ (individual drift + job switch), $L_z$, $L_\pi$, cross term $L_{z\pi} = -\sigma_z^2 \pi \partial_{z\pi}^2 W^i$, and $\hat{L}_g$ as empirical-measure average over $j \neq i$.

| Term                  | Spec                                                                                                | Code                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| $(r-\rho)W$ + penalty | §A                                                                                                  | `(r - cfg["rho"]) * W_i + penalty_deriv(a_i, cfg)`                                                 |
| $L_x$                 | $\partial_{a^i} W^i \cdot s^i + \lambda(n^i)(W^i_{\check{n}}-W^i)$                                  | `dW_da_i * s_i + lambda_of_n(n_i, cfg) * (W_i_flip - W_i)`                                         |
| $L_z$                 | $\partial_z W \cdot \mu_z + \tfrac{1}{2}\sigma_z^2 \partial_{zz} W$                                 | `dW_dz * mu_z_val + 0.5 * sig2 * d2W_dz2`                                                          |
| $L_\pi$               | $\partial_\pi W \cdot \mu_\pi + \tfrac{1}{2}\sigma_z^2 \pi^2 \partial_{\pi\pi} W$                   | `dW_dpi * mu_pi_val + 0.5 * sig2 * pi**2 * d2W_dpi2`                                               |
| $L_{z\pi}$            | $-\sigma_z^2 \pi \partial_{z\pi}^2 W$                                                               | `Lzpi = -sig2 * pi * d2W_dzdpi`                                                                    |
| $\hat{L}_g$           | $(N-1)^{-1}\sum_{j\neq i}[\partial_{a^j}W^i \cdot s^j + \lambda(n^j)(W^i_{n^j\to\check{n}^j}-W^i)]$ | Loop over others: `dW_da_others[k]*s_j + lambda_of_n(n_j)(W_i_flip_j - W_i)` then divide by `N-1`. |

Price-taking: wage is treated as given in derivatives (`w_fixed = stop_gradient(w)`).

### 1.4 Training phases vs spec §C

| Phase | Spec | Code |
|-------|------|------|
| 0 | Warm-start: fit $W$ to $W_0 = (w_{ss}n + r_{ss}a + \Pi_{ss}/N)^{-\gamma}$ | `_warmstart_loss`: target `c_ss**(-gamma)`; `sample_warmstart_batch`. |
| 1 | Steady-state PDE: fix $z=\bar{z}$, $\pi=0$; no $L_z,L_\pi,L_{z\pi},\hat{L}_g$ | `include_aggregate=False`, `include_distribution=False`; optional curriculum via `enable_phase1_curriculum` and `phase1_alpha_steps`. |
| 2 | Full PDE, fixed $w$ (e.g. $w_{ss}$) | `w_batch = w_ss`; full residual with aggregate and distribution; optional `enable_phase2_range_curriculum`. |
| 3 | Outer loop: solve $w$ from $\sum_i s^i(w)=0$; stock-clearing sampling; replay mix | `_solve_w_batch` (bisection + Newton); `sample_batch` with centered $a$; `refresh_replay_buffer` / `mix_batches`; optional active sampling and convergence stop. |

### 1.5 Data flow and equilibrium

- **Stock clearing**: Sampling uses centering then clip (`_project_assets_zero_supply`, `_enforce_asset_bounds`) so $\sum_i a^i = 0$ on sampled $\phi$ (spec §D, symmetric projection).
- **Flow clearing**: `solve_w` finds $w$ such that `excess_savings(model, w, ...) = 0$; used in Phase 3 and in simulation/diagnostics.
- **Warm-start target**: $c_{ss} = w_{ss} n + r_{ss} a + \Pi_{ss}/N$, $W_0 = c_{ss}^{-\gamma}$ (spec Q7).

---

## 2. Model (assets/main.tex) and implementation

### 2.1 What the LaTeX describes

- **Firms**: CES final good; intermediate producers with linear $Y_{jt}=Z_t n_{jt}$, Rotemberg cost, $dz_t = \mu_{zt}\,dt + \sigma_z dW_t^0$, NKPC for $\pi_t$ with $d\pi_t$ driven by $dY_t/Y_t$ (hence same $dW_t^0$).
- **Households**: CRRA $u(c)$, budget $da_{it} = (\tilde{r}_t a_{it} + w_t n_{it} - c_{it} + \tilde{\Pi}_t)\,dt$, Poisson employment $n_{it}$, borrowing limit $\underline{a}$.
- **Equilibrium**: Taylor rule $i_t = r^* + \phi_\pi \pi_t$, Fisher $i_t = r_t + \pi_t$ ⇒ **real rate** $r_t = r^* + (\phi_\pi - 1)\pi_t$; labor and bond market clearing; wage from bond clearing.
- **Master equation**: Recursive $V(a,n,z,\pi,g)$ with state dynamics for $a$, $n$, $z$, $\pi$, $g$; PDE in §4 (around lines 391–397).

### 2.2 Errors in main.tex (and how the code aligns with the correct model)

Two corrections are needed in `assets/main.tex` so that it matches the spec and the code.

**1) Cross Itô term (missing in main.tex)**

- **Issue**: The master equation in main.tex (lines 391–396) has TFP and inflation terms $\mu_z \partial_z V + \tfrac{1}{2}\sigma_z^2 \partial_{zz} V$ and $\mu_\pi \partial_\pi V + \tfrac{1}{2}\sigma_z^2 \pi^2 \partial_{\pi\pi} V$ but **no cross derivative** in $z$ and $\pi$.
- **Reason**: Both $dz_t$ and $d\pi_t$ depend on the same Brownian $dW_t^0$ (see $d\pi_t = \ldots - \sigma_z \pi_t dW_t^0$ and $dz_t = \ldots + \sigma_z dW_t^0$). Itô’s lemma for $V(z,\pi)$ therefore produces a cross term from the covariation of $z$ and $\pi$: $-\sigma_z^2 \pi \partial_{z\pi}^2 V$.
- **Spec/code**: The spec (§A, §G6) and `residual.py` include this term as $L_{z\pi} = -\sigma_z^2 \pi \partial_{z\pi}^2 W$. So the **implementation is correct**; the LaTeX is **incomplete**.
- **Fix in main.tex**: In the master equation (after line 395), add a new line:  
  `& \quad - \sigma_z^2 \pi \partial_{z\pi} V(a,n,z,\pi,g) \tag{Cross Itô}`  
  (before the distribution term).
  - “Cross (Itô): $-\sigma_z^2 \pi \partial_{z\pi}^2 V(a,n,z,\pi,g)$.”

**2) Interest rate in the master-equation block (wrong in main.tex)**

- **Issue**: Line 386 states $r(z,\pi,g) = r^* + \phi_{\pi} \pi$. That is the **nominal** rule; the **real** rate used in the household budget and in the PDE should satisfy the Fisher equation.
- **Correct formula**: $r(z,\pi,g) = r^* + (\phi_\pi - 1)\pi$ (as already stated correctly earlier in the same file, e.g. around line 290).
- **Code**: `economics.compute_r` uses `r_star + (phi_pi - 1) * pi`, which matches the correct real rate.
- **Fix in main.tex**: In the block where $s$, $\mu_\pi$, $\mu_g$, $r$ are defined (line 386), change `r(z,\pi,g) & = r^* + \phi_{\pi} \pi` to `r(z,\pi,g) & = r^* + (\phi_{\pi} - 1) \pi`.

With these two corrections, the written model in main.tex matches the PDE that the code implements and the spec describes.

---

## 3. Constructive feedback: can this implementation solve the model?

### 3.1 What is already aligned with solving the model

1. **PDE and economics**
   - The residual implements the full master-equation PDE (individual, aggregate, cross Itô, distribution) and soft penalty; $r$, $\mu_z$, $\mu_\pi$, $\Pi$, savings, and wage clearing match the intended economic model once main.tex is corrected as above.
2. **Equilibrium**
   - Stock clearing is enforced in sampling (centered + clip); flow clearing is enforced by solving for $w$ each batch in Phase 3 and in simulation; price-taking is respected via `stop_gradient(w)` in the residual.
3. **Training design**
   - Four-phase schedule (warm-start → steady-state PDE → full PDE with fixed $w$ → outer $w$-solve + replay) and optional curriculum/active sampling/checkpoints match the spec and support convergence.
4. **Diagnostics**
   - Dual human/machine outputs (plots + JSON, INTERPRETATION.md, diagnostics.json with pass rules) allow both inspection and automated checks.

### 3.2 Gaps and risks that could limit “solving” the model

1. **Convergence and scale**
   - Default step counts may be tight for full convergence; plans (e.g. finish-strategy) recommend a production-length run and then diagnosing. Rely on diagnostics (residual MSE, shape loss, wage-clearing error, heatmaps, sim paths) to decide if the current implementation has “solved” the model to a chosen tolerance.
2. **Phase 1 curriculum off by default**
   - Spec suggests blending warm-start with steady-state PDE in Phase 1; `enable_phase1_curriculum` is False by default. If Phase 1 is unstable or residual stays high, turning on Phase 1 curriculum (and optionally Phase 2 range curriculum) is a low-cost improvement.
3. **Permutation invariance (spec G4)**
   - Other agents are passed as a flat vector; the spec notes a possible DeepSet (permutation-invariant) upgrade for better generalization. For $N=25$ and current domains this may be acceptable, but if diagnostics show sensitivity to ordering or poor fit on off-manifold $\phi$, consider that upgrade.
4. **Validation**
   - There is no finite-difference or closed-form benchmark in the repo; adding a small FD or simplified benchmark (e.g. steady-state only) would strengthen the claim that the implementation “solves” the model.
5. **LaTeX vs code**
   - Until main.tex is corrected (cross Itô term and $r(z,\pi,g)$), the written model does not fully match the code. Fixing main.tex as in §2.2 is necessary for the documentation to “correctly correspond” to the implementation.

### 3.3 Summary

- The **codebase correctly implements** the master equation (including the cross Itô term) and the real interest rate rule; the **LaTeX model** in `assets/main.tex` should be updated as in §2.2.
- With those corrections, the implementation **can serve the goal of solving the model** in the sense of: (i) defining the same equilibrium (bond/labor clearing, Taylor rule, NKPC, OU TFP), (ii) training a NN to satisfy the master-equation residual and shape penalties, and (iii) evaluating solution quality via diagnostics.
- To confidently claim a “solved” solution: run at production scale, turn on dual diagnostics and pass rules, and optionally enable Phase 1 (and Phase 2) curriculum and add a small benchmark; then use diagnostics.json and plots to verify residual, wage clearing, and simulation behavior.

---

## 4. Plan references

- [deep_hank_finish_strategy_3a908965](.cursor/plans/deep_hank_finish_strategy_3a908965.plan.md): Full-scale run, dual diagnostics, checkpoints, optional curriculum/active sampling; autopilot pass/fail from diagnostics.json.
- [deep_hank_jax_code_d554b3b4](.cursor/plans/deep_hank_jax_code_d554b3b4.plan.md): Six-module + simulation JAX plan; all items completed.
- [dual_human-machine_diagnostics_5d35396b](.cursor/plans/dual_human-machine_diagnostics_5d35396b.plan.md): Paired human (plots + INTERPRETATION.md) and machine (JSON + pass rules) outputs; implemented in `diagnostics.py`.

---

*Document maps the full codebase to the spec and model, records LaTeX corrections, and assesses fitness for solving the model.*

<!-- amp-managed -->
