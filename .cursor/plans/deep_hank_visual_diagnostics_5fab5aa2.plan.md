---
name: Deep HANK Visual Diagnostics
overview: "Add a quick visual diagnostics runner that trains briefly and outputs plots verifying model solution quality: loss curves, PDE residual heatmaps, wage-clearing diagnostics, policy slices, simulated paths, and asset histograms."
todos:
  - id: diag_module
    content: Implement `deep_hank/diagnostics.py` to compute grids, wage-clearing stats, and generate matplotlib figures.
    status: in_progress
  - id: diag_cli
    content: Add `scripts/run_diagnostics.py` CLI that runs quick training, calls diagnostics, and saves plots + report to diagnostics/ timestamped folder.
    status: pending
  - id: sim_path_api
    content: Expose a public `simulate_path` helper in `deep_hank/simulation.py` that returns time-series arrays for (z, pi, w, a_all, n_all).
    status: pending
  - id: smoke_doc
    content: Add a short usage note (in README or docstring) showing how to run diagnostics with `.venv/bin/python scripts/run_diagnostics.py`.
    status: pending
isProject: false
---

# Deep HANK: Visual Runtime Diagnostics (Quick)

## Goal

Create a **quick (1–2 min CPU)** “visual unit test” run that produces a set of plots you can inspect to judge whether the solver is behaving like it is solving the model.

Outputs (PNG) will include:

- Loss curves by phase
- Residual heatmaps over $(a,z)$ for $n\inn_{low},n_{high}$
- Wage-clearing diagnostics (histogram of $F(w)=\sum_i s_i(w)$ evaluated at solved wages)
- Policy slices ($W(a)$ and $c(a)$)
- Simulation paths of $(z,\pi,w)$ and cross-sectional moments
- Asset distribution histograms (overall + by employment state)

## Where to put code

- Add `[deep_hank/diagnostics.py](deep_hank/diagnostics.py)` with pure functions that:
  - run a short training (via `deep_hank.train.train(cfg)`)
  - evaluate residuals/policies on grids using `deep_hank.residual.pde_residual_single` / `total_loss`
  - solve wages and compute wage-clearing errors using `deep_hank.economics.solve_w` + `excess_savings`
  - run a path simulation returning time series (add a public helper in `deep_hank/simulation.py` if needed)
  - generate matplotlib figures
- Add a CLI entrypoint `[scripts/run_diagnostics.py](scripts/run_diagnostics.py)`:
  - creates an output folder `diagnostics/YYYYMMDD_HHMMSS/`
  - saves figures (and a short `report.md` with key scalar stats)

## Diagnostic design (what exactly gets plotted)

### 1) Training loss curves

- Use the `history` returned by `train(cfg)`.
- Plot:
  - `warmstart_mse` during Phase 0
  - `residual_mse` and `shape_loss` during Phases 1–3
- Save `loss_curves.png`.

### 2) Residual heatmaps (PDE sanity)

- Choose a representative cross-section `phi`:
  - default: take one sample from the replay buffer after the quick run (or from `sample_batch` if replay empty)
- Construct a grid:
  - `a_grid`: ~60 points in `[a_bar, a_max]`
  - `z_grid`: ~41 points in `[z_min, z_max]`
  - fix `pi=0` (quick baseline) and solve `w` for each `z` (or reuse `w_ss` as a fallback)
- For each `n` in `{n_low,n_high}` compute `|R(a,z)|` using `pde_residual_single`.
- Save:
  - `residual_heatmap_nlow.png`
  - `residual_heatmap_nhigh.png`

### 3) Wage-clearing diagnostics

- Sample ~128 states `(z,pi,phi)` (from proposal or replay buffer).
- For each, solve `w` and compute `F(w)`.
- Plot histogram and report `mean`, `p95`, `max` of `|F(w)|`.
- Save `w_clearing.png`.

### 4) Policy slices

- Fix `(z,pi)` at a few values: `(z_bar,0)`, `(z_max,0)`, `(z_min,0)`.
- For each, solve `w` given a representative `phi`.
- Plot:
  - `W(a)` and `c(a)=W(a)^{-1/gamma}` for `n_low` and `n_high` over `a_grid`.
- Save `policy_slices.png`.

### 5) Simulation paths + asset moments

- Simulate a short path (e.g. 200 steps with `dt=0.05`).
- Plot time series:
  - `(z_t, pi_t, w_t)`
  - `mean(a_t)` (should be ~0 after recenter), `std(a_t)`
  - share in `n_high`
- Save `sim_paths.png`.

### 6) Asset distribution histograms

- From the last simulated period, plot histogram of assets:
  - overall
  - conditional on `n_low` vs `n_high`
- Save `asset_hist.png`.

## Quick-run configuration defaults

In `run_diagnostics.py`, override config to keep runtime short:

- `batch_size`: 32–64
- `phase0_steps`: 50–150
- `phase1_steps`: 50–150
- `phase2_steps`: 50–150
- `phase3_steps`: 25–75 (outer-loop wage solve is the bottleneck)
- replay: `sim_path_steps` ~ 64, `sim_refresh_every` ~ 20, `sim_mix_frac` ~ 0.3

## Improvement directions (after you review plots)

- If residual heatmaps show boundary blowups: increase boundary oversampling and shape penalties.
- If wage-clearing histogram is wide: tighten solver bracket/bisection steps and/or stabilize `solve_w` with damping.
- If simulated paths drift or explode: shrink `dt_sim`, add clipping rules, or increase Phase 0/1 steps.
- If policy slices look jagged: add more training steps or a smoother activation / weight decay.

## Files to change/add

- Add: `[deep_hank/diagnostics.py](deep_hank/diagnostics.py)`
- Add: `[scripts/run_diagnostics.py](scripts/run_diagnostics.py)`
- Update (if needed): `[deep_hank/simulation.py](deep_hank/simulation.py)` to expose a `simulate_path(...)` returning arrays
- (Optional) Update: `[requirements.txt](requirements.txt)` only if matplotlib is not already present (it is)

