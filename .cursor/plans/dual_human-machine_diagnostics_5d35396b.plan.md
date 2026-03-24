---
name: Dual human-machine diagnostics
overview: Refactor diagnostics so every check produces (1) a human-facing plot with a clear interpretation guide and (2) a machine-readable artifact with a fixed schema and explicit consumption rules for debugging and CI. No plot-reading by machines.
todos: []
isProject: false
---

# Dual Human / Machine Diagnostics

## Philosophy

Each diagnostic produces **two paired outputs**:

1. **For humans:** Plot(s) plus a short **interpretation guide** (what to look for, what good/bad means, implications).
2. **For machines:** **Structured data only** (JSON/NDJSON) with a fixed schema. Machines never parse images; they read numbers and apply explicit rules (thresholds, pass/fail, comparisons).

Implications:

- Humans get visual intuition and narrative; machines get scalars, arrays, and booleans.
- CI or scripts can gate on machine artifacts (e.g. "fail if residual_mse_final > 1e-3").
- Debugging can be done by inspecting JSON + config without opening PNGs.

---

## Output layout (per run)

```
diagnostics/YYYYMMDD_HHMMSS/
├── report.md                    # Human: summary + config (existing)
├── INTERPRETATION.md             # Human: how to read each plot (new)
├── diagnostics.json              # Machine: manifest + optional global pass (new)
├── loss_curves.png               # Human
├── loss_curves.json              # Machine
├── residual_heatmap_nlow.png     # Human
├── residual_heatmap_nhigh.png    # Human
├── residual_heatmaps.json        # Machine
├── w_clearing.png               # Human
├── wage_clearing.json           # Machine
├── policy_slices.png            # Human
├── policy_slices.json           # Machine
├── sim_paths.png                # Human
├── asset_hist.png               # Human
└── sim_paths.json               # Machine
```

---

## 1. Loss curves

### Human: plot + interpretation

- **Plot:** `loss_curves.png` — step vs. warmstart_mse, residual_mse, shape_loss (log scale).
- **How to interpret:**
  - **Good:** residual_mse and shape_loss trend **down** over steps; warmstart_mse decreases (warmstart target is being met).
  - **Bad:** Curves flat or **rising** → training not helping; try more steps, smaller LR, or check data.
  - **Implication:** This is the first check: if loss curves look bad, other diagnostics are less meaningful.

### Machine: artifact + how to read

- **File:** `loss_curves.json`
- **Schema (conceptual):**

```json
{
  "steps": [0, 10, 20, ...],
  "warmstart_mse": [...],
  "residual_mse": [...],
  "shape_loss": [...],
  "final": {
    "residual_mse": 1.2e-4,
    "shape_loss": 3.4e-5
  }
}
```

- **How machine should read it:**
  - Load JSON; read `final.residual_mse`, `final.shape_loss`.
  - **Pass rule (example):** `residual_mse_final < threshold_residual` and `shape_loss_final < threshold_shape` (e.g. 1e-3 and 1e-2).
  - **Optional:** Check last K steps are monotonically decreasing for residual_mse to assert "still improving".
  - **Debug use:** Compare `final` across runs; log values for regression tracking.

---

## 2. Residual heatmaps

### Human: plot + interpretation

- **Plots:** `residual_heatmap_nlow.png`, `residual_heatmap_nhigh.png` — $(a, z)$ heatmaps of $\log_{10}|R|$.
- **How to interpret:**
  - **Good:** Most of the domain is **cool** (e.g. $\log_{10}|R| \le -2$ or $-3$); small residual everywhere.
  - **Bad:** **Hot spots** (yellow/red) → PDE error large there (e.g. near boundaries, or at certain $z$). One panel much worse than the other → labor-state-specific issue.
  - **Implication:** Hot regions are where the HJB is violated; focus training or grid refinement there, or accept as known weakness.

### Machine: artifact + how to read

- **File:** `residual_heatmaps.json`
- **Schema (conceptual):**

```json
{
  "a_grid": [...],
  "z_grid": [...],
  "n_low": { "log10_abs_R": [[...], ...] },
  "n_high": { "log10_abs_R": [[...], ...] },
  "summary": {
    "n_low_max_log10_abs_R": -1.8,
    "n_low_mean_log10_abs_R": -3.2,
    "n_high_max_log10_abs_R": -2.1,
    "n_high_mean_log10_abs_R": -3.5
  }
}
```

- **How machine should read it:**
  - Use **summary** only (no need to parse 2D arrays unless debugging).
  - **Pass rule (example):** `max(n_low_max_log10_abs_R, n_high_max_log10_abs_R) < threshold`, e.g. $-1.5$ (so $|R| < \sim 0.03$).
  - **Debug use:** If fail, compare `n_low` vs `n_high` max to see which employment state is worse; log summary for trend over runs.

---

## 3. Wage clearing

### Human: plot + interpretation

- **Plot:** `w_clearing.png` — histogram of $|F(w)|$ over sampled states.
- **How to interpret:**
  - **Good:** Most mass **near 0**; wage-clearing error small for most states.
  - **Bad:** **Long right tail** or bimodal peak away from 0 → equilibrium condition fails for many states; policy may be off or solver tolerance too loose.
  - **Implication:** Large errors mean the "market-clearing wage" used in training doesn't actually clear; can bias dynamics in simulation.

### Machine: artifact + how to read

- **File:** `wage_clearing.json`
- **Schema (conceptual):**

```json
{
  "n_samples": 24,
  "abs_errors": [1.2e-5, ...],
  "mean_abs": 2.3e-5,
  "p95_abs": 8.1e-5,
  "max_abs": 1.2e-4
}
```

- **How machine should read it:**
  - Read **p95_abs** and **max_abs** (align with current `w_clear_p95_abs`, `w_clear_max_abs` in stats).
  - **Pass rule (example):** `p95_abs < 1e-4` and `max_abs < 5e-4` (or configurable thresholds).
  - **Debug use:** If fail, inspect `abs_errors` distribution (e.g. percentile list) to see if one tail or many states bad.

---

## 4. Policy slices

### Human: plot + interpretation

- **Plot:** `policy_slices.png` — $W(a)$ and $c(a)$ vs $a$ for a few $(z, n)$.
- **How to interpret:**
  - **Good:** $W(a)$ and $c(a)$ **increase with $a$**; no obvious kinks or non-monotonicity; levels differ by $z$ and $n$ as expected (e.g. higher $z$ → higher $W$, $c$).
  - **Bad:** **Decreasing** segments or flat over large ranges → violates theory or indicates poor fit; spikes → numerical issues.
  - **Implication:** Monotonicity is a basic sanity check; violations suggest shape penalty or training issue.

### Machine: artifact + how to read

- **File:** `policy_slices.json`
- **Schema (conceptual):**

```json
{
  "a_grid": [...],
  "W": {
    "z_bar": { "n_low": [...], "n_high": [...] },
    "z_min": { ... },
    "z_max": { ... }
  },
  "c": { ... },
  "summary": {
    "min_dW_da_over_all_slices": 0.02,
    "min_dc_da_over_all_slices": 0.01,
    "monotonicity_violations_W": 0,
    "monotonicity_violations_c": 0
  }
}
```

- **How machine should read it:**
  - Use **summary** only. Compute `min_dW_da` and `min_dc_da` (e.g. finite differences on saved curves); count violations (e.g. negative differences).
  - **Pass rule (example):** `monotonicity_violations_W == 0` and `monotonicity_violations_c == 0`, or `min_dW_da > 0` and `min_dc_da > 0`.
  - **Debug use:** If fail, which $(z,n)$ slice has min slope or violations can be derived from stored curves without reading the plot.

---

## 5. Simulated paths and histograms

### Human: plot + interpretation

- **Plots:** `sim_paths.png` (z, pi, w, mean(a), std(a), share n_high vs time), `asset_hist.png` (final asset distribution by employment).
- **How to interpret:**
  - **Good:** Paths **bounded**, no explosion; mean(a) stays near 0 (or drift is small); share_high in $[0,1]$; asset hist has plausible spread, possibly bimodal by employment.
  - **Bad:** **Exploding** z, w, or std(a); mean(a) drifting strongly; NaNs; asset hist with mass at boundaries only → simulation or equilibrium unstable.
  - **Implication:** Validates that the trained model produces sensible joint dynamics; failure here suggests residual or wage-clearing errors are affecting time evolution.

### Machine: artifact + how to read

- **File:** `sim_paths.json`
- **Schema (conceptual):**

```json
{
  "steps": 28,
  "time": [0, 0.05, ...],
  "z": [...],
  "pi": [...],
  "w": [...],
  "mean_a": [...],
  "std_a": [...],
  "share_high": [...],
  "final": {
    "mean_a_abs": 0.02,
    "std_a": 0.15,
    "share_high": 0.52
  },
  "summary": {
    "has_nan_or_inf": false,
    "share_high_in_valid_range": true,
    "max_abs_mean_a": 0.02
  }
}
```

- **How machine should read it:**
  - Read **final** and **summary**. No image parsing.
  - **Pass rule (example):** `has_nan_or_inf == false`, `share_high` in $[0.01, 0.99]$, `max_abs_mean_a < 0.5` (or configurable).
  - **Debug use:** Compare `final` and path arrays across runs; log `max_abs_mean_a`, `final.std_a` for stability tracking.

---

## 6. Manifest and interpretation doc (new)

### `diagnostics.json` (machine)

- Single entry point for scripts. Suggested schema:

```json
{
  "run_id": "20260224_095529",
  "config_snapshot": { "batch_size": 24, ... },
  "artifacts": {
    "loss_curves": "loss_curves.json",
    "residual_heatmaps": "residual_heatmaps.json",
    "wage_clearing": "wage_clearing.json",
    "policy_slices": "policy_slices.json",
    "sim_paths": "sim_paths.json"
  },
  "summary": {
    "residual_mse_final": 1.2e-4,
    "residual_heatmap_max_log10_R": -1.8,
    "w_clear_p95_abs": 8e-5,
    "policy_slices_monotonic": true,
    "sim_has_nan_or_inf": false
  },
  "pass": true
}
```

- **How machine should read it:** Load `diagnostics.json`; if `pass` is true, run passed all internal rules. Optionally re-run checks from linked artifacts for custom thresholds. `summary` duplicates key scalars so a single file can drive CI without opening other JSONs.

### `INTERPRETATION.md` (human)

- One short section per diagnostic (loss curves, residual heatmaps, wage clearing, policy slices, sim paths).
- For each: (1) which plot(s), (2) what "good" looks like, (3) what "bad" looks like, (4) implication in one sentence. No machine parsing of this file; it is documentation for you.

---

## Implementation notes

- **Code changes:** In [deep_hank/diagnostics.py](deep_hank/diagnostics.py): each plotting function also writes a JSON with the same schema as above; add `write_interpretation_md()` and `write_diagnostics_manifest()`; aggregate `summary` and optional `pass` from per-diag rules (thresholds can live in config, e.g. `diag_pass_residual_mse_max`, `diag_pass_w_clear_p95_max`).
- **Config:** Add optional keys like `diag_pass_residual_mse_max`, `diag_pass_w_clear_p95_max`, `diag_pass_residual_log10_R_max`, `diag_pass_sim_max_abs_mean_a`, etc. Defaults can be loose so existing runs don't start failing; CI can override.
- **Backward compatibility:** Keep existing `report.md` and PNGs; add new JSONs and INTERPRETATION.md so current workflows remain valid.

This gives you a clear split: **you** read plots and INTERPRETATION.md; **machines** read only JSON and apply the rules above (or custom ones you encode in config/script).