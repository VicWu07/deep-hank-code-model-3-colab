# Deep HANK - Code Model 3

## VS Code + Colab Runtime (primary workflow)

Use the Google Colab extension in VS Code with hosted Colab runtimes:

1. Open [colab/run_exact_chi.ipynb](colab/run_exact_chi.ipynb).
2. Select a Colab runtime from the notebook kernel picker.
3. Set `REPO_URL`, `CHI`, and run parameters in the first code cell.
4. Run all cells to mount Drive, pull code from GitHub, install deps, verify CUDA, and launch training.

The notebook uses [scripts/run_exact_chi_colab.sh](scripts/run_exact_chi_colab.sh), which wraps:

```bash
python run_hank_mrs_optimized_exact_chi.py \
  --chi <value> \
  --train-epochs <n> \
  --checkpoint-every <k> \
  --output-dir <drive_path>
```

Dependencies used in Colab are pinned in [requirements-colab.txt](requirements-colab.txt).
Full local-CLI + VS Code-Colab run instructions are in [colab/WORKFLOW.md](colab/WORKFLOW.md).

## Quick visual diagnostics

Run the quick runtime diagnostics suite (generates plots and a report):

```bash
.venv/bin/python scripts/run_diagnostics.py
```

Outputs are written to `diagnostics/YYYYMMDD_HHMMSS/` and include:
- `loss_curves.png`
- `residual_heatmap_nlow.png`, `residual_heatmap_nhigh.png`
- `w_clearing.png`
- `policy_slices.png`
- `sim_paths.png`
- `asset_hist.png`
- `report.md`
