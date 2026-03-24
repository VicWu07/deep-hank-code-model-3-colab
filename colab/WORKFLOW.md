# VS Code + Colab Extension Workflow

## 1) Iterate locally with Codex CLI

From the project root:

```bash
codex -C "/absolute/path/to/Deep HANK - Code Model 3"
```

Or run non-interactively:

```bash
codex exec -C "/absolute/path/to/Deep HANK - Code Model 3" "Apply requested changes and run quick checks."
```

Commit/push to GitHub after local validation.

## 2) Run hosted Colab compute from VS Code

1. Open `colab/run_exact_chi.ipynb` in VS Code.
2. Select a Colab runtime in the kernel picker.
3. Fill in `REPO_URL` and run parameters.
4. For first run in a fresh runtime, use `SYNC_CODE=True` and `INSTALL_DEPS=True`.
5. For repeated chi runs in the same runtime, set `SYNC_CODE=False` and `INSTALL_DEPS=False` to skip heavy setup.
6. Run notebook cells to mount Drive, prepare repo, verify CUDA, and launch the run.

Note on repo location:

- `REPO_STORAGE="ephemeral"` (default) keeps code at `/content/workspace/repo` (faster, not in Drive).
- `REPO_STORAGE="drive"` stores code under `DRIVE_REPO_DIR` so it is visible and persistent in Google Drive.

Prebuilt one-click notebooks (run all, no parameter edits needed):

- `colab/run_exact_chi_1p000.ipynb`
- `colab/run_exact_chi_1p500.ipynb`
- `colab/run_exact_chi_2p000.ipynb`

## 3) Parallel chi jobs

Use one runtime/session per chi. Keep unique `LABEL` values or chi values so output folders/logs do not collide.

## 4) Artifact checks

Each run should produce:

- `mrs_optimized_exact_chi_summary.json`
- `loss_curves.png`
- `consumption_benchmark_4panel.png`
- `checkpoints/`
- `snapshots/`
