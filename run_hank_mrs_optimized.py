import json
import threading
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ks_hank_hw.approximators import EMINN
from ks_hank_hw.config import RunConfig
from ks_hank_hw.evaluate import compare_fd_steady_anchor, compare_fd_z_slices
from ks_hank_hw.fd_benchmark import solve_fd_steady_anchor, solve_fd_z_slices
from ks_hank_hw.plots import plot_consumption_benchmark_4panel, plot_loss_history
from ks_hank_hw.residuals import HANKMRSOptimizedItoResidual
from ks_hank_hw.samplers import AdaptiveIntervalSampler
from ks_hank_hw.trainer import Trainer


def _format_eta(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def start_progress_monitor(monitor_path: Path, progress_path: Path, total_epochs: int, label: str):
    def worker():
        start_ts = time.time()
        last_line = None
        while True:
            if not monitor_path.exists():
                time.sleep(1.0)
                continue
            try:
                if monitor_path.stat().st_mtime < start_ts:
                    time.sleep(0.5)
                    continue
            except FileNotFoundError:
                time.sleep(0.5)
                continue
            lines = monitor_path.read_text(encoding="utf-8").strip().splitlines()
            if len(lines) < 3:
                time.sleep(1.0)
                continue
            line = lines[-1].strip()
            if not line or line == last_line:
                time.sleep(1.0)
                continue
            last_line = line
            parts = line.split(",")
            if len(parts) < 8:
                time.sleep(1.0)
                continue
            try:
                epoch = int(parts[0])
                loss = float(parts[1])
                residual = float(parts[2])
                elapsed = float(parts[7])
            except ValueError:
                time.sleep(1.0)
                continue
            percent = 100.0 * epoch / max(total_epochs, 1)
            eta_sec = elapsed * (total_epochs - epoch) / max(epoch, 1)
            msg = f"[{label}] {percent:6.2f}% | ETA {_format_eta(eta_sec)} | loss {loss:.3e} | res {residual:.3e}"
            print(msg, flush=True)
            progress_path.write_text(msg + "\n", encoding="utf-8")
            if epoch >= total_epochs:
                break
            time.sleep(0.5)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def summarize_history(hist: list[dict]):
    losses = np.array([h["loss"] for h in hist], dtype=np.float64)
    residual = np.array([h["residual"] for h in hist], dtype=np.float64)
    epoch_times = np.array([h["epoch_time_sec"] for h in hist], dtype=np.float64)

    last_n = min(100, len(hist))
    losses_tail = losses[-last_n:]
    res_tail = residual[-last_n:]
    return {
        "terminal_loss": float(losses[-1]),
        "best_loss": float(np.min(losses)),
        "last_window": last_n,
        "loss_mean_last_window": float(np.mean(losses_tail)),
        "loss_std_last_window": float(np.std(losses_tail)),
        "residual_mean_last_window": float(np.mean(res_tail)),
        "residual_std_last_window": float(np.std(res_tail)),
        "mean_epoch_time_sec": float(np.mean(epoch_times)),
        "total_train_time_sec": float(hist[-1]["elapsed_sec"]),
        "epochs_completed": int(hist[-1]["epoch"]),
        "stop_reason": hist[-1].get("stop_reason", "unknown"),
        "stop_epoch": int(hist[-1].get("stop_epoch", hist[-1]["epoch"])),
    }


def positive_bound_from_fd(cfg: RunConfig, fd_anchor: dict):
    y_high = cfg.model.n_high
    return float((10.0 * (fd_anchor["w"] * y_high + fd_anchor["r"] * cfg.model.a_max)) ** (-cfg.model.gam))


def _as_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _as_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_as_serializable(v) for v in obj]
    return obj


def _recover_value_from_marginal(a_grid: np.ndarray, w_vals: np.ndarray):
    # Recover V(a) up to an additive constant via V_a = W.
    v_vals = np.zeros_like(w_vals, dtype=np.float64)
    if a_grid.size <= 1:
        return v_vals
    da = np.diff(a_grid)
    increments = 0.5 * (w_vals[1:] + w_vals[:-1]) * da
    v_vals[1:] = np.cumsum(increments)
    return v_vals


def warmstart_marginal_target_numpy(a_grid: np.ndarray, a_lb: float):
    # Bounded positive warm start for W=V_a to avoid explosive c = W^(-1/gamma).
    centered = np.clip(a_grid - float(a_lb), -12.0, 12.0)
    return 1.0 + np.exp(-0.35 * centered)


def warmstart_marginal_target_torch(a_own: torch.Tensor, a_lb: float):
    centered = torch.clamp(a_own - float(a_lb), min=-12.0, max=12.0)
    return 1.0 + torch.exp(-0.35 * centered)


def _predict_marginal(model, x: np.ndarray, agg: np.ndarray, device: torch.device):
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    at = torch.tensor(agg, dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(xt, at).detach().cpu().numpy().reshape(-1)


def _build_exact_clearing_sweep_states(
    cfg: RunConfig,
    sampler: AdaptiveIntervalSampler,
    a_grid: np.ndarray,
    own_employment: float,
    z_val: float,
    pi_val: float,
):
    n_points = int(a_grid.size)
    n_pop = int(cfg.model.n_pop)
    n_other = max(n_pop - 1, 0)

    x = np.zeros((n_points, 2 * n_pop), dtype=np.float32)
    x[:, 0] = a_grid.astype(np.float32)

    if n_other > 0:
        # Fixed other-employment panel for smooth slices.
        p_low = cfg.model.lam1 / (cfg.model.lam1 + cfg.model.lam2)
        n_low = int(round(p_low * n_other))
        n_low = min(max(n_low, 0), n_other)
        y_other = np.concatenate(
            [np.zeros(n_low, dtype=np.float32), np.ones(n_other - n_low, dtype=np.float32)]
        )
        x[:, n_pop + 1 : 2 * n_pop] = y_other[None, :]

        # Use bounded-sum projection so each row exactly clears.
        base_raw = np.linspace(cfg.model.a_min, cfg.model.a_max, n_other, dtype=np.float64)
        target_total = float(cfg.model.n_pop) * float(sampler._per_capita_asset_supply())
        a_other = np.zeros((n_points, n_other), dtype=np.float64)
        for i in range(n_points):
            target_other_sum = target_total - float(a_grid[i])
            a_other[i, :] = sampler._project_row_to_bounded_sum(
                base_raw,
                target_other_sum,
                float(cfg.model.a_min),
                float(cfg.model.a_max),
            )
        x[:, 1:n_pop] = a_other.astype(np.float32)

    x[:, n_pop] = float(own_employment)
    agg = np.zeros((n_points, 2), dtype=np.float32)
    agg[:, 0] = float(z_val)
    agg[:, 1] = float(pi_val)
    return x, agg


def _plot_pretrain_loss(pretrain_hist: list[dict], output_path: Path):
    if not pretrain_hist:
        return
    epochs = np.array([item["epoch"] for item in pretrain_hist], dtype=np.int64)
    losses = np.array([item["loss"] for item in pretrain_hist], dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(epochs, losses, lw=1.5)
    ax.set_xlabel("Pretrain Epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title("Pretraining Loss")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_snapshot_panels(
    a_grid: np.ndarray,
    state_specs: list[tuple[str, float, float]],
    stage_tag: str,
    metric_name: str,
    pred_curves: dict[tuple[str, str], np.ndarray],
    target_curve: np.ndarray,
    output_path: Path,
    y_axis_label: str,
):
    emp_specs = [("n_low", 0.0), ("n_high", 1.0)]
    fig, axes = plt.subplots(len(state_specs), len(emp_specs), figsize=(12, 11), sharex=True)
    if len(state_specs) == 1:
        axes = np.array([axes])
    for row, (state_name, z_val, pi_val) in enumerate(state_specs):
        for col, (emp_name, _) in enumerate(emp_specs):
            ax = axes[row, col]
            curve = pred_curves[(state_name, emp_name)]
            ax.plot(a_grid, curve, lw=1.8, label="NN")
            ax.plot(a_grid, target_curve, lw=1.2, ls="--", label="Pretrain Target")
            ax.set_title(f"{state_name} (z={z_val:.3f}, pi={pi_val:.3f}), {emp_name}")
            ax.set_ylabel(y_axis_label)
            ax.grid(alpha=0.2)
            if row == len(state_specs) - 1:
                ax.set_xlabel("a")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"{metric_name} Snapshot: {stage_tag}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_snapshot_artifacts(
    *,
    model,
    cfg: RunConfig,
    sampler: AdaptiveIntervalSampler,
    device: torch.device,
    snapshot_dir: Path,
    stage_tag: str,
    pretrain_hist=None,
    train_hist=None,
):
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    a_grid = np.linspace(cfg.model.a_min, cfg.model.a_max, 400, dtype=np.float64)
    target_w = warmstart_marginal_target_numpy(a_grid, cfg.model.a_lb)
    target_c = np.maximum(target_w, 1e-8) ** (-1.0 / cfg.model.gam)
    target_v = _recover_value_from_marginal(a_grid, target_w)
    state_specs = [
        ("steady", float(cfg.model.z_bar), 0.0),
        ("z_low", float(cfg.model.z_low), 0.0),
        ("z_high", float(cfg.model.z_high), 0.0),
    ]
    emp_specs = [("n_low", 0.0), ("n_high", 1.0)]

    pred_w = {}
    pred_c = {}
    pred_v = {}
    model.eval()
    for state_name, z_val, pi_val in state_specs:
        for emp_name, emp_val in emp_specs:
            x, agg = _build_exact_clearing_sweep_states(cfg, sampler, a_grid, emp_val, z_val, pi_val)
            w_pred = _predict_marginal(model, x, agg, device)
            c_pred = np.maximum(w_pred, 1e-8) ** (-1.0 / cfg.model.gam)
            v_pred = _recover_value_from_marginal(a_grid, w_pred)
            pred_w[(state_name, emp_name)] = w_pred
            pred_c[(state_name, emp_name)] = c_pred
            pred_v[(state_name, emp_name)] = v_pred

    _plot_snapshot_panels(
        a_grid=a_grid,
        state_specs=state_specs,
        stage_tag=stage_tag,
        metric_name="Marginal Value W(a)",
        pred_curves=pred_w,
        target_curve=target_w,
        output_path=snapshot_dir / f"{stage_tag}_marginal_value.png",
        y_axis_label="W(a)",
    )
    _plot_snapshot_panels(
        a_grid=a_grid,
        state_specs=state_specs,
        stage_tag=stage_tag,
        metric_name="Consumption c(a)",
        pred_curves=pred_c,
        target_curve=target_c,
        output_path=snapshot_dir / f"{stage_tag}_consumption.png",
        y_axis_label="c(a)",
    )
    _plot_snapshot_panels(
        a_grid=a_grid,
        state_specs=state_specs,
        stage_tag=stage_tag,
        metric_name="Recovered Value V(a) (up to constant)",
        pred_curves=pred_v,
        target_curve=target_v,
        output_path=snapshot_dir / f"{stage_tag}_value.png",
        y_axis_label="V(a)",
    )

    if pretrain_hist is not None:
        _plot_pretrain_loss(pretrain_hist, snapshot_dir / f"{stage_tag}_loss_curve.png")
    if train_hist is not None:
        plot_loss_history(train_hist, snapshot_dir / f"{stage_tag}_loss_curve.png")


def main():
    cfg = RunConfig()
    cfg.train.train_epochs = 10000
    cfg.train.checkpoint_every_epochs = 500
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    output_dir = Path(__file__).resolve().parent / "mrs_optimized_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_path = output_dir / "latest_five_training_rows.txt"
    progress_path = output_dir / "progress.txt"
    checkpoint_dir = output_dir / "checkpoints"
    snapshot_dir = output_dir / "snapshots"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    fd_anchor = solve_fd_steady_anchor(cfg.model, cfg.fd)
    fd_slices = solve_fd_z_slices(cfg.model, cfg.fd)
    print(
        "FD anchor loaded: r={:.4f}, w={:.4f}, mrs_gap={:.3e}, asset_mean={:.3e}".format(
            fd_anchor["r"], fd_anchor["w"], fd_anchor["mrs_gap"], fd_anchor["asset_mean"]
        )
    )

    model = EMINN(
        input_dim=2 * cfg.model.n_pop + 2,
        width=cfg.net.width,
        layers=cfg.net.layers,
    )
    sampler = AdaptiveIntervalSampler(cfg.model, cfg.sampling)
    residual = HANKMRSOptimizedItoResidual(cfg.model)
    positive_bound = positive_bound_from_fd(cfg, fd_anchor)
    trainer = Trainer(cfg, model, sampler, residual, positive_bound=positive_bound)
    start_progress_monitor(monitor_path, progress_path, cfg.train.train_epochs, "hank-optimized")

    def pretrain_target(x, agg):
        _ = agg
        return warmstart_marginal_target_torch(x[:, [0]], cfg.model.a_lb)

    pretrain_time, pretrain_hist = trainer.pretrain(
        cfg.train.pretrain_epochs,
        pretrain_target,
        sample_fn=lambda n: sampler.pretrain_sample(n, np.asarray(fd_anchor["a_grid"]), np.asarray(fd_anchor["g"])),
        return_history=True,
    )
    pretrain_state = deepcopy(trainer.model.state_dict())
    torch.save(
        {
            "stage": "pretrain",
            "epoch": int(cfg.train.pretrain_epochs),
            "config": Trainer.dump_config(cfg),
            "model_state_dict": pretrain_state,
        },
        checkpoint_dir / "pretrain.pt",
    )
    _write_snapshot_artifacts(
        model=trainer.model,
        cfg=cfg,
        sampler=sampler,
        device=trainer.device,
        snapshot_dir=snapshot_dir,
        stage_tag="pretrain",
        pretrain_hist=pretrain_hist,
    )
    hist = trainer.train(
        cfg.train.train_epochs,
        monitor_path=monitor_path,
        checkpoint_dir=checkpoint_dir,
    )
    final_state = deepcopy(trainer.model.state_dict())
    generated_snapshot_epochs = []
    if hist:
        max_epoch = int(hist[-1]["epoch"])
        for epoch in range(500, max_epoch + 1, 500):
            ckpt_path = checkpoint_dir / f"epoch_{epoch:06d}.pt"
            if not ckpt_path.exists():
                continue
            payload = torch.load(ckpt_path, map_location=trainer.device)
            trainer.model.load_state_dict(payload["model_state_dict"])
            hist_prefix = [item for item in hist if int(item["epoch"]) <= epoch]
            _write_snapshot_artifacts(
                model=trainer.model,
                cfg=cfg,
                sampler=sampler,
                device=trainer.device,
                snapshot_dir=snapshot_dir,
                stage_tag=f"epoch_{epoch:06d}",
                train_hist=hist_prefix,
            )
            generated_snapshot_epochs.append(int(epoch))
    trainer.model.load_state_dict(final_state)

    steady_cmp = compare_fd_steady_anchor(
        model=trainer.model,
        fd_anchor=fd_anchor,
        cfg=cfg.model,
        device=trainer.device,
        seed=cfg.train.seed,
    )
    z_slice_cmp = compare_fd_z_slices(
        model=trainer.model,
        fd_slices=fd_slices,
        cfg=cfg.model,
        device=trainer.device,
        seed=cfg.train.seed,
    )

    train_summary = summarize_history(hist)
    summary = {
        "config": Trainer.dump_config(cfg),
        "timing": {"pretrain_time_sec": float(pretrain_time)},
        "training": train_summary,
        "fd_steady_anchor": {
            "r": float(fd_anchor["r"]),
            "w": float(fd_anchor["w"]),
            "z": float(fd_anchor["z"]),
            "pi": float(fd_anchor["pi"]),
            "mrs_gap": float(fd_anchor["mrs_gap"]),
            "asset_mean": float(fd_anchor["asset_mean"]),
        },
        "fd_z_slices": {
            "z_low": {
                "z": float(fd_slices["z_low"]["z"]),
                "w": float(fd_slices["z_low"]["w"]),
                "mrs_gap": float(fd_slices["z_low"]["mrs_gap"]),
                "asset_mean": float(fd_slices["z_low"]["asset_mean"]),
            },
            "z_high": {
                "z": float(fd_slices["z_high"]["z"]),
                "w": float(fd_slices["z_high"]["w"]),
                "mrs_gap": float(fd_slices["z_high"]["mrs_gap"]),
                "asset_mean": float(fd_slices["z_high"]["asset_mean"]),
            },
        },
        "benchmark_errors": {
            "steady_anchor": steady_cmp["error_stats"],
            "z_low": z_slice_cmp["z_low"]["error_stats"],
            "z_high": z_slice_cmp["z_high"]["error_stats"],
        },
        "artifacts": {
            "monitor_path": str(monitor_path),
            "checkpoint_dir": str(checkpoint_dir),
            "snapshot_dir": str(snapshot_dir),
            "pretrain_checkpoint": str(checkpoint_dir / "pretrain.pt"),
            "snapshot_epochs": generated_snapshot_epochs,
            "loss_plot": str(output_dir / "loss_curves.png"),
            "benchmark_plot": str(output_dir / "consumption_benchmark_4panel.png"),
        },
    }

    plot_loss_history(hist, output_dir / "loss_curves.png")
    plot_consumption_benchmark_4panel(
        steady_cmp=steady_cmp,
        z_slice_cmp=z_slice_cmp,
        output_path=output_dir / "consumption_benchmark_4panel.png",
    )

    with (output_dir / "mrs_optimized_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_as_serializable(summary), f, indent=2)
    with (output_dir / "fd_steady_anchor.json").open("w", encoding="utf-8") as f:
        json.dump(_as_serializable(fd_anchor), f, indent=2)
    with (output_dir / "fd_z_slices.json").open("w", encoding="utf-8") as f:
        json.dump(_as_serializable(fd_slices), f, indent=2)

    print("Saved outputs to:", output_dir)
    print("Summary:", output_dir / "mrs_optimized_summary.json")


if __name__ == "__main__":
    main()
