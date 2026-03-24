import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ks_hank_hw.approximators import EMINN
from ks_hank_hw.config import RunConfig
from ks_hank_hw.evaluate import compare_fd_steady_anchor, compare_fd_z_slices
from ks_hank_hw.fd_benchmark import solve_fd_steady_anchor, solve_fd_z_slices
from ks_hank_hw.plots import plot_consumption_benchmark_4panel, plot_loss_history
from ks_hank_hw.residuals import HANKMRSResidual
from ks_hank_hw.samplers import AdaptiveIntervalSampler
from ks_hank_hw.trainer import Trainer
from run_hank_mrs_optimized import warmstart_marginal_target_torch


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


def main():
    cfg = RunConfig()
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    output_dir = Path(__file__).resolve().parent / "mrs_baseline_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_path = output_dir / "latest_five_training_rows.txt"
    progress_path = output_dir / "progress.txt"
    checkpoint_dir = output_dir / "checkpoints"

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
    residual = HANKMRSResidual(cfg.model)
    positive_bound = positive_bound_from_fd(cfg, fd_anchor)
    trainer = Trainer(cfg, model, sampler, residual, positive_bound=positive_bound)
    start_progress_monitor(monitor_path, progress_path, cfg.train.train_epochs, "hank-mrs")

    def pretrain_target(x, agg):
        _ = agg
        return warmstart_marginal_target_torch(x[:, [0]], cfg.model.a_lb)

    pretrain_time = trainer.pretrain(
        cfg.train.pretrain_epochs,
        pretrain_target,
        sample_fn=lambda n: sampler.pretrain_sample(n, np.asarray(fd_anchor["a_grid"]), np.asarray(fd_anchor["g"])),
    )
    hist = trainer.train(
        cfg.train.train_epochs,
        monitor_path=monitor_path,
        checkpoint_dir=checkpoint_dir,
    )

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

    with (output_dir / "mrs_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_as_serializable(summary), f, indent=2)
    with (output_dir / "fd_steady_anchor.json").open("w", encoding="utf-8") as f:
        json.dump(_as_serializable(fd_anchor), f, indent=2)
    with (output_dir / "fd_z_slices.json").open("w", encoding="utf-8") as f:
        json.dump(_as_serializable(fd_slices), f, indent=2)

    print("Saved outputs to:", output_dir)
    print("Summary:", output_dir / "mrs_summary.json")


if __name__ == "__main__":
    main()
