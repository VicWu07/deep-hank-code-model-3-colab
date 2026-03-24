import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from ks_hank_hw.approximators import EMINN
from ks_hank_hw.config import RunConfig
from ks_hank_hw.evaluate import compare_fd_steady_anchor, compare_fd_z_slices
from ks_hank_hw.fd_benchmark import solve_fd_steady_anchor, solve_fd_z_slices
from ks_hank_hw.plots import plot_consumption_benchmark_4panel, plot_loss_history
from ks_hank_hw.residuals import HANKMRSOptimizedExactItoResidual
from ks_hank_hw.samplers import AdaptiveIntervalSampler
from ks_hank_hw.trainer import Trainer
from run_hank_mrs_optimized import (
    _as_serializable,
    _write_snapshot_artifacts,
    positive_bound_from_fd,
    start_progress_monitor,
    summarize_history,
    warmstart_marginal_target_torch,
)


def _chi_tag(chi: float) -> str:
    return f"{chi:.3f}".replace("-", "m").replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser(description="Exact optimized HANK run with custom chi.")
    parser.add_argument("--chi", type=float, required=True, help="MRS wage scale parameter.")
    parser.add_argument("--train-epochs", type=int, default=5000, help="Training epochs.")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Checkpoint cadence.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to mrs_optimized_exact_chi_<chi>_outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = RunConfig()
    cfg.model.chi = float(args.chi)
    cfg.train.train_epochs = int(args.train_epochs)
    cfg.train.checkpoint_every_epochs = int(args.checkpoint_every)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    base_dir = Path(__file__).resolve().parent
    if args.output_dir is None:
        output_dir = base_dir / f"mrs_optimized_exact_chi_{_chi_tag(cfg.model.chi)}_outputs"
    else:
        output_dir = Path(args.output_dir)
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
        "FD anchor loaded (chi={:.3f}): r={:.4f}, w={:.4f}, mrs_gap={:.3e}, asset_mean={:.3e}".format(
            cfg.model.chi,
            fd_anchor["r"],
            fd_anchor["w"],
            fd_anchor["mrs_gap"],
            fd_anchor["asset_mean"],
        )
    )

    model = EMINN(
        input_dim=2 * cfg.model.n_pop + 2,
        width=cfg.net.width,
        layers=cfg.net.layers,
    )
    sampler = AdaptiveIntervalSampler(cfg.model, cfg.sampling)
    residual = HANKMRSOptimizedExactItoResidual(cfg.model)
    positive_bound = positive_bound_from_fd(cfg, fd_anchor)
    trainer = Trainer(cfg, model, sampler, residual, positive_bound=positive_bound)
    start_progress_monitor(
        monitor_path,
        progress_path,
        cfg.train.train_epochs,
        f"hank-optimized-exact-chi={cfg.model.chi:.3f}",
    )

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
    trainer.model.load_state_dict(final_state)
    _write_snapshot_artifacts(
        model=trainer.model,
        cfg=cfg,
        sampler=sampler,
        device=trainer.device,
        snapshot_dir=snapshot_dir,
        stage_tag="final",
        train_hist=hist,
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
            "snapshot_dir": str(snapshot_dir),
            "pretrain_checkpoint": str(checkpoint_dir / "pretrain.pt"),
            "final_snapshot_tag": "final",
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

    with (output_dir / "mrs_optimized_exact_chi_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_as_serializable(summary), f, indent=2)
    with (output_dir / "fd_steady_anchor.json").open("w", encoding="utf-8") as f:
        json.dump(_as_serializable(fd_anchor), f, indent=2)
    with (output_dir / "fd_z_slices.json").open("w", encoding="utf-8") as f:
        json.dump(_as_serializable(fd_slices), f, indent=2)

    print("Saved outputs to:", output_dir)
    print("Summary:", output_dir / "mrs_optimized_exact_chi_summary.json")


if __name__ == "__main__":
    main()
