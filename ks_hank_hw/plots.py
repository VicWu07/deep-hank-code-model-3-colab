from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_history(history: list[dict], output_path: str | Path):
    if not history:
        return
    epochs = np.array([h["epoch"] for h in history], dtype=np.int64)
    losses = np.array([h["loss"] for h in history], dtype=np.float64)
    residuals = np.array([h["residual"] for h in history], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, losses, lw=1.5)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")

    axes[1].plot(epochs, residuals, lw=1.5, color="tab:orange")
    axes[1].set_title("Residual Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_pair(ax, a_grid, fd_c, nn_c, title: str):
    ax.plot(a_grid, fd_c[0], color="tab:blue", lw=1.8, label="FD n_low")
    ax.plot(a_grid, nn_c[0], color="tab:blue", lw=1.4, ls="--", label="NN n_low")
    ax.plot(a_grid, fd_c[1], color="tab:red", lw=1.8, label="FD n_high")
    ax.plot(a_grid, nn_c[1], color="tab:red", lw=1.4, ls="--", label="NN n_high")
    ax.set_title(title)
    ax.set_xlabel("a")
    ax.set_ylabel("c(a,n)")
    ax.grid(alpha=0.2)


def plot_consumption_benchmark_4panel(
    steady_cmp: dict,
    z_slice_cmp: dict,
    output_path: str | Path,
):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False, sharey=False)
    axes = axes.ravel()

    a_steady = steady_cmp["a_grid"]
    fd_steady = steady_cmp["fd_c"]
    nn_steady = steady_cmp["nn_c"]
    z_low = z_slice_cmp["z_low"]["z"]
    z_high = z_slice_cmp["z_high"]["z"]

    axes[0].plot(a_steady, fd_steady[0], color="tab:blue", lw=1.8, label="FD n_low")
    axes[0].plot(a_steady, nn_steady[0], color="tab:blue", lw=1.4, ls="--", label="NN n_low")
    axes[0].set_title("Steady Anchor: n_low (z=z_bar, pi=0)")
    axes[0].set_xlabel("a")
    axes[0].set_ylabel("c(a,n_low)")
    axes[0].grid(alpha=0.2)

    axes[1].plot(a_steady, fd_steady[1], color="tab:red", lw=1.8, label="FD n_high")
    axes[1].plot(a_steady, nn_steady[1], color="tab:red", lw=1.4, ls="--", label="NN n_high")
    axes[1].set_title("Steady Anchor: n_high (z=z_bar, pi=0)")
    axes[1].set_xlabel("a")
    axes[1].set_ylabel("c(a,n_high)")
    axes[1].grid(alpha=0.2)

    a_low = z_slice_cmp["z_low"]["a_grid"]
    _plot_pair(
        axes[2],
        a_low,
        z_slice_cmp["z_low"]["fd_c"],
        z_slice_cmp["z_low"]["nn_c"],
        f"Out-of-SS Low z (z={z_low:.3f}, pi=0)",
    )

    a_high = z_slice_cmp["z_high"]["a_grid"]
    _plot_pair(
        axes[3],
        a_high,
        z_slice_cmp["z_high"]["fd_c"],
        z_slice_cmp["z_high"]["nn_c"],
        f"Out-of-SS High z (z={z_high:.3f}, pi=0)",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
