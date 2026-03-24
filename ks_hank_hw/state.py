from dataclasses import dataclass

import torch

from .config import ModelConfig


@dataclass
class KSBatch:
    x: torch.Tensor
    z: torch.Tensor


def split_state(x: torch.Tensor, cfg: ModelConfig):
    n = cfg.n_pop
    a = x[:, :n]
    y = x[:, n : 2 * n]
    return a, y


def make_switched_batch(batch: KSBatch, cfg: ModelConfig, j: int) -> KSBatch:
    """Re-index state so agent j becomes index 0."""
    xj = batch.x.clone()
    n = cfg.n_pop
    a = xj[:, :n].clone()
    y = xj[:, n : 2 * n].clone()
    a0 = a[:, 0].clone()
    y0 = y[:, 0].clone()
    a[:, 0] = a[:, j]
    y[:, 0] = y[:, j]
    a[:, j] = a0
    y[:, j] = y0
    xj[:, :n] = a
    xj[:, n : 2 * n] = y
    return KSBatch(x=xj, z=batch.z)

