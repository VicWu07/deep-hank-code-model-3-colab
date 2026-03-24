from __future__ import annotations

import torch


def _safe_inverse_input(v: torch.Tensor, eps: float = 1e-8):
    pos = torch.clamp(v, min=eps)
    neg = torch.clamp(v, max=-eps)
    return torch.where(v >= 0.0, pos, neg)


def lecture_positive_penalty(v: torch.Tensor, positive_bound: float | None):
    if positive_bound is None:
        lower_bound = 1e-8
        return torch.mean(torch.square(torch.minimum(v - lower_bound, torch.zeros_like(v))))
    safe_v = _safe_inverse_input(v)
    bound = torch.tensor([positive_bound], dtype=v.dtype, device=v.device)
    term = 1.0 / bound - 1.0 / safe_v
    return torch.mean(torch.square(torch.minimum(term, torch.zeros_like(term))))


def weighted_loss(
    residuals: torch.Tensor,
    outputs: torch.Tensor,
    dvdz: torch.Tensor,
    dvda: torch.Tensor,
    w_residual: float,
    w_shape: float,
    w_dvdz: float,
    w_dvda: float,
    positive_bound: float | None = None,
):
    loss_res = torch.mean(torch.square(residuals))
    loss_shape = lecture_positive_penalty(outputs, positive_bound=positive_bound)
    loss_dvdz = torch.square(torch.maximum(torch.mean(dvdz), torch.zeros(1, device=dvdz.device)))
    # Encourage decreasing marginal value in own assets.
    loss_dvda = torch.square(torch.maximum(torch.mean(dvda), torch.zeros(1, device=dvda.device)))
    total = w_residual * loss_res + w_shape * loss_shape + w_dvdz * loss_dvdz + w_dvda * loss_dvda
    return total, loss_res, loss_shape, loss_dvdz, loss_dvda
