"""Stiffness ratio diagnostics for the learned vector field.

    s(x, t) = |lambda_max(J_f(x, t))| / |lambda_min(J_f(x, t))|
"""
from __future__ import annotations

from typing import Callable

import torch

from .jacobian import jacobian_spectrum, jacobian_per_sample


def stiffness_ratio(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    x: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-sample stiffness ratio. Returns (B,)."""
    eigvals = jacobian_spectrum(f, t, x)        # (B, d) complex
    absvals = eigvals.abs()                     # (B, d) real
    lam_max = absvals.amax(dim=-1)
    lam_min = absvals.amin(dim=-1).clamp_min(eps)
    return lam_max / lam_min


def stiffness_ratio_along_trajectory(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t: torch.Tensor,        # (T,)
    traj: torch.Tensor,     # (T, B, d)
    subsample: int = 20,
) -> torch.Tensor:
    """Stiffness ratio along a trajectory, subsampled for speed.

    Returns shape (n_subsample, B). Eigendecomposition is the bottleneck so we
    avoid evaluating it at every time step.
    """
    T = traj.shape[0]
    if subsample and subsample < T:
        idx = torch.linspace(0, T - 1, subsample).long()
    else:
        idx = torch.arange(T)
    out = []
    for ti in idx.tolist():
        x = traj[ti].detach().requires_grad_(True)
        out.append(stiffness_ratio(f, t[ti], x).detach())
    return torch.stack(out)
