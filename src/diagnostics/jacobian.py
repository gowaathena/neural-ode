"""Evaluation-time Jacobian diagnostics.

These functions use exact Jacobians via autograd row-by-row, which is appropriate
for the low-dim systems in this study (d <= 10). They should NOT be called inside
the training loop because they re-run the autograd graph.
"""
from __future__ import annotations

from typing import Callable

import torch


def jacobian_per_sample(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    x: torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    """Full Jacobian J[b, i, j] = df_i/dx_j evaluated at x[b].

    Args:
        f: vector field, called as f(t, x).
        t: scalar or (B,) time tensor.
        x: (B, d) state tensor.

    Returns:
        (B, d, d) tensor of per-sample Jacobians.
    """
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)
    v = f(t, x)
    B, d = v.shape
    J = x.new_zeros(B, d, d)
    for i in range(d):
        grad_outputs = torch.zeros_like(v)
        grad_outputs[:, i] = 1.0
        (g,) = torch.autograd.grad(
            v, x, grad_outputs=grad_outputs,
            create_graph=create_graph, retain_graph=True,
        )
        J[:, i, :] = g
    return J


def jacobian_spectrum(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Eigenvalues of J_f(t, x) at each point in x. Returns (B, d) complex."""
    J = jacobian_per_sample(f, t, x, create_graph=False)
    return torch.linalg.eigvals(J)


def jacobian_frobenius_along_trajectory(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t: torch.Tensor,        # (T,)
    traj: torch.Tensor,     # (T, B, d)
    subsample: int = 0,
) -> torch.Tensor:
    """||J_f(t_n, x_n)||_F along a trajectory.

    If subsample > 0, only that many evenly-spaced time points are evaluated and
    the returned tensor has shape (subsample, B). Otherwise shape (T, B).
    """
    T = traj.shape[0]
    if subsample and subsample < T:
        idx = torch.linspace(0, T - 1, subsample).long()
    else:
        idx = torch.arange(T)
    norms = []
    for ti in idx.tolist():
        x = traj[ti].detach().requires_grad_(True)
        J = jacobian_per_sample(f, t[ti], x, create_graph=False)
        norms.append(torch.linalg.matrix_norm(J, ord="fro").detach())
    return torch.stack(norms)
