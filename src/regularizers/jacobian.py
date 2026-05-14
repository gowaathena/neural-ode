"""Jacobian Frobenius-norm penalty (exact and Hutchinson estimator).

See docs/theory/02_jacobian_regularization.md and 04_hutchinson_estimator.md.
"""
from __future__ import annotations

from typing import Callable, Literal

import torch


def _sample_noise(shape, device, dtype, kind: Literal["gaussian", "rademacher"]) -> torch.Tensor:
    if kind == "gaussian":
        return torch.randn(shape, device=device, dtype=dtype)
    if kind == "rademacher":
        return torch.randint(0, 2, shape, device=device).to(dtype) * 2 - 1
    raise ValueError(f"Unknown noise kind: {kind!r}")


def jacobian_frobenius_exact(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    x: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """Exact ||J_f(t, x)||_F^2 via d backward passes (one per output coordinate).

    Cost: O(d * cost(f)). Suitable for diagnostics on low-dim systems (d <= 10).

    Args:
        f: vector field, called as f(t, x).
        t: scalar or (B,) time tensor.
        x: (B, d) state tensor, must require grad for autograd to work.
        create_graph: True if the penalty will be backpropagated through
            (training time); False for diagnostic-only use.

    Returns:
        (B,) tensor of per-sample squared Frobenius norms.
    """
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)
    v = f(t, x)                                       # (B, d)
    B, d = v.shape
    sq = x.new_zeros(B)
    for i in range(d):
        grad_outputs = torch.zeros_like(v)
        grad_outputs[:, i] = 1.0
        (g,) = torch.autograd.grad(
            v, x, grad_outputs=grad_outputs,
            create_graph=create_graph, retain_graph=True,
        )
        sq = sq + (g ** 2).sum(dim=-1)
    return sq


def jacobian_frobenius_hutchinson(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    x: torch.Tensor,
    K: int = 1,
    noise: Literal["gaussian", "rademacher"] = "rademacher",
    create_graph: bool = True,
) -> torch.Tensor:
    """Unbiased estimate of ||J_f(t, x)||_F^2 via K vector-Jacobian products.

    Cost: O(K * cost(f)) — independent of state dimension.
    """
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)
    v = f(t, x)                                       # (B, d)
    total = x.new_zeros(x.shape[0])
    for _ in range(K):
        eps = _sample_noise(x.shape, x.device, x.dtype, noise)
        (g,) = torch.autograd.grad(
            v, x, grad_outputs=eps,
            create_graph=create_graph, retain_graph=True,
        )
        # g[b] = eps[b]^T J[b]; ||g||^2 = eps^T J J^T eps  -->  E[.] = ||J||_F^2.
        total = total + (g ** 2).sum(dim=-1)
    return total / K
