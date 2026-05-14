"""Tests for the Jacobian penalty (exact + Hutchinson) and diagnostics."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.regularizers import jacobian_frobenius_exact, jacobian_frobenius_hutchinson
from src.diagnostics import jacobian_per_sample


class LinearVF(nn.Module):
    """Vector field v(t, x) = A x. Jacobian is the constant A."""

    def __init__(self, A: torch.Tensor):
        super().__init__()
        self.A = nn.Parameter(A.clone(), requires_grad=False)

    def forward(self, t, x):
        return x @ self.A.T


def test_exact_jacobian_matches_underlying_matrix():
    torch.manual_seed(0)
    A = torch.randn(3, 3)
    f = LinearVF(A)
    x = torch.randn(5, 3)
    J = jacobian_per_sample(f, torch.tensor(0.0), x)            # (5, 3, 3)
    A_expanded = A.unsqueeze(0).expand_as(J)
    assert torch.allclose(J, A_expanded, atol=1e-6)


def test_exact_frobenius_matches_manual():
    torch.manual_seed(1)
    A = torch.randn(3, 3)
    f = LinearVF(A)
    x = torch.randn(7, 3)
    expected = (A ** 2).sum().expand(x.shape[0])
    got = jacobian_frobenius_exact(f, torch.tensor(0.0), x, create_graph=False)
    assert torch.allclose(got, expected, atol=1e-5)


def test_hutchinson_is_unbiased():
    """Average of many Hutchinson samples converges to the true Frobenius norm^2."""
    torch.manual_seed(2)
    A = torch.randn(4, 4)
    f = LinearVF(A)
    x = torch.randn(64, 4)
    true_F2 = (A ** 2).sum().item()
    samples = []
    for _ in range(200):
        s = jacobian_frobenius_hutchinson(f, torch.tensor(0.0), x, K=1, create_graph=False)
        samples.append(s.mean().item())
    mc_mean = sum(samples) / len(samples)
    # ~5% tolerance is comfortable for 200 batches of 64 samples
    assert abs(mc_mean - true_F2) / true_F2 < 0.05


def test_hutchinson_gradient_flows_to_parameters():
    """The penalty must be differentiable through f's parameters."""
    torch.manual_seed(3)
    A = torch.randn(3, 3, requires_grad=True)
    f = LinearVF(A.detach())
    f.A.requires_grad_(True)
    x = torch.randn(4, 3)
    pen = jacobian_frobenius_hutchinson(f, torch.tensor(0.0), x, K=2, create_graph=True).mean()
    pen.backward()
    assert f.A.grad is not None
    assert torch.linalg.norm(f.A.grad) > 0
