"""Sanity tests for ground-truth ODE systems."""
from __future__ import annotations

import pytest
import torch

from src.data.dynamics import HarmonicOscillator, Robertson, VanDerPol


def _autograd_jacobian(system, t, x):
    """Compute the Jacobian of system.rhs via autograd, for comparison."""
    x = x.detach().requires_grad_(True)
    v = system.rhs(t, x)
    B, d = v.shape
    J = x.new_zeros(B, d, d)
    for i in range(d):
        grad_outputs = torch.zeros_like(v)
        grad_outputs[:, i] = 1.0
        (g,) = torch.autograd.grad(v, x, grad_outputs=grad_outputs, retain_graph=True)
        J[:, i, :] = g
    return J


@pytest.mark.parametrize("system", [
    HarmonicOscillator(),
    VanDerPol(mu=5.0),
    Robertson(),
])
def test_analytic_jacobian_matches_autograd(system):
    torch.manual_seed(0)
    x = torch.randn(8, system.state_dim) * 0.5 + 0.1
    if isinstance(system, Robertson):
        x = x.abs()  # Robertson concentrations are positive
    t = torch.tensor(0.5)
    J_analytic = system.jacobian(t, x)
    J_autograd = _autograd_jacobian(system, t, x)
    assert torch.allclose(J_analytic, J_autograd, rtol=1e-5, atol=1e-5)


def test_harmonic_jacobian_is_constant():
    system = HarmonicOscillator(omega=2.0, gamma=0.3)
    x_a = torch.randn(4, 2) * 3
    x_b = torch.randn(4, 2) * 0.01
    t = torch.tensor(0.0)
    assert torch.allclose(system.jacobian(t, x_a), system.jacobian(t, x_b))


def test_harmonic_analytic_solution():
    """Numerically integrate against the closed-form damped oscillator solution."""
    from torchdiffeq import odeint
    system = HarmonicOscillator(omega=1.0, gamma=0.1)
    x0 = torch.tensor([[1.0, 0.0]])
    t = torch.linspace(0.0, 5.0, 100)
    traj = odeint(lambda tt, xx: system.rhs(tt, xx), x0, t, method="dopri5",
                  rtol=1e-9, atol=1e-11)
    # Closed-form for underdamped oscillator x'' + 2 gamma x' + omega^2 x = 0:
    #   x(t) = exp(-gamma t) [cos(wd t) + (gamma/wd) sin(wd t)]
    #   wd = sqrt(omega^2 - gamma^2)
    gamma, omega = 0.1, 1.0
    wd = (omega ** 2 - gamma ** 2) ** 0.5
    x_true = torch.exp(-gamma * t) * (torch.cos(wd * t) + (gamma / wd) * torch.sin(wd * t))
    assert torch.allclose(traj[:, 0, 0], x_true, rtol=1e-3, atol=1e-3)
