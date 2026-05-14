"""Ground-truth ODE systems used to generate training trajectories.

Each system exposes:
    - rhs(t, x) -> dx/dt           (callable used by the reference solver)
    - jacobian(t, x) -> J          (analytic Jacobian, when known)
    - default_init_state() -> x0
    - default_time_grid() -> torch.Tensor
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class DynamicalSystem(Protocol):
    state_dim: int

    def rhs(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor: ...
    def jacobian(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor: ...
    def default_init_state(self) -> torch.Tensor: ...
    def default_time_grid(self) -> torch.Tensor: ...


@dataclass
class HarmonicOscillator:
    """Damped harmonic oscillator (Task A — smooth, stiffness ratio ~ 1).

        x1' = x2
        x2' = -omega^2 x1 - 2 gamma x2
    """

    omega: float = 1.0
    gamma: float = 0.1
    state_dim: int = 2

    def rhs(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0], x[..., 1]
        dx1 = x2
        dx2 = -(self.omega ** 2) * x1 - 2.0 * self.gamma * x2
        return torch.stack([dx1, dx2], dim=-1)

    def jacobian(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Constant Jacobian for the linear oscillator.
        batch_shape = x.shape[:-1]
        J = x.new_zeros(*batch_shape, 2, 2)
        J[..., 0, 1] = 1.0
        J[..., 1, 0] = -(self.omega ** 2)
        J[..., 1, 1] = -2.0 * self.gamma
        return J

    def default_init_state(self) -> torch.Tensor:
        return torch.tensor([1.0, 0.0])

    def default_time_grid(self) -> torch.Tensor:
        return torch.linspace(0.0, 20.0, 200)


@dataclass
class VanDerPol:
    """Van der Pol oscillator (Task B — mildly stiff for mu >> 1).

        x1' = x2
        x2' = mu (1 - x1^2) x2 - x1
    """

    mu: float = 50.0
    state_dim: int = 2

    def rhs(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0], x[..., 1]
        dx1 = x2
        dx2 = self.mu * (1.0 - x1 ** 2) * x2 - x1
        return torch.stack([dx1, dx2], dim=-1)

    def jacobian(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0], x[..., 1]
        batch_shape = x.shape[:-1]
        J = x.new_zeros(*batch_shape, 2, 2)
        J[..., 0, 1] = 1.0
        J[..., 1, 0] = -2.0 * self.mu * x1 * x2 - 1.0
        J[..., 1, 1] = self.mu * (1.0 - x1 ** 2)
        return J

    def default_init_state(self) -> torch.Tensor:
        return torch.tensor([2.0, 0.0])

    def default_time_grid(self) -> torch.Tensor:
        # Scale time grid with mu so we always see at least one slow-manifold cycle.
        T_end = max(20.0, 2.0 * self.mu)
        return torch.linspace(0.0, T_end, 500)


@dataclass
class Robertson:
    """Robertson kinetics (Task C — strongly stiff, optional).

        y1' = -0.04 y1 + 1e4 y2 y3
        y2' =  0.04 y1 - 1e4 y2 y3 - 3e7 y2^2
        y3' =  3e7 y2^2
    """

    state_dim: int = 3

    def rhs(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y1, y2, y3 = x[..., 0], x[..., 1], x[..., 2]
        d1 = -0.04 * y1 + 1.0e4 * y2 * y3
        d2 = 0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 ** 2
        d3 = 3.0e7 * y2 ** 2
        return torch.stack([d1, d2, d3], dim=-1)

    def jacobian(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y1, y2, y3 = x[..., 0], x[..., 1], x[..., 2]
        batch_shape = x.shape[:-1]
        J = x.new_zeros(*batch_shape, 3, 3)
        J[..., 0, 0] = -0.04
        J[..., 0, 1] = 1.0e4 * y3
        J[..., 0, 2] = 1.0e4 * y2
        J[..., 1, 0] = 0.04
        J[..., 1, 1] = -1.0e4 * y3 - 6.0e7 * y2
        J[..., 1, 2] = -1.0e4 * y2
        J[..., 2, 1] = 6.0e7 * y2
        return J

    def default_init_state(self) -> torch.Tensor:
        return torch.tensor([1.0, 0.0, 0.0])

    def default_time_grid(self) -> torch.Tensor:
        # Robertson is conventionally integrated on a log time grid.
        return torch.cat([torch.zeros(1), torch.logspace(-5, 5, 99)])


def make_system(task: str, kwargs: dict | None = None) -> DynamicalSystem:
    """Factory used by the trainer to instantiate a system by name."""
    kwargs = kwargs or {}
    if task == "harmonic":
        return HarmonicOscillator(**kwargs)
    if task == "vanderpol":
        return VanDerPol(**kwargs)
    if task == "robertson":
        return Robertson(**kwargs)
    raise ValueError(f"Unknown task: {task!r}")
