"""Data generation and PyTorch Dataset wrapper for ground-truth trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchdiffeq import odeint

from .dynamics import DynamicalSystem


@dataclass
class TrajectoryBatch:
    x0: torch.Tensor          # (B, d)
    t: torch.Tensor           # (T,)
    target: torch.Tensor      # (T, B, d)


def generate_trajectories(
    system: DynamicalSystem,
    initial_conditions: torch.Tensor,   # (B, d)
    t: torch.Tensor,                    # (T,)
    method: str = "dopri5",
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> torch.Tensor:
    """Integrate `system.rhs` with a high-accuracy reference solver.

    Returns target tensor of shape (T, B, d). Tolerances are tighter than the
    training-time solver tolerances so the ground truth is effectively exact.
    Use method='implicit_adams' for stiff systems (Robertson).
    """
    def f(tt: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return system.rhs(tt, x)

    with torch.no_grad():
        if method == "dopri5":
            traj = odeint(f, initial_conditions, t, method="dopri5", rtol=rtol, atol=atol)
        elif method == "implicit_adams":
            traj = odeint(f, initial_conditions, t, method="implicit_adams",
                          rtol=rtol, atol=atol)
        else:
            traj = odeint(f, initial_conditions, t, method=method, rtol=rtol, atol=atol)
    return traj


def sample_initial_conditions(
    system: DynamicalSystem,
    n: int,
    noise: float = 0.1,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample n initial conditions around the system's default initial state."""
    x0_base = system.default_init_state()                           # (d,)
    perturb = torch.randn(n, system.state_dim, generator=generator) * noise
    return x0_base.unsqueeze(0) + perturb                           # (n, d)


class TrajectoryDataset(Dataset):
    """Pairs of (x0, target_trajectory). Time grid is shared across the dataset."""

    def __init__(self, x0: torch.Tensor, target: torch.Tensor, t: torch.Tensor):
        assert x0.shape[0] == target.shape[1]
        assert t.shape[0] == target.shape[0]
        self.x0 = x0
        self.target = target
        self.t = t

    def __len__(self) -> int:
        return self.x0.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # target stored as (T, B, d) -> per-sample target is (T, d).
        return self.x0[idx], self.target[:, idx, :]
