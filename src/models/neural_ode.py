"""Neural ODE wrapper: integrates an MLP vector field with a chosen solver."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..solvers.wrappers import integrate


class NeuralODE(nn.Module):
    """Wraps a vector field with an ODE solver.

    The choice of solver is delegated to `solvers.wrappers.integrate` so the same
    model can be evaluated under multiple solvers without re-initialization.
    """

    def __init__(self, vector_field: nn.Module, solver: str = "dopri5") -> None:
        super().__init__()
        self.f = vector_field
        self.solver_name: str = solver
        self.nfe: int = 0

    def reset_nfe(self) -> None:
        self.nfe = 0

    def _counted_f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        return self.f(t, x)

    def forward(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Integrate from x0 over the time grid `t`. Returns (T, B, d)."""
        self.reset_nfe()
        return integrate(self._counted_f, x0, t, solver=self.solver_name)
