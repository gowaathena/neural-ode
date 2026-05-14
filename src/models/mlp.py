"""Small MLP used as the vector field f_theta(x, t) in the Neural ODE."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Time-dependent MLP: input is concat([x, t]), output is dx/dt.

    Kept intentionally simple (one config knob: hidden width and depth) so that
    variation across experiment cells is attributable to the experimental factors,
    not architecture differences.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        num_hidden: int = 2,
        time_dependent: bool = True,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.time_dependent = time_dependent
        in_dim = state_dim + (1 if time_dependent else 0)
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), activation()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Vector field: returns dx/dt of shape matching x.

        Accepts t as either a scalar tensor (torchdiffeq convention) or a (B,)
        per-sample tensor (used by the diagnostics on sampled trajectory points).
        """
        if self.time_dependent:
            if t.ndim == 0:
                t_feat = t.expand(x.shape[0]).unsqueeze(-1)
            else:
                t_feat = t.reshape(-1, 1)
            inp = torch.cat([x, t_feat], dim=-1)
        else:
            inp = x
        return self.net(inp)
