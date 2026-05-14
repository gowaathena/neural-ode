"""Thin wrapper over torchdiffeq's odeint exposing a uniform integrate(...) API."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torchdiffeq import odeint


@dataclass
class SolverSpec:
    name: str                                       # method name passed to torchdiffeq
    kwargs: dict = field(default_factory=dict)      # rtol/atol or step_size
    kind: str = "explicit_adaptive"                 # {explicit_adaptive, explicit_fixed, implicit}
    stages_per_step: int = 6                        # used by step-size proxies


SOLVER_REGISTRY: dict[str, SolverSpec] = {
    "dopri5": SolverSpec(
        name="dopri5",
        kwargs={"rtol": 1e-5, "atol": 1e-7},
        kind="explicit_adaptive",
        stages_per_step=6,
    ),
    "rk4": SolverSpec(
        name="rk4",
        kwargs={"options": {"step_size": 0.05}},
        kind="explicit_fixed",
        stages_per_step=4,
    ),
    # Ancillary implicit option. torchdiffeq's implicit_adams uses fixed-point
    # iteration (not Newton), but it has the right A-stability story for the
    # ancillary comparison.
    "implicit_adams": SolverSpec(
        name="implicit_adams",
        kwargs={"rtol": 1e-5, "atol": 1e-7},
        kind="implicit",
        stages_per_step=1,
    ),
}


def integrate(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    t: torch.Tensor,
    solver: str = "dopri5",
) -> torch.Tensor:
    """Integrate dx/dt = f(t, x) from x0 over the time grid t.

    Returns trajectory of shape (T, *x0.shape).
    """
    spec = SOLVER_REGISTRY[solver]
    return odeint(f, x0, t, method=spec.name, **spec.kwargs)
