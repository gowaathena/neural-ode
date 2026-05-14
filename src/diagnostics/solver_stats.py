"""Adaptive-solver step-size statistics.

torchdiffeq does not expose the per-step accepted/rejected history through the
high-level API, so this module provides two access paths:

1. `step_size_proxy(nfe, t_span, solver)` — a cheap proxy that estimates the mean
   accepted step size from NFE and the nominal stages-per-step of the method.
   This is what the trainer uses for periodic logging.

2. `MonkeyPatchStepLogger` — context manager that monkey-patches torchdiffeq's
   internal `_AdaptiveStepsizeODESolver._advance_until_event`-style hooks to
   record the actual accepted step times. Use this for the paper's step-size
   histograms; do not use it during training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from ..solvers.wrappers import SOLVER_REGISTRY


@dataclass
class StepSizeStats:
    accepted_steps: list[float] = field(default_factory=list)
    rejected_count: int = 0

    @property
    def n_accepted(self) -> int:
        return len(self.accepted_steps)

    def summary(self) -> dict[str, float]:
        if not self.accepted_steps:
            return {
                "min": math.nan, "mean": math.nan, "max": math.nan, "std": math.nan,
                "n_accepted": 0, "n_rejected": self.rejected_count,
            }
        s = torch.tensor(self.accepted_steps, dtype=torch.float32)
        return {
            "min": float(s.min()),
            "mean": float(s.mean()),
            "max": float(s.max()),
            "std": float(s.std(unbiased=False)),
            "n_accepted": self.n_accepted,
            "n_rejected": self.rejected_count,
        }


def step_size_proxy(nfe: int, t_span: float, solver: str) -> dict[str, float]:
    """Rough step-size statistics derivable from NFE alone.

    For fixed-step methods, the mean equals the (constant) step size set in the
    solver kwargs. For adaptive methods, mean = t_span / (nfe / stages_per_step).
    Min, max, and std are reported as NaN; use `MonkeyPatchStepLogger` for
    proper step-size histograms.
    """
    spec = SOLVER_REGISTRY[solver]
    if spec.kind == "explicit_fixed":
        h = spec.kwargs.get("options", {}).get("step_size", math.nan)
        n_steps = max(1, int(t_span / h)) if h and not math.isnan(h) else 1
        return {
            "min": float(h), "mean": float(h), "max": float(h),
            "std": 0.0, "n_accepted": n_steps, "n_rejected": 0,
        }
    # Adaptive (explicit or implicit): infer mean step from NFE.
    n_steps = max(1, nfe // spec.stages_per_step)
    return {
        "min": math.nan,
        "mean": float(t_span) / n_steps,
        "max": math.nan,
        "std": math.nan,
        "n_accepted": n_steps,
        "n_rejected": 0,
    }


class MonkeyPatchStepLogger:
    """Context manager: logs accepted step sizes via torchdiffeq monkey-patch.

    Usage:
        with MonkeyPatchStepLogger() as logger:
            traj = model(x0, t)
        print(logger.stats.summary())

    Only patches the dopri5 path. Other adaptive solvers will fall back to the
    cheap proxy. This is intentionally limited because the monkey-patch is
    fragile and only the dopri5 path is exercised heavily in the experiment matrix.
    """

    def __init__(self) -> None:
        self.stats = StepSizeStats()
        self._original_advance = None

    def __enter__(self) -> "MonkeyPatchStepLogger":
        try:
            from torchdiffeq._impl.dopri5 import Dopri5Solver
        except ImportError:
            return self  # graceful no-op if torchdiffeq internals change

        original_step = Dopri5Solver._step_func
        accepted = self.stats.accepted_steps

        def logged_step(self_, func, t0, dt, t1, y0):
            accepted.append(float(dt))
            return original_step(self_, func, t0, dt, t1, y0)

        self._original_advance = original_step
        Dopri5Solver._step_func = logged_step
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._original_advance is None:
            return
        from torchdiffeq._impl.dopri5 import Dopri5Solver
        Dopri5Solver._step_func = self._original_advance
        self._original_advance = None
