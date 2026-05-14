"""Context manager and decorator for counting NFEs.

NFE counting is the central numerical-cost metric of this study, so the counter
should be the single source of truth used by both the trainer and the diagnostics.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable


class NFECounter:
    """Wraps a vector field, incrementing `count` on every call.

    Usage:
        counter = NFECounter(model.f)
        traj = odeint(counter, x0, t)
        print(counter.count)
    """

    def __init__(self, f: Callable):
        self.f = f
        self.count = 0

    def __call__(self, t, x):
        self.count += 1
        return self.f(t, x)

    def reset(self) -> None:
        self.count = 0


@contextmanager
def count_nfes(model):
    """Context manager that resets and reads the NFE counter on a NeuralODE."""
    model.reset_nfe()
    yield model
    # model.nfe is now the count for this block
