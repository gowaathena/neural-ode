"""Evaluation metrics for trajectory regression."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def trajectory_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error over all (time, batch, state) entries.

    Args:
        pred:   (T, B, d)
        target: (T, B, d)
    """
    return ((pred - target) ** 2).mean()


def summarize_metrics(values: Iterable[float], ci: float = 0.95) -> dict[str, float]:
    """Mean / std / bootstrap CI across seeds. Returns NaNs if inputs empty."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return {"mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    mean = float(v.mean())
    std = float(v.std(ddof=0))
    # Bootstrap CI
    n_boot = 1000
    rng = np.random.default_rng(0)
    boot = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"mean": mean, "std": std, "ci_low": float(lo), "ci_high": float(hi), "n": int(v.size)}
