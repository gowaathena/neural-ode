"""Training loop for one (task, solver, lambda_J, seed) experiment cell."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..data.dataset import generate_trajectories, sample_initial_conditions
from ..data.dynamics import make_system
from ..diagnostics import (
    jacobian_frobenius_along_trajectory,
    step_size_proxy,
    stiffness_ratio_along_trajectory,
)
from ..models import MLP, NeuralODE
from ..regularizers import jacobian_frobenius_exact, jacobian_frobenius_hutchinson
from ..utils import set_seed


@dataclass
class TrainConfig:
    # task / data
    task: str = "harmonic"
    task_kwargs: dict = field(default_factory=dict)
    n_train: int = 256
    n_val: int = 64
    ic_noise: float = 0.1
    # model
    hidden_dim: int = 64
    num_hidden: int = 2
    # solver
    solver: str = "dopri5"
    # regularization
    lambda_J: float = 0.0
    jac_K: int = 1
    jac_exact: bool = False
    penalty_batch: int = 32
    # optimization
    lr: float = 1.0e-3
    epochs: int = 100
    batch_size: int = 32
    # ground-truth data generation
    truth_method: str = "dopri5"
    truth_rtol: float = 1.0e-9
    truth_atol: float = 1.0e-11
    # reproducibility
    seed: int = 0
    # logging
    log_every: int = 10
    output_dir: str = "results"


class Trainer:
    """Runs one experiment cell end-to-end and writes metrics to disk."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        set_seed(cfg.seed)

        self.system = make_system(cfg.task, cfg.task_kwargs)
        self.t = self.system.default_time_grid()

        gen = torch.Generator().manual_seed(cfg.seed)
        x0_train = sample_initial_conditions(self.system, cfg.n_train, cfg.ic_noise, gen)
        x0_val = sample_initial_conditions(self.system, cfg.n_val, cfg.ic_noise, gen)
        self.target_train = generate_trajectories(
            self.system, x0_train, self.t,
            method=cfg.truth_method, rtol=cfg.truth_rtol, atol=cfg.truth_atol,
        )                                           # (T, n_train, d)
        self.target_val = generate_trajectories(
            self.system, x0_val, self.t,
            method=cfg.truth_method, rtol=cfg.truth_rtol, atol=cfg.truth_atol,
        )
        self.x0_train = x0_train
        self.x0_val = x0_val

        self.f = MLP(
            state_dim=self.system.state_dim,
            hidden_dim=cfg.hidden_dim,
            num_hidden=cfg.num_hidden,
        )
        self.model = NeuralODE(self.f, solver=cfg.solver)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --- penalty sampling --------------------------------------------------

    def _sample_penalty_points(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick n (t, x) samples from the training targets for the regularizer."""
        T, B, d = self.target_train.shape
        idx = torch.randint(0, T * B, (n,))
        t_idx = idx // B
        b_idx = idx % B
        x = self.target_train[t_idx, b_idx, :]
        t = self.t[t_idx]
        return t, x

    def _penalty(self) -> torch.Tensor:
        t_p, x_p = self._sample_penalty_points(self.cfg.penalty_batch)
        x_p = x_p.detach().requires_grad_(True)
        if self.cfg.jac_exact:
            return jacobian_frobenius_exact(self.f, t_p, x_p, create_graph=True).mean()
        return jacobian_frobenius_hutchinson(
            self.f, t_p, x_p, K=self.cfg.jac_K, create_graph=True
        ).mean()

    # --- training loop -----------------------------------------------------

    def _train_step(self, x0_batch: torch.Tensor, target_batch: torch.Tensor) -> float:
        self.opt.zero_grad()
        pred = self.model(x0_batch, self.t)                         # (T, B, d)
        task_loss = ((pred - target_batch) ** 2).mean()
        loss = task_loss
        if self.cfg.lambda_J > 0:
            loss = loss + self.cfg.lambda_J * self._penalty()
        loss.backward()
        self.opt.step()
        return float(task_loss.detach())

    def run(self) -> dict:
        target_train_per_sample = self.target_train.transpose(0, 1)  # (n_train, T, d)
        loader = DataLoader(
            TensorDataset(self.x0_train, target_train_per_sample),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )
        history: list[dict] = []
        for epoch in range(self.cfg.epochs):
            t0 = time.time()
            self.model.train()
            train_losses: list[float] = []
            for x0_b, target_b in loader:
                target_b = target_b.transpose(0, 1).contiguous()    # back to (T, B, d)
                train_losses.append(self._train_step(x0_b, target_b))
            wall_epoch = time.time() - t0
            if (epoch + 1) % self.cfg.log_every == 0 or epoch == self.cfg.epochs - 1:
                metrics = self.evaluate(detailed=False)
                metrics.update({
                    "epoch": epoch,
                    "train_mse": float(sum(train_losses) / max(1, len(train_losses))),
                    "wall_clock_per_epoch": wall_epoch,
                })
                history.append(metrics)

        final = self.evaluate(detailed=True)
        final["config"] = {
            "task": self.cfg.task, "task_kwargs": self.cfg.task_kwargs,
            "solver": self.cfg.solver, "lambda_J": self.cfg.lambda_J,
            "jac_K": self.cfg.jac_K, "jac_exact": self.cfg.jac_exact,
            "seed": self.cfg.seed,
        }
        final["history"] = history
        self._save(final)
        return final

    # --- evaluation --------------------------------------------------------

    def evaluate(self, detailed: bool = False) -> dict:
        self.model.eval()
        with torch.no_grad():
            t0 = time.time()
            pred = self.model(self.x0_val, self.t)
            wall_forward = time.time() - t0
        val_mse = float(((pred - self.target_val) ** 2).mean())
        nfe = int(self.model.nfe)
        out: dict = {
            "val_mse": val_mse,
            "nfe_forward": nfe,
            "wall_clock_per_forward": wall_forward,
        }
        if not detailed:
            return out

        # Detailed diagnostics: Jacobian norm, stiffness ratio, step-size proxy.
        jac_norms = jacobian_frobenius_along_trajectory(self.f, self.t, pred, subsample=20)
        out["jac_frobenius_mean"] = float(jac_norms.mean())
        out["jac_frobenius_max"] = float(jac_norms.max())
        stiff = stiffness_ratio_along_trajectory(self.f, self.t, pred, subsample=20)
        out["stiffness_ratio_mean"] = float(stiff.mean())
        out["stiffness_ratio_max"] = float(stiff.max())
        t_span = float(self.t[-1] - self.t[0])
        for k, v in step_size_proxy(nfe, t_span, self.cfg.solver).items():
            out[f"step_size_{k}"] = float(v)
        return out

    # --- IO ----------------------------------------------------------------

    def _save(self, summary: dict) -> None:
        slug = (
            f"{self.cfg.task}-{self.cfg.solver}-"
            f"lam{self.cfg.lambda_J}-seed{self.cfg.seed}"
        )
        path = self.output_dir / f"{slug}.json"
        with path.open("w") as f:
            json.dump(summary, f, indent=2, default=float)
