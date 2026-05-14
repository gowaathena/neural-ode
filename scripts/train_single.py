"""Train one (task, solver, lambda_J, seed) experiment cell.

Usage:
    python scripts/train_single.py --config configs/experiments/E1_baseline_dopri5_smooth.yaml --seed 0
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.training import Trainer, TrainConfig
from src.utils import set_seed


def load_merged_config(experiment_yaml: Path, base_yaml: Path) -> dict:
    with base_yaml.open() as f:
        cfg = yaml.safe_load(f) or {}
    with experiment_yaml.open() as f:
        override = yaml.safe_load(f) or {}
    cfg.update(override)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=REPO_ROOT / "configs" / "base.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--lambda-J", dest="lambda_J", type=float, default=None,
                        help="Override config lambda_J (Jacobian penalty weight)")
    parser.add_argument("--jac-K", dest="jac_K", type=int, default=None,
                        help="Override config jac_K (Hutchinson sample count)")
    parser.add_argument("--mu", type=float, default=None,
                        help="Override task_kwargs.mu (Van der Pol stiffness parameter)")
    parser.add_argument("--rtol", type=float, default=None,
                        help="Override solver rtol (and reflect in NFE counter behaviour)")
    parser.add_argument("--atol", type=float, default=None,
                        help="Override solver atol")
    args = parser.parse_args()

    cfg_dict = load_merged_config(args.config, args.base_config)
    cfg_dict["seed"] = args.seed
    cfg_dict["output_dir"] = str(args.output_dir)
    if args.epochs is not None:
        cfg_dict["epochs"] = args.epochs
    if args.lambda_J is not None:
        cfg_dict["lambda_J"] = args.lambda_J
    if args.jac_K is not None:
        cfg_dict["jac_K"] = args.jac_K
    if args.mu is not None:
        cfg_dict.setdefault("task_kwargs", {})["mu"] = args.mu
    if args.rtol is not None or args.atol is not None:
        # Patch SOLVER_REGISTRY at runtime so the chosen solver uses the override.
        from src.solvers.wrappers import SOLVER_REGISTRY
        spec = SOLVER_REGISTRY[cfg_dict.get("solver", "dopri5")]
        if args.rtol is not None:
            spec.kwargs["rtol"] = args.rtol
        if args.atol is not None:
            spec.kwargs["atol"] = args.atol

    # Drop unknown keys so we don't crash on harmless yaml entries.
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    filtered = {k: v for k, v in cfg_dict.items() if k in known}

    cfg = TrainConfig(**filtered)
    set_seed(cfg.seed)
    trainer = Trainer(cfg)
    summary = trainer.run()
    # Print top-level numbers; full history is on disk.
    print({k: v for k, v in summary.items() if k not in ("history", "config")})


if __name__ == "__main__":
    main()
