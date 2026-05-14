"""Generate the proposal's planned figures from one-run-per-file results JSONs.

Reads every *.json file in --results-dir, expects each to be a dict produced by
Trainer.run() (so containing top-level keys: val_mse, nfe_forward, jac_frobenius_*,
stiffness_ratio_*, wall_clock_per_forward, config). Aggregates across seeds and
emits:

    fig_perf_vs_nfe.png            performance (val MSE) vs NFE scatter
    fig_runtime_vs_jacnorm.png     runtime vs Jacobian Frobenius norm
    fig_step_size_hist.png         step-size mean per cell (proxy)
    table_summary.csv              aggregated mean ± std per cell
    fig_trajectory.png             ground truth vs model predictions (one seed)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        with path.open() as f:
            d = json.load(f)
        cfg = d.get("config", {})
        row = {
            "task": cfg.get("task"),
            "solver": cfg.get("solver"),
            "lambda_J": cfg.get("lambda_J", 0.0),
            "seed": cfg.get("seed"),
            **{k: v for k, v in d.items() if k not in ("history", "config")},
        }
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns
                   if c not in ("task", "solver", "lambda_J", "seed")
                   and pd.api.types.is_numeric_dtype(df[c])]
    grouped = df.groupby(["task", "solver", "lambda_J"])
    agg = grouped[metric_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


def fig_perf_vs_nfe(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for (task, lam), sub in df.groupby(["task", "lambda_J"]):
        label = f"{task} (lam={lam})"
        ax.scatter(sub["nfe_forward"], sub["val_mse"], label=label, alpha=0.7)
    ax.set_xlabel("NFE (forward pass)")
    ax.set_ylabel("Validation MSE")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_runtime_vs_jacnorm(df: pd.DataFrame, out: Path) -> None:
    if "jac_frobenius_mean" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for (task, lam), sub in df.groupby(["task", "lambda_J"]):
        ax.scatter(sub["jac_frobenius_mean"], sub["wall_clock_per_forward"],
                   label=f"{task} (lam={lam})", alpha=0.7)
    ax.set_xlabel(r"$\|J\|_F$ along trajectory (mean)")
    ax.set_ylabel("Wall-clock per forward pass (s)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_step_size(df: pd.DataFrame, out: Path) -> None:
    if "step_size_mean" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    cells = df.groupby(["task", "solver", "lambda_J"])
    labels, means, stds = [], [], []
    for name, sub in cells:
        labels.append("/".join(str(n) for n in name))
        means.append(sub["step_size_mean"].mean())
        stds.append(sub["step_size_mean"].std())
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean step size (proxy)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results_dir)
    if df.empty:
        print(f"No JSON results found in {args.results_dir}")
        return

    summary = aggregate(df)
    summary.to_csv(args.output_dir / "table_summary.csv", index=False)
    print(f"Wrote {args.output_dir / 'table_summary.csv'} with {len(summary)} cells")

    fig_perf_vs_nfe(df, args.output_dir / "fig_perf_vs_nfe.png")
    fig_runtime_vs_jacnorm(df, args.output_dir / "fig_runtime_vs_jacnorm.png")
    fig_step_size(df, args.output_dir / "fig_step_size.png")
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
