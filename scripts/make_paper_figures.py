"""Generate the figures referenced by docs/paper/paper.tex.

Outputs:
    docs/paper/figures/fig_engagement.pdf   -- Jacobian Frobenius vs lambda_J
                                               at mu=10 (engages) and mu=50 (no-op).
    docs/paper/figures/fig_h3_slope.pdf     -- A1/E3 wall-clock ratio vs mu.
    docs/paper/figures/fig_tol_scaling.pdf  -- NFE vs rtol (log-log) with
                                               theoretical slope reference.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO / "docs" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_all() -> list[dict]:
    rows = []
    for f in sorted(RESULTS.glob("*.json")):
        d = json.load(f.open())
        c = d["config"]
        rows.append({"exp": "main", "task": c["task"], "solver": c["solver"],
                     "lam": c["lambda_J"],
                     "mu": c["task_kwargs"].get("mu") if c["task"] == "vanderpol" else None,
                     "rtol": 1e-5, "K": c.get("jac_K", 1), "seed": c["seed"], **d})
    for f in sorted((RESULTS / "mu10_sweep").glob("*.json")):
        d = json.load(f.open()); c = d["config"]
        rows.append({"exp": "mu10", "task": c["task"], "solver": c["solver"],
                     "lam": c["lambda_J"], "mu": c["task_kwargs"]["mu"],
                     "rtol": 1e-5, "K": 1, "seed": c["seed"], **d})
    for f in sorted((RESULTS / "k4_sweep").glob("*.json")):
        d = json.load(f.open()); c = d["config"]
        rows.append({"exp": "k4", "task": c["task"], "solver": c["solver"],
                     "lam": c["lambda_J"], "mu": 50.0, "rtol": 1e-5,
                     "K": 4, "seed": c["seed"], **d})
    for subdir in (RESULTS / "tol_sweep").iterdir():
        rtol = float(subdir.name.replace("rtol", ""))
        for f in subdir.glob("*.json"):
            d = json.load(f.open()); c = d["config"]
            rows.append({"exp": "tol", "task": c["task"], "solver": c["solver"],
                         "lam": c["lambda_J"], "mu": 50.0, "rtol": rtol, "K": 1,
                         "seed": c["seed"], **d})
    for subdir in (RESULTS / "mu_sweep").iterdir():
        mu = float(subdir.name.replace("mu", ""))
        for f in subdir.glob("*.json"):
            d = json.load(f.open()); c = d["config"]
            rows.append({"exp": "mu_sweep", "task": c["task"], "solver": c["solver"],
                         "lam": c["lambda_J"], "mu": mu, "rtol": 1e-5, "K": 1,
                         "seed": c["seed"], **d})
    return rows


def mean_std(xs):
    if len(xs) < 2:
        return statistics.mean(xs), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def fig_engagement(rows: list[dict]) -> None:
    """Jacobian Frobenius vs lambda_J at mu=10 and mu=50."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for mu, color, marker in [(10.0, "C3", "o"), (50.0, "C0", "s")]:
        lams_data = defaultdict(list)
        for r in rows:
            if (r["task"], r["solver"]) != ("vanderpol", "dopri5"):
                continue
            if r["mu"] != mu or r["K"] != 1:
                continue
            lams_data[r["lam"]].append(r["jac_frobenius_mean"])
        lams = sorted(lams_data.keys())
        means = [statistics.mean(lams_data[l]) for l in lams]
        stds = [statistics.stdev(lams_data[l]) if len(lams_data[l]) > 1 else 0.0 for l in lams]
        # Plot lam=0 as a "0" point; use a tiny x value for log scale.
        xs = [(l if l > 0 else 1e-5) for l in lams]
        ax.errorbar(xs, means, yerr=stds, marker=marker, color=color, capsize=3,
                    label=fr"$\mu={int(mu)}$ (val MSE $\approx${'' if mu==10 else ' '}{1.18 if mu==10 else 1.94:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel(r"Regularization weight $\lambda_J$ (0 plotted at $10^{-5}$)")
    ax.set_ylabel(r"Learned $\|J_{f_\theta}\|_F$ (mean along val trajectory)")
    ax.set_title("Jacobian regularization engagement regime (Van der Pol)")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_engagement.pdf")
    plt.close(fig)
    print(f"Wrote {OUT / 'fig_engagement.pdf'}")


def fig_h3_slope(rows: list[dict]) -> None:
    """Seed-matched implicit/dopri5 wall-clock ratio vs mu."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    mus, ratios, ratio_stds = [], [], []
    for mu in [25.0, 50.0, 100.0]:
        e3 = {r["seed"]: r["wall_clock_per_forward"] for r in rows
              if (r["task"], r["solver"]) == ("vanderpol", "dopri5")
              and r["mu"] == mu and r["lam"] == 0 and r["rtol"] == 1e-5 and r["K"] == 1}
        a1 = {r["seed"]: r["wall_clock_per_forward"] for r in rows
              if (r["task"], r["solver"]) == ("vanderpol", "implicit_adams")
              and r["mu"] == mu and r["lam"] == 0 and r["rtol"] == 1e-5}
        seeds = sorted(set(e3) & set(a1))
        if len(seeds) < 5:
            continue
        per_seed_ratios = [a1[s] / e3[s] for s in seeds]
        mus.append(mu)
        ratios.append(statistics.mean(per_seed_ratios))
        ratio_stds.append(statistics.stdev(per_seed_ratios))
    ax.errorbar(mus, ratios, yerr=ratio_stds, marker="o", capsize=4,
                color="C2", label="implicit / explicit wall-clock")
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5, label="parity")
    ax.set_xlabel(r"Van der Pol stiffness parameter $\mu$")
    ax.set_ylabel(r"Wall-clock ratio (implicit\_adams / dopri5)")
    ax.set_title(r"H3: Implicit advantage shrinks (ratio grows) with $\mu$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_h3_slope.pdf")
    plt.close(fig)
    print(f"Wrote {OUT / 'fig_h3_slope.pdf'}")


def fig_tol_scaling(rows: list[dict]) -> None:
    """NFE vs rtol on log-log, with theoretical slope -1/(p+1) for dopri5."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for solver, color, marker, p in [("dopri5", "C0", "o", 5), ("implicit_adams", "C3", "s", None)]:
        rtols, nfes, stds = [], [], []
        for rtol in [1e-3, 1e-5, 1e-7]:
            data = [r["nfe_forward"] for r in rows
                    if (r["task"], r["solver"]) == ("vanderpol", solver)
                    and r["rtol"] == rtol and r["lam"] == 0 and r["mu"] == 50.0]
            if data:
                rtols.append(rtol)
                nfes.append(statistics.mean(data))
                stds.append(statistics.stdev(data) if len(data) > 1 else 0.0)
        ax.errorbar(rtols, nfes, yerr=stds, marker=marker, color=color, capsize=4,
                    label=f"{solver}")
        if p is not None:
            # Plot theoretical slope through the middle point.
            base = nfes[1]
            theo_xs = np.array(rtols)
            theo_ys = base * (1e-5 / theo_xs) ** (1 / (p + 1))
            ax.plot(theo_xs, theo_ys, ":", color=color, alpha=0.6,
                    label=fr"theory: $N \propto \tau^{{-1/{p+1}}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Relative tolerance $r_\text{tol}$")
    ax.set_ylabel("NFE (forward pass)")
    ax.set_title(r"H0: NFE--tolerance scaling, Van der Pol $\mu=50$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tol_scaling.pdf")
    plt.close(fig)
    print(f"Wrote {OUT / 'fig_tol_scaling.pdf'}")


def main() -> None:
    rows = load_all()
    print(f"Loaded {len(rows)} runs")
    fig_engagement(rows)
    fig_h3_slope(rows)
    fig_tol_scaling(rows)


if __name__ == "__main__":
    main()
