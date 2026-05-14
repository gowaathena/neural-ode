# When Does Jacobian Regularization Engage in Neural ODEs?

Empirical Neural ODE study for ENM 5220, focused on when Jacobian Frobenius
regularization changes learned dynamics and solver cost on smooth versus stiff
trajectory-regression tasks.

The paper is in [`docs/paper/paper.pdf`](docs/paper/paper.pdf), with LaTeX
source in [`docs/paper/paper.tex`](docs/paper/paper.tex).

## Summary

The project trains small MLP Neural ODE vector fields on damped harmonic
oscillator and Van der Pol trajectories. It compares:

- Jacobian regularization weights `lambda_J in {0, 1e-3, 1e-2, 1e-1}` where applicable.
- Solvers `dopri5`, `rk4`, and `implicit_adams` through `torchdiffeq`.
- Van der Pol stiffness settings `mu in {5, 10, 25, 50, 100}`.
- Train/eval solver tolerances `rtol in {1e-3, 1e-5, 1e-7}` for the tolerance sweep.

The main finding is an engagement decay: the Jacobian penalty visibly reduces
the learned Jacobian norm at lower `mu`, but the effect fades across the
Van der Pol sweep and is nearly inert in the poorer-fit, stiffer settings.
The paper frames this as a penalty-fit-stiffness trade-off rather than as a
universal claim about all stiff Neural ODEs.

## Repository Layout

```text
.
├── configs/                 # Base/task/solver/experiment YAML configs
├── docs/
│   ├── paper/               # Final paper source, PDF, refs, and figures
│   └── theory/              # Supporting numerical-analysis notes
├── proposal/                # Original proposal source/PDF
├── results/                 # Result JSONs used by the paper
├── scripts/                 # Training, sweep, plotting, and paper-figure scripts
├── src/                     # Importable project code
│   ├── data/                # ODE systems and trajectory generation
│   ├── diagnostics/         # Jacobian, stiffness, and solver diagnostics
│   ├── evaluation/          # Metrics and statistical tests
│   ├── models/              # MLP vector field and NeuralODE wrapper
│   ├── regularizers/        # Exact and Hutchinson Jacobian penalties
│   ├── solvers/             # torchdiffeq wrappers and NFE counting
│   ├── training/            # Experiment trainer
│   └── utils/               # Seeding utilities
└── tests/                   # Unit tests
```

Result JSONs are intentionally kept in the repository for paper
reproducibility. Local environments, Python caches, LaTeX build intermediates,
and transient logs are ignored by `.gitignore`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Common Commands

Run tests:

```bash
pytest -q
```

Train one experiment cell:

```bash
python scripts/train_single.py \
  --config configs/experiments/E1_baseline_dopri5_smooth.yaml \
  --seed 0
```

Run the original experiment matrix across seeds:

```bash
python scripts/run_matrix.py --seeds 0,1,2,3,4
```

Regenerate paper figures from the bundled result JSONs:

```bash
python scripts/make_paper_figures.py
```

Build the paper PDF:

```bash
cd docs/paper
latexmk -pdf -interaction=nonstopmode paper.tex
```

## Notes

- The tolerance sweep mutates the solver tolerance before training each cell,
  so it is a train/eval tolerance sweep rather than an eval-only sweep.
- `implicit_adams` is the fixed-point implicit method exposed by `torchdiffeq`;
  it is not a Newton-based BDF or Radau solver.
- The Van der Pol default time grid scales as `T_end = max(20, 2 * mu)`, so
  higher-`mu` sweeps co-vary stiffness, horizon length, and raw time-input
  range. This limitation is discussed in the paper.
