# 03 — Stiffness and implicit time integration

A response to the professor's question: *"If the problem is with stiff ODE systems,
what about exploring the use of implicit time integration? Using large time steps
would decrease NFE?"*

The short answer: yes, implicit methods bypass the stability step restriction
(Mechanism B in `01_step_size_and_stability.md`) and can use much larger steps on
stiff problems — but at a higher per-step cost. Including an implicit solver in the
study **strengthens** the research question by giving us two distinct strategies for
handling stiffness, which we can directly compare.

## 1. Why implicit methods help on stiff problems

For the linear test equation $\dot y = \lambda y$, $\text{Re}(\lambda) < 0$:

- **Explicit Euler:** $y_{n+1} = (1 + h\lambda) y_n$. Stable iff $|1 + h\lambda| \leq 1$,
  i.e., $h \leq 2/|\lambda|$.
- **Implicit Euler:** $y_{n+1} = (1 - h\lambda)^{-1} y_n$. Stable for *all* $h > 0$
  (A-stable). Damping factor $|1 - h\lambda|^{-1} \to 0$ as $h \to \infty$, so it is
  also L-stable.

For an order-$p$ method on an autonomous system, A-stability means the stability
region contains the entire left half-plane, so the step size is governed *only by
accuracy*, not by stability. On a stiff system with widely separated eigenvalues,
this is dramatic: the fast modes decay quickly anyway, so the slow-mode time scale
determines the step.

## 2. The cost trade-off

An implicit step requires solving a nonlinear equation
$$x_{n+1} - x_n - h \, \Phi(x_{n+1}, x_n, h) = 0,$$
typically by Newton iteration. Each Newton step requires:
1. one evaluation of $f_\theta$ (an NFE);
2. one evaluation of $J_{f_\theta}$ (a Jacobian — itself $O(d)$ backward passes for
   reverse-mode, or one matrix worth of forward-mode if using JVPs);
3. one linear solve of $(I - h \gamma J) \Delta = -r$ (cost $O(d^3)$ dense, or
   $O(d \cdot \text{nnz})$ sparse).

So the "NFE" metric is no longer the right unit. For an implicit BDF or Radau IIA:
$$\text{cost per step} = (\text{Newton iters}) \times (\text{stages}) \times (\text{NFE + Jac cost})$$

This is why NFE-only comparisons between explicit and implicit methods are
misleading. For the implicit ancillary experiment, log all of:

- outer integrator step count,
- cumulative Newton iterations,
- cumulative NFE,
- cumulative Jacobian evaluations,
- wall-clock time.

Then make the comparison on **wall-clock at matched validation MSE**.

## 3. Two strategies for stiffness — and the comparison

This project compares two distinct interventions for the "stiff target → expensive
solver" problem:

| Strategy | What it changes | Cost paid |
|----------|-----------------|-----------|
| Jacobian regularization (explicit solver) | Trains $f_\theta$ to have small $\rho(J)$; learned dynamics are smoother than the target | Training cost ↑ (Jacobian per step); breaks adjoint method; potential bias if penalty too strong |
| Implicit solver (no regularization) | Uses a solver whose stability region contains the stiff eigenvalues; learned $f_\theta$ is unconstrained | Per-step cost ↑ (Newton + linear solve); requires Jacobian access at solve time |

The professor's intuition — "implicit methods take larger steps, so NFE drops" — is
correct, and matches the L-stable theory. The interesting empirical question is then
**whether the two strategies are redundant or complementary**:

- If `implicit + no-reg` already achieves low wall-clock cost at matched MSE on the
  stiff task, Jacobian regularization is *useful for explicit-only setups* but not a
  general necessity. The takeaway becomes: "regularize, or switch solver".
- If `implicit + reg` beats both individually, the interventions attack different
  aspects (e.g., regularization improves Newton convergence in addition to step
  size).
- If `explicit + reg` matches `implicit + no-reg` at lower training cost, regularize.

This is now framed as Hypothesis H3 (ancillary).

## 4. Practical caveats for Neural ODEs

- **Library support.** `torchdiffeq` has limited implicit-solver support
  (`implicit_adams` is the main option, and it lacks Newton-iteration logging).
  Consider `torchode` (PyTorch, has implicit Euler and Trapezoidal) or `diffrax`
  (JAX, has full BDF/Kvaerno). If you stick with PyTorch end-to-end, `torchode` is
  the path of least resistance.
- **Differentiability.** Backpropagating through an implicit step requires either
  (i) differentiating through the Newton iteration (memory-heavy), or
  (ii) the implicit-function theorem applied to the Newton residual (cleaner, but
  requires a linear solve on the backward pass too).
- **Jacobian access.** Newton iteration needs $J_{f_\theta}$ at each step anyway —
  so the implicit solver "already pays" the Jacobian cost. This is a sharp asymmetry
  with the explicit + regularization variant, where Jacobian access during training
  is the main bottleneck.

## 5. Updated experiment matrix (with ancillary)

Add to the core 8-cell 2×2×2 design:

| Cell | Dynamics variant | Solver | Task | Role |
|------|------------------|--------|------|------|
| A1 | Baseline | Implicit (BDF / Kvaerno5 / implicit Adams) | Mildly stiff | Implicit alternative to regularization |
| A2 (optional) | Jacobian-reg | Implicit | Mildly stiff | Tests whether regularization stacks with implicit |

Five seeds per ancillary cell adds 5–10 runs to the budget. Report wall-clock and
the cost decomposition from §2 alongside MSE.

## 6. Proof obligations for the report

- **State and prove A-stability of implicit Euler** on the linear test equation.
  One-paragraph proof; cite Hairer–Wanner Vol. II §IV.3.
- **State (no proof needed) L-stability of BDF1 and BDF2**, and the Dahlquist barrier
  (no A-stable linear multistep method has order $> 2$).
- **Cost decomposition lemma.** For an order-$q$ implicit Runge–Kutta with $s$
  stages, $N$ outer steps, and average Newton iterations $\bar K$ per step, the
  cumulative cost is $N \bar K s$ NFE + $N \bar K$ Jacobian evaluations + $N \bar K$
  linear solves. (This is the unit you use to compare against explicit-method NFE.)
