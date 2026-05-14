# 02 — What Jacobian regularization actually does

A response to the professor's note: "regularization is a sophisticated nickname for
ad-hoc fixing". Jacobian regularization is **not** ad-hoc — it has a clean
interpretation as a soft Lipschitz constraint, and its effect on solver cost is
predicted by the step-size theory in `01_step_size_and_stability.md`.

## 1. Definition

The training objective is
$$\mathcal{L}_\text{total}(\theta) = \mathcal{L}_\text{task}(\theta)
   + \lambda_J \cdot \mathbb{E}_{(x,t) \sim \mu}\bigl[ \|J_{f_\theta}(x,t)\|_F^2 \bigr],$$
where $\mu$ is the empirical distribution over states and times sampled along
training trajectories, and $\|\cdot\|_F$ is the Frobenius norm:
$$\|J\|_F^2 = \sum_{i,j} J_{ij}^2 = \text{tr}(J^\top J) = \sum_i \sigma_i(J)^2.$$

## 2. Three equivalent interpretations

### (a) Soft Lipschitz bound on $f_\theta$

The operator norm $\|J\|_2$ is the local Lipschitz constant of $f_\theta$, and
$\|J\|_2 \leq \|J\|_F$. So minimising $\|J\|_F^2$ caps how fast $f_\theta$ can vary
with state — exactly the property that controls how hard an ODE is to integrate.
This is the same intuition as weight decay (which caps the norm of a linear map),
extended to a nonlinear map.

### (b) Soft cap on local sensitivity / amplification

For a small perturbation $\delta x$ of the state at time $t$, the flow's sensitivity
after $\Delta t$ is approximately $\exp(J \Delta t) \delta x$. Reducing $\|J\|$
reduces how aggressively perturbations are amplified — the learned flow is closer to
a contraction.

### (c) Soft spectral-radius cap (the stiff-system mechanism)

From `01_step_size_and_stability.md` §4: $\rho(J)^2 \leq \|J\|_F^2$. So the penalty
puts a *quadratic* upper bound on the spectral radius. For stiff systems where the
solver's step size is set by $\rho(J)$, this is the directly load-bearing inequality.

## 3. What the penalty does **not** do

- It is **not** a constraint — $\lambda_J$ controls a soft trade-off. If task loss
  benefits from a momentarily large $\|J\|$, the optimiser will accept that.
- It is **not** the only way to bound $\|J\|$. Spectral normalisation and weight
  clipping bound it more strictly; the Frobenius penalty is preferred because it is
  differentiable, cheap (with Hutchinson), and global rather than per-layer.
- It does **not** assume the target dynamics are smooth. The penalty is on the
  *learned* $f_\theta$. The target can be arbitrarily stiff; the penalty is asking
  whether the network needs to *reproduce* that stiffness internally.

## 4. The implementation choice: exact vs Hutchinson

| Variant | Cost per evaluation | When to use |
|---------|---------------------|-------------|
| Exact $\|J\|_F^2$ | $d$ backward passes through $f_\theta$ → $O(d \cdot \text{params})$ | $d \leq 10$, diagnostic evaluation, reported metrics |
| Hutchinson $\hat{\|J\|}_F^2$ | $K$ backward passes; unbiased estimate | training-time penalty; $K = 1$ usually sufficient |

See `04_hutchinson_estimator.md` for the unbiasedness proof and variance analysis.

## 5. Why this is not "ad-hoc fixing"

The phrase "ad-hoc fix" applies when a penalty has no derivation from the problem
structure and is added because empirical results were unsatisfying. By contrast:

1. The step-size theory in `01_step_size_and_stability.md` predicts that NFE scales
   with $\|J\|$ (accuracy) and $\rho(J)$ (stability).
2. The Frobenius norm is the *largest convex differentiable upper bound* on the
   spectral radius squared. Penalising it is the natural relaxation of "constrain
   $\rho(J)$".
3. The penalty has a probabilistic interpretation (a Gaussian prior on the
   off-diagonal blocks of $J$, roughly) and a kinetic-energy interpretation (see
   Finlay et al. 2020 §3 — kinetic regularization $\int \|f\|^2 dt$ and Jacobian
   regularization together arise as a relaxation of the optimal-transport
   variational principle).
4. Empirically, it does not require task-specific tuning beyond $\lambda_J$.

The empirical contribution of this project is precisely to test how well prediction
(1) holds across the smooth/stiff axis — turning the "fix" into a quantitative claim
with a falsifiable interaction prediction.

## 6. Proof obligation for the report

State and justify:

> **Lemma.** Let $f_\theta : \mathbb{R}^d \to \mathbb{R}^d$ be $C^1$. Then for any
> state $x$, the local Lipschitz constant of $f_\theta$ at $x$ in the operator-norm
> sense satisfies $L_\text{loc}(x) = \|J_{f_\theta}(x)\|_2 \leq \|J_{f_\theta}(x)\|_F$.
> Consequently, the soft penalty $\lambda_J \|J\|_F^2$ is a soft upper bound on the
> squared local Lipschitz constant.

Then cite the step-size results T1, T2, T3 from `01_step_size_and_stability.md` to
connect the Lipschitz interpretation back to NFE.
