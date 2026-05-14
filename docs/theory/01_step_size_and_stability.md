# 01 — Step-size restrictions and the Jacobian

This note establishes the two distinct mechanisms that govern step size in explicit
Runge–Kutta methods, and shows how both depend on the local Jacobian of the vector
field. This is the theoretical backbone of hypotheses H0 and H1.

## 1. Setup

Consider the autonomous ODE
$$\dot x = f(x), \qquad x(0) = x_0, \qquad x : [0, T] \to \mathbb{R}^d.$$
Let $J(x) = \partial f / \partial x \in \mathbb{R}^{d \times d}$ denote the Jacobian.
An explicit Runge–Kutta method of order $p$ produces an approximation $x_{n+1}$ from
$x_n$ with step size $h_n$.

## 2. Mechanism A: Truncation-error step restriction (accuracy)

For a $p$-th order method, the local truncation error satisfies
$$\tau_n = \frac{h_n^{p+1}}{(p+1)!} \, x^{(p+1)}(\xi_n) + O(h_n^{p+2}),$$
where $x^{(p+1)}$ is the $(p+1)$-th time derivative of the true solution. Using
$\dot x = f(x)$ repeatedly,
$$x^{(2)} = J f, \qquad x^{(3)} = (D_x J \cdot f) f + J^2 f, \qquad \dots$$
so $x^{(p+1)}$ is a polynomial in $J$, $D_x J$, and higher derivatives of $f$.
**Bounding $\|J\|$ bounds the truncation error.** An embedded adaptive method (e.g.,
Dormand–Prince) selects the next step size via
$$h_{n+1} = h_n \left( \frac{\text{tol}}{\|\hat e_n\|} \right)^{1/(p+1)},$$
so smaller $\|\hat e_n\| \sim h_n^{p+1} \|J\|^{p}$ permits larger $h_{n+1}$ and fewer
total steps to traverse $[0, T]$.

> **Takeaway (smooth regime).** When the dynamics are non-stiff, NFE scales roughly as
> $\Theta\!\bigl(T \cdot \|J\|^{p/(p+1)} \cdot \text{tol}^{-1/(p+1)}\bigr)$. Reducing
> $\|J\|_F$ via a penalty produces a *modest* power-law reduction in NFE.

## 3. Mechanism B: Linear-stability step restriction (stability)

Linearise the ODE around an operating point $x^\star$:
$$\dot y = J(x^\star)\, y, \qquad y = x - x^\star.$$
Diagonalising $J(x^\star)$ with eigenvalues $\{\lambda_i\}$, the method is stable on
the test equation iff $h \lambda_i \in S$ for all $i$, where $S \subset \mathbb{C}$ is
the method's stability region. For RK4, $S$ contains the strip
$|\text{Re}(h\lambda)| \lesssim 2.78$ on the negative real axis. So the *stable* step
size is
$$h_{\text{stab}} \approx \frac{C_p}{\rho(J)}, \qquad \rho(J) = \max_i |\lambda_i(J)|.$$
On a stiff system, $\rho(J) \gg 1$ even when $\|x^{(p+1)}\|$ is moderate, so the
solver is forced into tiny steps purely for stability — independent of how accurate
those steps would otherwise need to be.

## 4. Why the Jacobian Frobenius norm is the right penalty

The Frobenius norm bounds the spectral radius:
$$\rho(J)^2 \;\leq\; \sum_i |\lambda_i|^2 \;=\; \|J\|_F^2 - (\text{off-diagonal energy}),$$
and the inequality $\rho(J) \leq \|J\|_F$ holds in general (sum of squared singular
values upper-bounds squared spectral radius). So penalising $\|J\|_F^2$ provides:

| Regime | Mechanism limited by | Effect of penalty |
|--------|---------------------|-------------------|
| Smooth (non-stiff) | Truncation error (Mechanism A) | Power-law NFE reduction, $\sim \|J\|^{p/(p+1)}$ |
| Stiff | Stability (Mechanism B) | Direct reduction of $\rho(J)$, lifts a stricter constraint |

## 5. The interaction hypothesis (formal statement of H1)

Let $\Delta\text{NFE}(s) = \text{NFE}_{\text{baseline}}(s) - \text{NFE}_{\text{reg}}(s)$
at matched validation MSE, where $s$ indexes target-system stiffness. Assume an
explicit RK method of order $p$ and that the penalty reduces $\|J\|_F$ by a constant
factor $\beta < 1$. The model in §2–§3 predicts
$$\Delta\text{NFE}_{\text{smooth}} \;\propto\; (1 - \beta^{p/(p+1)}), \qquad
  \Delta\text{NFE}_{\text{stiff}} \;\propto\; (1 - \beta),$$
so $\Delta\text{NFE}_{\text{stiff}} / \Delta\text{NFE}_{\text{smooth}} > 1$ whenever
$p \geq 1$ (with the gap widening for higher-order methods). This is the analytic
prediction the experiments test.

## 6. Theorems to write up (proof obligations for the final report)

These are the formal statements you should state and prove (or cite with proof) in
the report's "Theoretical background" section:

- **T1** (Accuracy step bound). For an order-$p$ embedded RK method with tolerance
  $\text{tol}$ applied to $\dot x = f(x)$ over $[0, T]$, the expected number of steps
  satisfies $N \leq C \cdot T \cdot M^{p/(p+1)} \cdot \text{tol}^{-1/(p+1)}$ where
  $M \geq \sup_t \|J(x(t))\|$ and $C$ depends on the method. *Status: well-known;
  cite Hairer–Nørsett–Wanner Vol. I, Ch. II.*
- **T2** (Stability step bound). For the same method, stability requires
  $h \cdot \rho(J(x(t))) \leq C_p$. Equivalently, $N \geq T \cdot \rho(J) / C_p$.
  *Status: well-known; cite Hairer–Wanner Vol. II, Ch. IV.*
- **T3** (Frobenius dominates spectral). $\rho(J) \leq \|J\|_F$ with equality iff
  $J$ is rank one (or normal with all eigenvalue mass on one eigenvalue). *Proof:
  one-line, from $\rho(J)^2 \leq \sum_i \sigma_i(J)^2 = \|J\|_F^2$.*
- **T4** (Interaction prediction). Assuming the regularizer reduces $\|J\|_F$ by a
  multiplicative factor and the smooth task is accuracy-limited while the stiff task
  is stability-limited, $\Delta\text{NFE}_\text{stiff} > \Delta\text{NFE}_\text{smooth}$.
  *Status: novel synthesis — this is yours to state and justify from T1–T3.*

T4 is the "theory" contribution that motivates the empirical work. The experiments
then test whether the assumptions of T4 hold in practice on learned Neural ODEs.
