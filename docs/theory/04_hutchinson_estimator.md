# 04 — Hutchinson estimator for the Jacobian Frobenius norm

The exact Frobenius-norm penalty requires $d$ backward passes per training point.
This note proves the Hutchinson estimator is unbiased, bounds its variance, and
gives the sample size needed for a target relative error.

## 1. Estimator definition

For a matrix $A \in \mathbb{R}^{d \times d}$ and random vector $\epsilon \in \mathbb{R}^d$
with $\mathbb{E}[\epsilon] = 0$ and $\mathbb{E}[\epsilon \epsilon^\top] = I_d$, define
$$\widehat{\|A\|}_F^2 \;:=\; \frac{1}{K} \sum_{k=1}^K \|A^\top \epsilon_k\|^2,$$
where $\epsilon_1, \dots, \epsilon_K$ are i.i.d. samples. Each term requires one
vector–Jacobian product (one backward pass), so the total cost is $K$ backward
passes regardless of $d$.

## 2. Unbiasedness

$$\mathbb{E}_\epsilon\bigl[\|A^\top \epsilon\|^2\bigr]
  = \mathbb{E}_\epsilon\bigl[\epsilon^\top A A^\top \epsilon\bigr]
  = \mathbb{E}_\epsilon\bigl[\text{tr}(A A^\top \epsilon \epsilon^\top)\bigr]
  = \text{tr}\bigl(A A^\top \, \mathbb{E}[\epsilon \epsilon^\top]\bigr)
  = \text{tr}(A A^\top)
  = \|A\|_F^2.$$
So $\mathbb{E}[\widehat{\|A\|}_F^2] = \|A\|_F^2$.

## 3. Variance bound

For Rademacher $\epsilon \in \{-1, +1\}^d$ (each entry i.i.d. $\pm 1$ with
probability $1/2$),
$$\text{Var}\bigl[\widehat{\|A\|}_F^2\bigr]
  = \frac{1}{K} \cdot 2 \sum_{i \neq j} |(A A^\top)_{ij}|^2
  \;\leq\; \frac{2}{K} \|A A^\top\|_F^2.$$
For Gaussian $\epsilon \sim \mathcal{N}(0, I)$ the variance is slightly larger by
constants but the same order: $\text{Var} = \frac{2}{K} \|A A^\top\|_F^2$ exactly.

## 4. Sample size for the regularizer

In our setting $A = J_{f_\theta}(x, t)$ and we want the *gradient* of $\|J\|_F^2$
with respect to $\theta$ to be a useful learning signal, not necessarily the value
itself. Because (a) we sample many trajectory points per minibatch, and (b) the
penalty is averaged over those points, the effective sample size is $K \cdot B$
where $B$ is the minibatch size. So:

- $K = 1$ is the standard choice (Finlay et al. 2020 use $K = 1$).
- $K = 4$ is a safer fallback if training is noisy.
- For *evaluation/diagnostic* reporting in the paper, use exact Jacobians instead.

## 5. Connection to the trace of $A A^\top$

The Hutchinson estimator is most often introduced for trace estimation:
$\text{tr}(M) = \mathbb{E}[\epsilon^\top M \epsilon]$. Setting $M = A A^\top$ gives
$\text{tr}(A A^\top) = \|A\|_F^2$, recovering our estimator. The general result is:

> **Theorem (Hutchinson, 1990).** For $M \in \mathbb{R}^{d \times d}$ and any
> distribution on $\epsilon$ with $\mathbb{E}[\epsilon] = 0$,
> $\mathbb{E}[\epsilon \epsilon^\top] = I$, the estimator
> $\hat T = \frac{1}{K} \sum_k \epsilon_k^\top M \epsilon_k$ satisfies
> $\mathbb{E}[\hat T] = \text{tr}(M)$. Variance is minimised over the class of such
> distributions when $\epsilon$ is Rademacher.

## 6. Implementation note (PyTorch)

```python
def jacobian_frobenius_hutchinson(f, x, t, K=1):
    """One vector-Jacobian product per sample, then sum-of-squares.
    Returns an unbiased estimate of ||J_f(x,t)||_F^2 averaged over a minibatch."""
    # x: (B, d)  requires_grad=True
    total = 0.0
    for _ in range(K):
        eps = torch.randn_like(x)              # or torch.randint -> Rademacher
        v = f(t, x)
        # vector-Jacobian product: eps^T (df/dx)
        (vjp,) = torch.autograd.grad(v, x, grad_outputs=eps, create_graph=True)
        total = total + (vjp ** 2).sum(dim=-1)  # shape (B,)
    return total.mean() / K
```

The `create_graph=True` is required so the penalty is differentiable through to
$\theta$.

## 7. Proof obligation for the report

State Theorem 1 (Hutchinson) above and prove unbiasedness with the four-line trace
identity in §2. Optionally state the Rademacher variance bound from §3 and note that
$K=1$ is the standard choice in the Neural ODE literature.
