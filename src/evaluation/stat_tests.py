"""Statistical tests for paired comparisons across seeds.

The proposal commits to:
    - two-sided Wilcoxon signed-rank for paired comparisons;
    - Bonferroni correction across multiple comparisons within one experiment;
    - rank-biserial correlation as an effect size, reported with the p-value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class WilcoxonResult:
    statistic: float
    pvalue: float
    effect_size: float          # rank-biserial correlation


def wilcoxon_paired(baseline: np.ndarray, treatment: np.ndarray) -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank test on paired (baseline, treatment) seeds.

    Rank-biserial correlation r = (W+ - W-) / (W+ + W-) where W+ and W- are the
    sums of positive and negative signed ranks respectively.
    """
    baseline = np.asarray(baseline, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    diffs = treatment - baseline
    # Drop exact zeros (Wilcoxon's "wilcox" convention).
    nonzero = diffs[diffs != 0]
    if nonzero.size == 0:
        return WilcoxonResult(statistic=0.0, pvalue=1.0, effect_size=0.0)
    ranks = stats.rankdata(np.abs(nonzero))
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    total = w_plus + w_minus
    r_rb = (w_plus - w_minus) / total if total > 0 else 0.0
    result = stats.wilcoxon(baseline, treatment, zero_method="wilcox",
                            alternative="two-sided")
    return WilcoxonResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        effect_size=float(r_rb),
    )


def bonferroni_correct(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Bonferroni multiple-comparison correction.

    Returns booleans: True iff the corresponding hypothesis is rejected at
    family-wise alpha after correction (i.e., p_i < alpha / m).
    """
    m = len(pvalues)
    if m == 0:
        return []
    threshold = alpha / m
    return [p < threshold for p in pvalues]
