"""Bootstrap uncertainty for a sample mean.

Shared by the rejection-sampling baseline scripts to report an error bar on
mean-energy, not just the point estimate. Pure CPU statistics on an already
-computed 1D tensor of per-sample values -- no model or GPU involved.
"""

from __future__ import annotations

import torch


def bootstrap_mean_ci(values: torch.Tensor, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Bootstrap standard error and percentile confidence interval for the mean.

    Args:
        values: 1D tensor of per-sample values (e.g. energies).
        n_boot: Number of bootstrap resamples.
        ci: Confidence level for the percentile interval (e.g. 0.95 for a 95% CI).
        seed: RNG seed, for reproducibility across runs.

    Returns:
        Dict with "se" (bootstrap standard error of the mean), "ci_low", "ci_high"
        (the percentile confidence interval bounds), and "analytic_se" (the
        plain std/sqrt(n) formula, for comparison -- the two should agree
        closely unless the distribution is notably skewed/heavy-tailed).
    """
    values = values.flatten().float()
    n = values.shape[0]
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (n_boot, n), generator=generator)
    boot_means = values[idx].mean(dim=1)

    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    ci_low, ci_high = torch.quantile(boot_means, torch.tensor([lo_q, hi_q])).tolist()

    analytic_se = (values.std(unbiased=True) / n**0.5).item()

    return {"se": boot_means.std().item(), "ci_low": ci_low, "ci_high": ci_high, "analytic_se": analytic_se}
