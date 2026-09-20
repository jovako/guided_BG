"""Bootstrap uncertainty for a sample mean.

Shared by the rejection-sampling baseline scripts to report an error bar on
mean-energy, not just the point estimate.
"""

from __future__ import annotations

import torch


def bootstrap_mean_ci(values: torch.Tensor, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Bootstrap standard error and percentile confidence interval for the mean.

    Returns:
        Dict with "se", "ci_low", "ci_high", and "analytic_se" (plain std/sqrt(n), for comparison).
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
