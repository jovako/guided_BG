"""Cost-shaping helpers for guidance terminal costs.

These turn an observable value into a per-sample scalar cost to hand to
``FlowMatchingModule.guidance_cost_fn``. Kept separate from the observable
computation (``observables.py``) so the same shaping (e.g. "only penalize
values on one side of a threshold") can be reused across different
observables (phi, psi, distances, ...).
"""

from __future__ import annotations

import torch


def one_sided_quadratic_penalty(value: torch.Tensor, threshold: float = 0.0, penalize_below: bool = True) -> torch.Tensor:
    """Quadratic penalty on one side of a threshold, zero on the other.

    Args:
        value: Observable value(s), any shape.
        threshold: The value below/above which the penalty kicks in.
        penalize_below: If True, cost is ``relu(threshold - value)**2``
            (penalizes ``value < threshold``, zero for ``value >= threshold``).
            If False, cost is ``relu(value - threshold)**2`` (penalizes
            ``value > threshold``).

    Returns:
        Per-element cost, same shape as ``value``.
    """
    margin = (threshold - value) if penalize_below else (value - threshold)
    return torch.relu(margin) ** 2
