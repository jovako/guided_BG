"""Cost-shaping helpers for guidance terminal costs.

Turn an observable value into a per-sample scalar cost, for use inside a
``terminal_cost`` callback passed to ``euler_density_integrator``. Kept
separate from the observable computation (``observables.py``) so the same
shaping can be reused across different observables (phi, psi, distances, ...).
"""

from __future__ import annotations

import torch


def one_sided_quadratic_penalty(value: torch.Tensor, threshold: float = 0.0, penalize_below: bool = True) -> torch.Tensor:
    """Quadratic penalty on one side of ``threshold``, zero on the other."""
    margin = (threshold - value) if penalize_below else (value - threshold)
    return torch.relu(margin) ** 2


def box_quadratic_penalty(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Zero cost inside ``[low, high]``, quadratic penalty outside on either side."""
    return torch.relu(low - value) ** 2 + torch.relu(value - high) ** 2


def wrapped_angle_diff(a: torch.Tensor, b: float) -> torch.Tensor:
    """Signed difference ``a - b``, wrapped into ``(-pi, pi]`` (shortest arc on a circle)."""
    diff = a - b
    return diff - 2 * torch.pi * torch.round(diff / (2 * torch.pi))


def torus_distance(phi: torch.Tensor, psi: torch.Tensor, center_phi: float, center_psi: float) -> torch.Tensor:
    """Euclidean distance in (phi, psi) space, each axis wrapped periodically on ``[-pi, pi]``."""
    dphi = wrapped_angle_diff(phi, center_phi)
    dpsi = wrapped_angle_diff(psi, center_psi)
    return torch.sqrt(dphi**2 + dpsi**2)


def quadratic_target_penalty(distance: torch.Tensor) -> torch.Tensor:
    """Quadratic bowl centered on a target point: ``distance**2``. Pair with ``torus_distance``."""
    return distance**2


def within_radius_penalty(distance: torch.Tensor, radius: float) -> torch.Tensor:
    """Zero cost within ``radius``, quadratic penalty growing outside it (attractive containment)."""
    return torch.relu(distance - radius) ** 2


def repel_within_radius_penalty(distance: torch.Tensor, radius: float) -> torch.Tensor:
    """Zero cost outside ``radius``, quadratic penalty growing inside it (repulsive exclusion zone)."""
    return torch.relu(radius - distance) ** 2
