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


def box_quadratic_penalty(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Zero cost inside ``[low, high]``, quadratic penalty outside on either side.

    Useful for pulling an observable into a target region (e.g. a box in
    phi/psi space) rather than just past a single threshold -- guidance then
    concentrates density into wherever the cost is zero.

    Args:
        value: Observable value(s), any shape.
        low: Lower edge of the zero-cost region.
        high: Upper edge of the zero-cost region.

    Returns:
        Per-element cost, same shape as ``value``.
    """
    return torch.relu(low - value) ** 2 + torch.relu(value - high) ** 2


def wrapped_angle_diff(a: torch.Tensor, b: float) -> torch.Tensor:
    """Signed difference ``a - b``, wrapped into ``(-pi, pi]`` -- the shortest
    arc between two angles on a circle, rather than their raw numeric gap.
    """
    diff = a - b
    return diff - 2 * torch.pi * torch.round(diff / (2 * torch.pi))


def torus_distance(phi: torch.Tensor, psi: torch.Tensor, center_phi: float, center_psi: float) -> torch.Tensor:
    """Euclidean distance in (phi, psi) space, with each axis wrapped
    periodically (both are angles on a circle, ``[-pi, pi]``).

    Use this instead of a plain ``sqrt((phi-cx)**2 + (psi-cy)**2)`` for any
    target region -- otherwise a region whose radius crosses the +-pi seam on
    either axis is silently cut in half, and points near the seam that are
    actually close to the target (via wraparound) read as far away.

    Args:
        phi, psi: Angle value(s) in radians, any shape (broadcastable together).
        center_phi, center_psi: Target center, in radians.

    Returns:
        Distance(s), same shape as ``phi``/``psi``.
    """
    dphi = wrapped_angle_diff(phi, center_phi)
    dpsi = wrapped_angle_diff(psi, center_psi)
    return torch.sqrt(dphi**2 + dpsi**2)


def quadratic_target_penalty(distance: torch.Tensor) -> torch.Tensor:
    """Quadratic bowl centered on a target point: ``distance**2``, zero only
    where ``distance == 0``.

    Pair with ``torus_distance`` for a periodic single-point target (e.g.
    pull toward one exact (phi, psi) rather than a region) -- unlike
    ``within_radius_penalty``/``box_quadratic_penalty``, there's no flat
    zero-cost region, just a single zero-cost point.

    Args:
        distance: Distance(s) from the target point, any shape.

    Returns:
        Per-element cost, same shape as ``distance``.
    """
    return distance**2


def within_radius_penalty(distance: torch.Tensor, radius: float) -> torch.Tensor:
    """Zero cost within ``radius``, quadratic penalty growing outside it.

    An "attractive" containment constraint (e.g. keep a point inside a circle)
    -- pair with a distance computed from some center.

    Args:
        distance: Distance(s) from a reference point, any shape.
        radius: Distance below which cost is zero.

    Returns:
        Per-element cost, same shape as ``distance``.
    """
    return torch.relu(distance - radius) ** 2


def repel_within_radius_penalty(distance: torch.Tensor, radius: float) -> torch.Tensor:
    """Zero cost outside ``radius``, quadratic penalty growing as distance shrinks inside it.

    A "repulsive" exclusion-zone constraint (e.g. push a point away from a
    small circle) -- the mirror image of ``within_radius_penalty``.

    Args:
        distance: Distance(s) from a reference point, any shape.
        radius: Distance within which cost grows (a repulsive bump centered at distance=0).

    Returns:
        Per-element cost, same shape as ``distance``.
    """
    return torch.relu(radius - distance) ** 2
