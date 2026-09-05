"""Differentiable collective-variable observables for guided sampling.

The evaluation-side phi/psi code (``evaluation/plots/plot_ramachandran.py``,
``evaluation/metrics/wasserstein_distances.py``, ``data/preprocessing/tica.py``)
computes dihedral angles via ``mdtraj``, which operates on unbatched numpy
trajectories and is not differentiable. Guided sampling needs the gradient of
an observable with respect to the (batched) atom positions, so this module
reimplements the dihedral angle as a pure-torch, batched, differentiable
function instead.

The atom *indices* that define a dihedral (e.g. which four atoms form phi for
a given residue) only depend on the topology's bond graph, not on coordinates,
so those are still looked up once via ``mdtraj`` (see
``get_dihedral_atom_indices``) and then baked into a guidance cost closure --
no ``mdtraj`` calls happen inside the sampling loop.
"""

from __future__ import annotations

from typing import Literal

import mdtraj as md
import numpy as np
import torch

_COMPUTE_FN = {
    "phi": md.compute_phi,
    "psi": md.compute_psi,
    "omega": md.compute_omega,
}


def get_dihedral_atom_indices(topology: md.Topology, kind: Literal["phi", "psi", "omega"] = "phi") -> np.ndarray:
    """Return the atom index quadruples ``mdtraj`` uses for a dihedral kind on this topology.

    Coordinates don't affect which atoms form each dihedral -- ``mdtraj``
    derives the quadruples purely from the topology's bond graph, so a dummy
    single-frame trajectory (zeros) is built just to satisfy its API; only the
    returned indices are used, never the (meaningless) angle values.

    Args:
        topology: mdtraj topology (e.g. ``datamodule.topology``).
        kind: One of "phi", "psi", "omega".

    Returns:
        Atom index quadruples, shape ``(num_dihedrals, 4)``. For phi this is
        ``[C(i-1), N(i), CA(i), C(i)]`` per residue, matching the atom order
        expected by ``dihedrals()``.
    """
    dummy_traj = md.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology=topology)
    indices, _ = _COMPUTE_FN[kind](dummy_traj)
    return indices


def dihedral_angle(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """Batched, differentiable dihedral angle in radians, ``mdtraj``-compatible sign convention.

    Args:
        p0, p1, p2, p3: Positions of the four atoms defining the dihedral,
            each ``(..., 3)``. For phi: p0=C(i-1), p1=N(i), p2=CA(i), p3=C(i).

    Returns:
        Dihedral angle(s) in ``(-pi, pi]``, shape ``(...,)``.
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1 = b1 / b1.norm(dim=-1, keepdim=True)

    # project b0 and b2 into the plane perpendicular to b1
    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1

    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


def dihedrals(x: torch.Tensor, atom_indices: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Batched dihedral angles for one or more atom quadruples.

    Args:
        x: Positions ``(batch, atoms, 3)``.
        atom_indices: Atom index quadruples ``(num_dihedrals, 4)``, e.g. from
            ``get_dihedral_atom_indices``.

    Returns:
        Dihedral angles ``(batch, num_dihedrals)`` in radians.
    """
    atom_indices = torch.as_tensor(atom_indices, device=x.device, dtype=torch.long)
    p = x[:, atom_indices]  # (batch, num_dihedrals, 4, 3)
    return dihedral_angle(p[..., 0, :], p[..., 1, :], p[..., 2, :], p[..., 3, :])
