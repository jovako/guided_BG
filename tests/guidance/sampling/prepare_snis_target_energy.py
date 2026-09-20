"""Combine the density-sampling pool's chunk files into one dataset, adding
the smiley-region membership mask and the combined target energy needed for
SNIS reweighting.

Nothing is discarded -- every sample is kept, with a boolean `in_smiley_mask`
for downstream subsetting alongside `valid` (orientation-preserving).

Reweighting with just the original (unconstrained) MD energy would undo the
guidance and recover the unconstrained equilibrium ensemble. Adding the
smiley cost function's own potential (zero inside the region, growing
outside) instead gives `energy_combined = energy_md + smiley_cost`, so
`logw = neg_logq - energy_combined` reweights toward the constrained
distribution.

Run with:
    uv run python tests/guidance/sampling/prepare_snis_target_energy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, "tests/guidance/visualization")

from transferable_samplers.guidance.costs import repel_within_radius_penalty, torus_distance, within_radius_penalty
from transferable_samplers.guidance.observables import dihedrals, get_dihedral_atom_indices

from plot_guided_euler_ramachandran import (
    EYE_CENTERS,
    EYE_MOUTH_WEIGHT,
    EYE_RADIUS,
    FACE_CENTER,
    FACE_RADIUS,
    MOUTH_CENTERS,
    MOUTH_RADIUS,
    OUT_DIR,
    SEQUENCE,
    load_model_and_data,
)

POOL_DIR = Path(f"{OUT_DIR}/snis_pool_smiley")
SAVE_PATH = POOL_DIR / "combined_pool_with_target_energy.pt"


def smiley_cost(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    dist_to_face = torus_distance(phi, psi, FACE_CENTER[0], FACE_CENTER[1])
    cost = within_radius_penalty(dist_to_face, FACE_RADIUS)
    for cx, cy in EYE_CENTERS:
        d = torus_distance(phi, psi, cx, cy)
        cost = cost + EYE_MOUTH_WEIGHT * repel_within_radius_penalty(d, EYE_RADIUS)
    for cx, cy in MOUTH_CENTERS:
        d = torus_distance(phi, psi, cx, cy)
        cost = cost + EYE_MOUTH_WEIGHT * repel_within_radius_penalty(d, MOUTH_RADIUS)
    return cost


def in_smiley_mask_fn(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    dist_to_face = torus_distance(phi, psi, FACE_CENTER[0], FACE_CENTER[1])
    mask = dist_to_face <= FACE_RADIUS
    for cx, cy in EYE_CENTERS:
        d = torus_distance(phi, psi, cx, cy)
        mask = mask & (d > EYE_RADIUS)
    for cx, cy in MOUTH_CENTERS:
        d = torus_distance(phi, psi, cx, cy)
        mask = mask & (d > MOUTH_RADIUS)
    return mask


def main() -> None:
    _, datamodule, num_atoms = load_model_and_data()
    eval_ctx = datamodule.prepare_eval(sequence=SEQUENCE, stage="test")
    phi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="phi")
    psi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="psi")

    chunk_paths = sorted(POOL_DIR.glob("chunk_*.pt"))
    print(f"Found {len(chunk_paths)} chunk files in {POOL_DIR}")

    all_x, all_x_phys, all_neg_logq, all_valid, all_energy_md = [], [], [], [], []
    for p in chunk_paths:
        d = torch.load(p, weights_only=False)
        all_x.append(d["x"])
        all_x_phys.append(d["x_phys"])
        all_neg_logq.append(d["neg_logq"])
        all_valid.append(d["valid"])
        all_energy_md.append(d["energy"])

    x = torch.cat(all_x, dim=0)
    x_phys = torch.cat(all_x_phys, dim=0)  # already chirality-canonical (fixed at save time)
    neg_logq = torch.cat(all_neg_logq, dim=0)
    valid = torch.cat(all_valid, dim=0)
    energy_md = torch.cat(all_energy_md, dim=0)
    n_total = x.shape[0]
    print(f"Total samples: {n_total}, valid (orientation-preserving): {valid.sum().item()}/{n_total}")

    phi = dihedrals(x_phys, phi_idx).squeeze(-1)
    psi = dihedrals(x_phys, psi_idx).squeeze(-1)

    in_smiley = in_smiley_mask_fn(phi, psi)
    cost = smiley_cost(phi, psi)
    energy_combined = energy_md + cost

    n_in_smiley = in_smiley.sum().item()
    n_in_smiley_and_valid = (in_smiley & valid).sum().item()
    n_zero_cost = (cost == 0).sum().item()
    print(f"In smiley region (face, outside eyes/mouth): {n_in_smiley}/{n_total} ({100 * n_in_smiley / n_total:.1f}%)")
    print(f"In smiley region AND valid: {n_in_smiley_and_valid}/{n_total} ({100 * n_in_smiley_and_valid / n_total:.1f}%)")
    print(f"smiley_cost == 0 (sanity check, should equal in-smiley count): {n_zero_cost}/{n_total}")
    print(f"smiley_cost stats: mean={cost.mean().item():.3f} max={cost.max().item():.3f}")
    print(f"energy_md stats: mean={energy_md.mean().item():.3f}  energy_combined stats: mean={energy_combined.mean().item():.3f}")

    torch.save(
        {
            "x": x, "x_phys": x_phys, "neg_logq": neg_logq, "valid": valid,
            "phi": phi, "psi": psi,
            "energy_md": energy_md, "smiley_cost": cost, "energy_combined": energy_combined,
            "in_smiley_mask": in_smiley,
        },
        SAVE_PATH,
    )
    print(f"\nsaved all {n_total} samples (nothing discarded) to {SAVE_PATH}")


if __name__ == "__main__":
    main()
