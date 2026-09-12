"""Second reference trajectory (see linear_coupling_trajectory_smiley_center.py),
built the same way but with a DIFFERENT rejection sample (the 2nd-closest to
FACE_CENTER, not the closest) and a different noise seed -- to test whether
smiley_reference's high failure rate was specific to the first reference
sample (bad luck / a numerically fragile point) or a general property of the
reference-tracking approach.

Run with:
    uv run python tests/guidance/integrator_checks/linear_coupling_trajectory_smiley_center_2.py
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "tests/guidance/visualization")

from transferable_samplers.guidance.costs import torus_distance
from transferable_samplers.guidance.observables import dihedrals, get_dihedral_atom_indices

from plot_guided_euler_ramachandran import FACE_CENTER, FACE_RADIUS, OUT_DIR, SEQUENCE, load_model_and_data

SEED = 43  # different from the first reference's seed=42
N_STATES = 250
RANK = 1  # 0 = closest (already used), 1 = second-closest
REJECTION_SAMPLES_PATH = f"{OUT_DIR}/rejection_smiley_samples.pt"
SAVE_PATH = f"{OUT_DIR}/linear_coupling_trajectory_smiley_center_2.pt"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, datamodule, num_atoms = load_model_and_data()
    model = model.to(device).eval()

    eval_ctx = datamodule.prepare_eval(sequence=SEQUENCE, stage="test")
    phi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="phi")
    psi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="psi")

    rejection_data = torch.load(REJECTION_SAMPLES_PATH, weights_only=False)
    samples_physical = rejection_data["samples_physical"]
    phi = dihedrals(samples_physical, phi_idx).squeeze(-1)
    psi = dihedrals(samples_physical, psi_idx).squeeze(-1)
    dist = torus_distance(phi, psi, FACE_CENTER[0], FACE_CENTER[1])
    ranked = torch.argsort(dist)
    best_idx = ranked[RANK].item()
    print(
        f"Rank-{RANK} closest rejection sample to FACE_CENTER: idx={best_idx}, "
        f"phi/psi=({phi[best_idx]:.4f}, {psi[best_idx]:.4f}), dist={dist[best_idx]:.4f} "
        f"(FACE_RADIUS={FACE_RADIUS:.4f}) -- rank-0 (idx={ranked[0].item()}) was the first reference's sample."
    )

    x1_phys = samples_physical[best_idx : best_idx + 1].to(device)
    x1 = x1_phys / eval_ctx.normalization_std

    torch.manual_seed(SEED)
    x0 = model.prior.sample(1, num_atoms, device=device)

    t_values = torch.linspace(0.0, 1.0, N_STATES, device=device)
    t_bcast = t_values.view(-1, 1, 1)
    x_t_all = (1.0 - t_bcast) * x0 + t_bcast * x1

    recon_err = (x_t_all[-1] - x1[0]).abs().max().item()
    print(f"endpoint check: max|x_t[t=1] - x1| = {recon_err:.2e} (should be ~0)")

    torch.save(
        {
            "t_values": t_values.cpu(), "x_t_all": x_t_all.detach().cpu(),
            "x0": x0.detach().cpu(), "x1": x1.detach().cpu(), "x1_phys": x1_phys.detach().cpu(),
            "rejection_idx": best_idx, "phi_psi_x1": (phi[best_idx].item(), psi[best_idx].item()),
            "dist_to_center": dist[best_idx].item(), "seed": SEED, "n_states": N_STATES,
        },
        SAVE_PATH,
    )
    print(f"\nsaved {N_STATES} states to {SAVE_PATH}")


if __name__ == "__main__":
    main()
