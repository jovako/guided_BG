"""Build the 250-state linear-coupling interpolation path x_t = (1-t)*x0 + t*x1
between a real accepted (rejection-sampled, unguided) sample near the smiley
center and a Gaussian noise draw, and save every intermediate state.

x1 is whichever rejection_smiley_samples.pt sample sits closest to
FACE_CENTER; x0 ~ model.prior. Pure interpolation arithmetic, no network call.

Run with:
    uv run python tests/guidance/integrator_checks/linear_coupling_trajectory_smiley_center.py
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "tests/guidance/visualization")

from transferable_samplers.guidance.costs import torus_distance
from transferable_samplers.guidance.observables import dihedrals, get_dihedral_atom_indices

from plot_guided_euler_ramachandran import FACE_CENTER, FACE_RADIUS, OUT_DIR, SEQUENCE, load_model_and_data

SEED = 42
N_STATES = 250
REJECTION_SAMPLES_PATH = f"{OUT_DIR}/rejection_smiley_samples.pt"
SAVE_PATH = f"{OUT_DIR}/linear_coupling_trajectory_smiley_center.pt"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, datamodule, num_atoms = load_model_and_data()
    model = model.to(device).eval()

    eval_ctx = datamodule.prepare_eval(sequence=SEQUENCE, stage="test")
    phi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="phi")
    psi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="psi")

    rejection_data = torch.load(REJECTION_SAMPLES_PATH, weights_only=False)
    samples_physical = rejection_data["samples_physical"]  # (N, atoms, 3), already accepted + chirality-canonical
    phi = dihedrals(samples_physical, phi_idx).squeeze(-1)
    psi = dihedrals(samples_physical, psi_idx).squeeze(-1)
    dist = torus_distance(phi, psi, FACE_CENTER[0], FACE_CENTER[1])
    best_idx = torch.argmin(dist).item()
    print(
        f"Closest rejection sample to FACE_CENTER: idx={best_idx}, "
        f"phi/psi=({phi[best_idx]:.4f}, {psi[best_idx]:.4f}), dist={dist[best_idx]:.4f} "
        f"(FACE_RADIUS={FACE_RADIUS:.4f})"
    )

    x1_phys = samples_physical[best_idx : best_idx + 1].to(device)
    x1 = x1_phys / eval_ctx.normalization_std  # to normalized space

    torch.manual_seed(SEED)
    x0 = model.prior.sample(1, num_atoms, device=device)

    t_values = torch.linspace(0.0, 1.0, N_STATES, device=device)
    # (N_STATES, 1, 1) broadcasts against x0/x1's (1, atoms, dims)
    t_bcast = t_values.view(-1, 1, 1)
    x_t_all = (1.0 - t_bcast) * x0 + t_bcast * x1  # (N_STATES, atoms, dims)

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
