"""Find one unguided velocity vector v_t at t=0.9 whose linear endpoint
extrapolation x1_hat = x_t + 0.1*v_t lands near the smiley's FACE_CENTER.

Draws a batch of prior samples, integrates them unguided (fixed-step Euler)
up to t=0.9, evaluates the network's velocity there once, and picks whichever
sample's x1_hat is closest (periodic torus_distance) to FACE_CENTER.

Run with:
    uv run python tests/guidance/integrator_checks/find_velocity_toward_smiley_center.py
"""

from __future__ import annotations

import copy
import sys

import torch

sys.path.insert(0, "tests/guidance/visualization")

from transferable_samplers.guidance.costs import torus_distance
from transferable_samplers.guidance.euler_density_integrator import make_guided_euler_step
from transferable_samplers.guidance.observables import dihedrals, get_dihedral_atom_indices
from transferable_samplers.utils.chirality import ChiralitySignChecker

from plot_guided_euler_ramachandran import FACE_CENTER, FACE_RADIUS, OUT_DIR, SEQUENCE, load_model_and_data

BATCH = 256
SEED = 42
EULER_STEPS = 250  # t=0.9 falls exactly on a step boundary
T_TARGET = 0.9
STEPS_TO_T = round(T_TARGET * EULER_STEPS)  # 225
SAVE_PATH = f"{OUT_DIR}/velocity_near_smiley_center_t0.9.pt"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, datamodule, num_atoms = load_model_and_data()
    model = model.to(device).eval()

    eval_ctx = datamodule.prepare_eval(sequence=SEQUENCE, stage="test")
    phi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="phi")
    psi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="psi")
    chirality_checker = ChiralitySignChecker(eval_ctx.topology, eval_ctx.true_data.samples[:1])

    net = copy.deepcopy(model.net)
    net.requires_grad_(False)
    dt = 1.0 / EULER_STEPS

    # n_inner=0 skips the guidance inner loop entirely -- plain unguided step.
    step = make_guided_euler_step(
        net, None, terminal_cost=lambda x1, t: (x1 * 0.0).sum(),
        dt=dt, gamma=0.0, alpha=0.0, n_inner=0, use_score_deviation=False,
    )

    torch.manual_seed(SEED)
    z = model.prior.sample(BATCH, num_atoms, device=device)
    x = z.reshape(BATCH, -1)

    print(f"Integrating {STEPS_TO_T} unguided Euler steps to t={T_TARGET} ({BATCH} samples)...")
    with torch.no_grad():
        for k in range(STEPS_TO_T):
            t = torch.as_tensor(k * dt, device=device, dtype=x.dtype)
            x = step(t, x)
        t_final = torch.as_tensor(STEPS_TO_T * dt, device=device, dtype=x.dtype)
        v = net(t_final.reshape(1), x, encodings=None)

    x1_hat = x + (1.0 - T_TARGET) * v  # linear extrapolation to t=1, same as the guidance inner objective

    x1_hat_atoms = x1_hat.reshape(BATCH, num_atoms, -1).detach()
    with torch.no_grad():
        flip_mask = chirality_checker.flip_mask(x1_hat_atoms)
    sign = torch.where(flip_mask, -1.0, 1.0).to(x1_hat_atoms)[:, None, None]
    x1_hat_fixed = x1_hat_atoms * sign
    phi = dihedrals(x1_hat_fixed, phi_idx).squeeze(-1)
    psi = dihedrals(x1_hat_fixed, psi_idx).squeeze(-1)

    dist = torus_distance(phi, psi, FACE_CENTER[0], FACE_CENTER[1])
    best_idx = torch.argmin(dist).item()

    print(f"\nBest of {BATCH}: sample {best_idx}")
    print(f"  x1_hat phi/psi = ({phi[best_idx].item():.4f}, {psi[best_idx].item():.4f})")
    print(f"  FACE_CENTER    = {FACE_CENTER}, FACE_RADIUS = {FACE_RADIUS:.4f}")
    print(f"  distance to center = {dist[best_idx].item():.4f} (inside face: {dist[best_idx].item() <= FACE_RADIUS})")
    print(f"  ||v_t|| = {v[best_idx].norm().item():.4f}")

    torch.save(
        {
            "t": T_TARGET, "x_t": x[best_idx].detach().cpu(), "v_t": v[best_idx].detach().cpu(),
            "x1_hat": x1_hat[best_idx].detach().cpu(), "phi_psi_x1_hat": (phi[best_idx].item(), psi[best_idx].item()),
            "dist_to_center": dist[best_idx].item(), "seed": SEED, "batch_index": best_idx,
            "euler_steps": EULER_STEPS, "steps_to_t": STEPS_TO_T,
        },
        SAVE_PATH,
    )
    print(f"\nsaved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
