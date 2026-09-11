"""GIF of the noised smiley-center reference sample's OWN phi/psi as it moves
under the linear coupling schedule x_t = (1-t)*x0 + t*x1, t: 0 -> 1.

Uses the 250 states already saved by linear_coupling_trajectory_smiley_center.py
(tests/guidance/out/linear_coupling_trajectory_smiley_center.pt) -- this is the
RAW interpolated point's own dihedral angles at each t, no network involved
(dihedral angles are scale-invariant, so computing them directly on the
normalized-space x_t is the same as on destandardized coordinates, same
convention the guidance cost functions already use). At t=0 this is a
(mostly meaningless) dihedral reading of pure Gaussian noise; at t=1 it's
exactly the real rejection-sampled smiley-center point (x_t_all[-1] == x1 by
construction, verified in that script).

Not to be confused with the "current prediction" phi/psi used by the
smiley_reference guidance cost (the network's endpoint prediction AT each
noised state) -- this plots the noised state's own raw angle, no v_theta call.

Run with:
    uv run python tests/guidance/visualization/gif_noised_target_trajectory_phi_psi.py
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from transferable_samplers.guidance.observables import dihedrals, get_dihedral_atom_indices
from transferable_samplers.utils.chirality import ChiralitySignChecker

from plot_guided_euler_ramachandran import (
    EYE_CENTERS, EYE_RADIUS, FACE_CENTER, FACE_RADIUS, MOUTH_CENTERS, MOUTH_RADIUS,
    OUT_DIR, REFERENCE_TRAJECTORY_PATH, SEQUENCE, load_model_and_data,
)

GIF_PATH = f"{OUT_DIR}/noised_target_trajectory_phi_psi.gif"
_FULL_RANGE = (-math.pi, math.pi)


def main() -> None:
    model, datamodule, num_atoms = load_model_and_data()

    eval_ctx = datamodule.prepare_eval(sequence=SEQUENCE, stage="test")
    phi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="phi")
    psi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="psi")
    chirality_checker = ChiralitySignChecker(eval_ctx.topology, eval_ctx.true_data.samples[:1])

    ref_data = torch.load(REFERENCE_TRAJECTORY_PATH, weights_only=False)
    x_t_all = ref_data["x_t_all"]  # (N_STATES, atoms, dims), normalized space
    t_values = ref_data["t_values"]  # (N_STATES,)

    with torch.no_grad():
        flip_mask = chirality_checker.flip_mask(x_t_all)
    sign = torch.where(flip_mask, -1.0, 1.0).to(x_t_all)[:, None, None]
    x_t_fixed = x_t_all * sign
    phi = dihedrals(x_t_fixed, phi_idx).squeeze(-1).numpy()
    psi = dihedrals(x_t_fixed, psi_idx).squeeze(-1).numpy()
    t_values = t_values.numpy()

    print(f"Loaded {len(t_values)} states from {REFERENCE_TRAJECTORY_PATH}")
    print(f"t=0.000: phi/psi=({phi[0]:.3f}, {psi[0]:.3f})   t=1.000: phi/psi=({phi[-1]:.3f}, {psi[-1]:.3f})")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(*_FULL_RANGE)
    ax.set_ylim(*_FULL_RANGE)
    ax.set_xlabel(r"$\varphi$", fontsize=14)
    ax.set_ylabel(r"$\psi$", fontsize=14)
    ax.set_aspect("equal")

    # Smiley outline for reference -- x1 (t=1) is the rejection sample closest
    # to FACE_CENTER, so the path should end right at the face.
    ax.add_patch(Circle(FACE_CENTER, FACE_RADIUS, fill=False, edgecolor="black", linewidth=1.2, zorder=2))
    for cx, cy in EYE_CENTERS:
        ax.add_patch(Circle((cx, cy), EYE_RADIUS, fill=False, edgecolor="black", linewidth=1.0, zorder=2))
    for cx, cy in MOUTH_CENTERS:
        ax.add_patch(Circle((cx, cy), MOUTH_RADIUS, fill=False, edgecolor="black", linewidth=1.0, zorder=2))

    # Full path drawn faintly up front (deterministic, known ahead of time) --
    # careful with the +-pi wraparound: don't draw a spurious line across the
    # whole plot when the path crosses the seam.
    dphi = np.abs(np.diff(phi))
    dpsi = np.abs(np.diff(psi))
    wrap = (dphi > math.pi) | (dpsi > math.pi)
    phi_path = phi.copy()
    psi_path = psi.copy()
    phi_path[np.append(wrap, False)] = np.nan
    ax.plot(phi_path, psi_path, color="gray", alpha=0.4, linewidth=1.0, zorder=1)

    trail, = ax.plot([], [], color="red", alpha=0.5, linewidth=1.5, zorder=4)
    point = ax.scatter([], [], s=60, c="red", edgecolors="black", zorder=5)
    title = ax.set_title("")

    def update(frame_idx: int):
        trail.set_data(phi_path[: frame_idx + 1], psi_path[: frame_idx + 1])
        point.set_offsets([[phi[frame_idx], psi[frame_idx]]])
        title.set_text(f"t={t_values[frame_idx]:.3f}")
        return trail, point, title

    anim = FuncAnimation(fig, update, frames=len(t_values), interval=60, blit=False)
    anim.save(GIF_PATH, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"saved {GIF_PATH}")


if __name__ == "__main__":
    main()
