"""GIF of a reference (noised) sample's own model prediction as it moves under
the linear coupling schedule x_t = (1-t)*x0 + t*x1, t: 0 -> 1.

Unlike gif_noised_target_trajectory_phi_psi.py (raw noised point x_t's own
dihedral angles, no network involved), this plots phi/psi of the network's
endpoint prediction x1_hat = x_t + (1-t)*v_theta(t, x_t) -- the same quantity
the "smiley_reference" guidance cost tracks.

Set REFERENCE_PATH below to whichever saved reference file to visualize.

Run with:
    uv run python tests/guidance/visualization/gif_reference_prediction_trajectory_phi_psi.py
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
    OUT_DIR, SEQUENCE, load_model_and_data,
)

REFERENCE_PATH = f"{OUT_DIR}/linear_coupling_trajectory_smiley_center_2.pt"
GIF_PATH = f"{OUT_DIR}/reference_prediction_trajectory_phi_psi_2.gif"
N_FRAMES = 100  # subsampled from the reference's own N_STATES

_FULL_RANGE = (-math.pi, math.pi)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, datamodule, num_atoms = load_model_and_data()
    model = model.to(device).eval()

    eval_ctx = datamodule.prepare_eval(sequence=SEQUENCE, stage="test")
    phi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="phi")
    psi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="psi")
    chirality_checker = ChiralitySignChecker(eval_ctx.topology, eval_ctx.true_data.samples[:1])

    ref_data = torch.load(REFERENCE_PATH, weights_only=False)
    x0_ref = ref_data["x0"].to(device)
    x1_ref = ref_data["x1"].to(device)
    t_values_all = ref_data["t_values"]
    print(f"Reference: rejection_idx={ref_data['rejection_idx']}, phi_psi_x1={ref_data['phi_psi_x1']}, "
          f"seed={ref_data['seed']}")

    idx = np.linspace(0, len(t_values_all) - 1, N_FRAMES, dtype=int)
    t_values = t_values_all[idx]

    phi_list, psi_list = [], []
    print(f"Evaluating the network's endpoint prediction at {len(t_values)} t values...")
    with torch.no_grad():
        for t_val in t_values:
            t = torch.as_tensor(t_val, device=device, dtype=x0_ref.dtype)
            x_t_ref = (1.0 - t) * x0_ref + t * x1_ref
            v_ref = model.net(t.reshape(1), x_t_ref.reshape(1, -1), encodings=None).reshape_as(x_t_ref)
            x1_hat_ref = x_t_ref + (1.0 - t) * v_ref
            flip_ref = chirality_checker.flip_mask(x1_hat_ref)
            sign_ref = torch.where(flip_ref, -1.0, 1.0).to(x1_hat_ref)[:, None, None]
            x1_hat_ref_fixed = x1_hat_ref * sign_ref
            phi_ref = dihedrals(x1_hat_ref_fixed, phi_idx).squeeze(-1)
            psi_ref = dihedrals(x1_hat_ref_fixed, psi_idx).squeeze(-1)
            phi_list.append(phi_ref.item())
            psi_list.append(psi_ref.item())
    phi = np.array(phi_list)
    psi = np.array(psi_list)
    print(f"t=0.000: phi/psi=({phi[0]:.3f}, {psi[0]:.3f})   t=1.000: phi/psi=({phi[-1]:.3f}, {psi[-1]:.3f})")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(*_FULL_RANGE)
    ax.set_ylim(*_FULL_RANGE)
    ax.set_xlabel(r"$\varphi$", fontsize=14)
    ax.set_ylabel(r"$\psi$", fontsize=14)
    ax.set_aspect("equal")

    ax.add_patch(Circle(FACE_CENTER, FACE_RADIUS, fill=False, edgecolor="black", linewidth=1.2, zorder=2))
    for cx, cy in EYE_CENTERS:
        ax.add_patch(Circle((cx, cy), EYE_RADIUS, fill=False, edgecolor="black", linewidth=1.0, zorder=2))
    for cx, cy in MOUTH_CENTERS:
        ax.add_patch(Circle((cx, cy), MOUTH_RADIUS, fill=False, edgecolor="black", linewidth=1.0, zorder=2))

    # Break the static path at +-pi wraparound crossings so we don't draw a
    # spurious line across the whole plot.
    dphi = np.abs(np.diff(phi))
    dpsi = np.abs(np.diff(psi))
    wrap = (dphi > math.pi) | (dpsi > math.pi)
    phi_path = phi.copy()
    psi_path = psi.copy()
    phi_path[np.append(wrap, False)] = np.nan
    ax.plot(phi_path, psi_path, color="tab:blue", alpha=0.35, linewidth=1.0, zorder=1)

    trail, = ax.plot([], [], color="tab:blue", alpha=0.6, linewidth=1.5, zorder=4)
    point = ax.scatter([], [], s=90, c="tab:blue", marker="*", edgecolors="black", zorder=5)
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
