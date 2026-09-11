"""Visualize the smiley guidance cost function over phi/psi space.

Pure geometry -- no model, no GPU, safe to run alongside anything else. Plots
the exact cost from plot_guided_euler_ramachandran.py's smiley_cost_fn
(within_radius_penalty for the face + EYE_MOUTH_WEIGHT * repel_within_radius_
penalty for each eye/mouth hole, using the periodic torus_distance), importing
that script's live geometry directly rather than duplicating it.

Left panel: full [-pi, pi] Ramachandran domain, showing how small the smiley
region is relative to the whole space (rendered as a colored box, since it is
a few pixels wide at that scale). Right panel: zoomed to the smiley itself.

Run with:
    uv run python tests/guidance/visualization/visualize_smiley_cost.py
"""

from __future__ import annotations

import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from transferable_samplers.guidance.costs import repel_within_radius_penalty, torus_distance, within_radius_penalty

from plot_guided_euler_ramachandran import (
    EYE_CENTERS,
    EYE_MOUTH_WEIGHT,
    EYE_RADIUS,
    FACE_CENTER,
    FACE_RADIUS,
    MOUTH_CENTERS,
    MOUTH_RADIUS,
    PHI_TARGET,
    PSI_TARGET,
)

OUT_PATH = "tests/guidance/out/smiley_cost_landscape.png"


def cost(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    dist_to_face = torus_distance(phi, psi, FACE_CENTER[0], FACE_CENTER[1])
    c = within_radius_penalty(dist_to_face, FACE_RADIUS)
    for cx, cy in EYE_CENTERS:
        d = torus_distance(phi, psi, cx, cy)
        c = c + EYE_MOUTH_WEIGHT * repel_within_radius_penalty(d, EYE_RADIUS)
    for cx, cy in MOUTH_CENTERS:
        d = torus_distance(phi, psi, cx, cy)
        c = c + EYE_MOUTH_WEIGHT * repel_within_radius_penalty(d, MOUTH_RADIUS)
    return c


def grid(phi_range, psi_range, n=500):
    phi = torch.linspace(*phi_range, n)
    psi = torch.linspace(*psi_range, n)
    PHI, PSI = torch.meshgrid(phi, psi, indexing="xy")
    return PHI, PSI, cost(PHI, PSI).numpy()


def main() -> None:
    matplotlib_cmap = "viridis_r"  # sequential, light (zero cost) -> dark (high cost)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 6))

    # --- left: full Ramachandran domain, smiley box highlighted ---
    full_range = (-math.pi, math.pi)
    PHI, PSI, C = grid(full_range, full_range, n=500)
    ax_full.pcolormesh(PHI.numpy(), PSI.numpy(), np.log1p(C), cmap=matplotlib_cmap, shading="auto")
    rect = mpatches.Rectangle(
        (PHI_TARGET[0], PSI_TARGET[0]),
        PHI_TARGET[1] - PHI_TARGET[0],
        PSI_TARGET[1] - PSI_TARGET[0],
        fill=False,
        edgecolor="#E8543A",
        linewidth=1.8,
    )
    ax_full.add_patch(rect)
    ax_full.set_xlim(*full_range)
    ax_full.set_ylim(*full_range)
    ax_full.set_xlabel(r"$\varphi$", fontsize=16)
    ax_full.set_ylabel(r"$\psi$", fontsize=16)
    ax_full.set_title("Full Ramachandran domain\n(red box = smiley region, shown right)", fontsize=11)
    ax_full.set_aspect("equal")

    # --- right: zoomed smiley ---
    margin = 0.25 * FACE_RADIUS
    phi_range = (FACE_CENTER[0] - FACE_RADIUS - margin, FACE_CENTER[0] + FACE_RADIUS + margin)
    psi_range = (FACE_CENTER[1] - FACE_RADIUS - margin, FACE_CENTER[1] + FACE_RADIUS + margin)
    PHI, PSI, C = grid(phi_range, psi_range, n=600)
    im = ax_zoom.pcolormesh(PHI.numpy(), PSI.numpy(), C, cmap=matplotlib_cmap, norm=LogNorm(vmin=1e-4, vmax=C.max()), shading="auto")
    ax_zoom.set_xlim(*phi_range)
    ax_zoom.set_ylim(*psi_range)
    ax_zoom.set_xlabel(r"$\varphi$", fontsize=16)
    ax_zoom.set_ylabel(r"$\psi$", fontsize=16)
    ax_zoom.set_title("Smiley cost (zoomed): zero inside face, penalized in eyes/mouth", fontsize=11)
    ax_zoom.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax_zoom, fraction=0.046, pad=0.04)
    cbar.set_label("cost (log scale)", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
