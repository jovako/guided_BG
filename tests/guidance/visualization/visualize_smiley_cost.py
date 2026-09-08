"""Visualize the smiley guidance cost function over phi/psi space.

Pure geometry -- no model, no GPU, safe to run alongside anything else. Plots
the exact cost from make_cost_fn in the hparam_search_smiley_* scripts
(within_radius_penalty for the face + EYE_MOUTH_WEIGHT * repel_within_radius_
penalty for each eye/mouth hole), reusing those functions directly rather
than reimplementing the math.

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

from transferable_samplers.guidance.costs import repel_within_radius_penalty, within_radius_penalty

OUT_PATH = "tests/guidance/out/smiley_cost_landscape.png"
EYE_MOUTH_WEIGHT = 15.0  # matches the hparam_search_smiley_* scripts

# Same box target + derived smiley geometry as the guidance scripts.
PHI_TARGET = (-2.0, -1.0)
PSI_TARGET = (-0.5, 0.5)
_BOX_CENTER = ((PHI_TARGET[0] + PHI_TARGET[1]) / 2, (PSI_TARGET[0] + PSI_TARGET[1]) / 2)
_BOX_HALF_EXTENT = min(PHI_TARGET[1] - PHI_TARGET[0], PSI_TARGET[1] - PSI_TARGET[0]) / 2
_SMILEY_SCALE = 0.9 * _BOX_HALF_EXTENT / 2.2
_INNER_SCALE = 0.75

FACE_CENTER = _BOX_CENTER
FACE_RADIUS = 2.2 * _SMILEY_SCALE
EYE_CENTERS = [
    (_BOX_CENTER[0] + dx * _INNER_SCALE * _SMILEY_SCALE, _BOX_CENTER[1] + dy * _INNER_SCALE * _SMILEY_SCALE)
    for dx, dy in [(-1.0, 1.0), (1.0, 1.0)]
]
EYE_RADIUS = 0.4 * _SMILEY_SCALE
MOUTH_PHIS = [-1.5, -1.125, -0.75, -0.375, 0.0, 0.375, 0.75, 1.125, 1.5]
MOUTH_CENTERS = [
    (
        _BOX_CENTER[0] + p * _INNER_SCALE * _SMILEY_SCALE,
        _BOX_CENTER[1] + (-1.5 + 0.35 * p**2) * _INNER_SCALE * _SMILEY_SCALE,
    )
    for p in MOUTH_PHIS
]
MOUTH_RADIUS = 0.35 * _SMILEY_SCALE


def cost(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    dist_to_face = torch.sqrt((phi - FACE_CENTER[0]) ** 2 + (psi - FACE_CENTER[1]) ** 2)
    c = within_radius_penalty(dist_to_face, FACE_RADIUS)
    for cx, cy in EYE_CENTERS:
        d = torch.sqrt((phi - cx) ** 2 + (psi - cy) ** 2)
        c = c + EYE_MOUTH_WEIGHT * repel_within_radius_penalty(d, EYE_RADIUS)
    for cx, cy in MOUTH_CENTERS:
        d = torch.sqrt((phi - cx) ** 2 + (psi - cy) ** 2)
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
