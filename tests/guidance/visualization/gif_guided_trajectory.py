"""Animated GIF of a guided sampler's endpoint prediction converging on an
objective.

``make_guided_trajectory_gif`` is the reusable entry point: it takes a model,
topology-derived indices, a cost function, and guidance hyperparameters as
plain arguments (no dependency on ``plot_guided_euler_ramachandran.py``), and
integrates + animates + saves the GIF. Any script or notebook can call it
directly with its own objective and hyperparameters.

``main()`` below is this script's own standalone use of it: three switches at
the top control everything --

    GIF_OBJECTIVE: "pos_phi", "phi_target", "phi_psi_target", "smiley",
        "smiley_reference", or "unguided" (no-guidance baseline).
    GIF_ZOOM: "full" (the whole [-pi,pi] x [-pi,pi] torus) or "zoom" (a
        window around the objective's own target region; "pos_phi" has no
        bounded target region, so it's always full).
    GIF_BACKGROUND: "none", "density" (a quick fresh unguided Euler pass's
        phi/psi density, for comparison against where guidance redistributes
        mass), or "cost" (a static heatmap of the cost function's values;
        only defined for time-independent costs, not "smiley_reference" or
        "unguided").

Its hyperparameters (EULER_STEPS, GUIDANCE_GAMMA, GUIDANCE_LR,
GUIDANCE_W_TERMINAL, etc.) are imported from plot_guided_euler_ramachandran.py
so it stays in sync with whatever config that script is currently using.

Run with:
    uv run python tests/guidance/visualization/gif_guided_trajectory.py
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

from transferable_samplers.guidance.costs import (
    box_quadratic_penalty,
    one_sided_quadratic_penalty,
    quadratic_target_penalty,
    repel_within_radius_penalty,
    torus_distance,
    within_radius_penalty,
)
from transferable_samplers.guidance.euler_density_integrator import make_guided_euler_step
from transferable_samplers.guidance.observables import dihedrals, get_dihedral_atom_indices
from transferable_samplers.utils.chirality import ChiralitySignChecker

from plot_guided_euler_ramachandran import (
    EULER_STEPS,
    EYE_CENTERS,
    EYE_MOUTH_WEIGHT,
    EYE_RADIUS,
    FACE_CENTER,
    FACE_RADIUS,
    GUIDANCE_GAMMA,
    GUIDANCE_INNER_STEPS,
    GUIDANCE_LR,
    GUIDANCE_W_TERMINAL,
    MOUTH_CENTERS,
    MOUTH_RADIUS,
    OUT_DIR,
    PHI_PSI_TARGET,
    PHI_TARGET,
    PSI_TARGET,
    REFERENCE_TRAJECTORY_PATH,
    SEQUENCE,
    SMILEY_REFERENCE_T_SWITCH,
    load_model_and_data,
)
from plot_unguided_euler_ramachandran import EULER_STEPS as UNGUIDED_EULER_STEPS

Tensor = torch.Tensor

__all__ = ["make_guided_trajectory_gif"]

GIF_OBJECTIVE = "phi_psi_target"  # "pos_phi", "phi_target", "phi_psi_target", "smiley", "smiley_reference", "unguided"
GIF_ZOOM = "full"  # "full" or "zoom"
GIF_BACKGROUND = "density"  # "none", "density", or "cost"

NUM_SAMPLES = 200  # bump to 1000+ for a cleaner point cloud when GPU headroom allows
NUM_BACKGROUND_SAMPLES = 2000  # only used for GIF_BACKGROUND="density"
SEED = 42
NUM_FRAMES = 60  # subsampled from EULER_STEPS
GIF_PATH = f"{OUT_DIR}/guided_trajectory_{GIF_OBJECTIVE}_{GIF_ZOOM}_{GIF_BACKGROUND}.gif"

_FULL_RANGE = (-math.pi, math.pi)


# --------------------------------------------------------------------------------------
# Reusable core: integrate + animate + save, parameterized by cost fn and hyperparameters
# --------------------------------------------------------------------------------------


def make_guided_trajectory_gif(
    model,
    num_atoms: int,
    phi_idx: np.ndarray,
    psi_idx: np.ndarray,
    chirality_checker: ChiralitySignChecker,
    cost_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    gif_path: str | Path,
    num_samples: int = 200,
    euler_steps: int = 100,
    num_frames: int = 60,
    gamma: float | Callable[[Tensor], float] = 1.0,
    alpha: float = 0.1,
    n_inner: int = 1,
    seed: int = 42,
    device: str | torch.device = "cpu",
    phi_range: tuple[float, float] = _FULL_RANGE,
    psi_range: tuple[float, float] = _FULL_RANGE,
    draw_targets: Callable[[plt.Axes], None] | None = None,
    background_density: tuple[np.ndarray, np.ndarray] | None = None,
    background_grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    reference_fn: Callable[[float], tuple[float, float] | None] | None = None,
    point_color: str = "red",
    point_size: float = 12,
    fps: int = 10,
    figsize: tuple[float, float] = (6.5, 6.5),
) -> Path:
    """Integrate guided Euler steps, recording phi/psi after each recorded
    step, and save the trajectory as an animated GIF.

    Args:
        model: A ``FlowMatchingModule`` (``.net``/``.prior`` used).
        num_atoms: Atoms per conformation.
        phi_idx, psi_idx: Dihedral atom index quadruples, see
            ``get_dihedral_atom_indices``.
        chirality_checker: Precomputed ``ChiralitySignChecker`` for the topology.
        cost_fn: Batched terminal cost, ``(x1: (B, atoms, 3), t) -> (B,)``.
            Pass e.g. ``lambda x1, t: torch.zeros(x1.shape[0])`` for an
            unguided baseline.
        gif_path: Output path; parent directories are created if needed.
        num_samples: Number of trajectories to draw.
        euler_steps: Number of Euler steps.
        num_frames: Number of frames to record, subsampled from ``euler_steps``.
        gamma, alpha, n_inner: See ``make_guided_euler_step`` (``alpha=0``/
            ``n_inner=0`` gives plain unguided Euler).
        seed: RNG seed for the prior draw.
        device: Device to sample on.
        phi_range, psi_range: Axis limits.
        draw_targets: Optional ``ax -> None`` callback drawing a static
            target overlay (circles, markers, lines, ...).
        background_density: Optional ``(phi, psi)`` arrays plotted as a log
            2-D histogram behind the trajectory.
        background_grid: Optional ``(grid_phi, grid_psi, cost)`` arrays
            plotted as a static log-scale heatmap behind the trajectory.
        reference_fn: Optional ``t -> (phi, psi) | None`` giving a moving
            reference point to overlay per frame (``None`` hides it for that
            frame).
        point_color, point_size: Trajectory scatter style.
        fps: Output GIF frame rate.
        figsize: Figure size.

    Returns:
        ``gif_path`` (as a ``Path``), after saving.
    """
    net = copy.deepcopy(model.net)
    net.requires_grad_(False)

    def single_sample_cost_fn(x1_flat: Tensor, t: Tensor) -> Tensor:
        return cost_fn(x1_flat.view(1, -1, 3), t).squeeze()

    dt = 1.0 / euler_steps
    step = make_guided_euler_step(
        net, None, single_sample_cost_fn,
        dt=dt, gamma=gamma, alpha=alpha, n_inner=n_inner, use_score_deviation=False,
    )

    def phi_psi_of(x1_flat: Tensor, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        x1 = x1_flat.reshape(batch_size, num_atoms, -1).detach()
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        x1_fixed = x1 * sign
        phi = dihedrals(x1_fixed, phi_idx).squeeze(-1)
        psi = dihedrals(x1_fixed, psi_idx).squeeze(-1)
        return phi.cpu().numpy(), psi.cpu().numpy()

    torch.manual_seed(seed)
    z = model.prior.sample(num_samples, num_atoms, device=device)
    x = z.reshape(num_samples, -1)

    # Plots x_t right after each recorded step (what make_guided_euler_step
    # returns), so the last recorded frame is the true final sample (t=1).
    record_steps = set(np.linspace(0, euler_steps - 1, num_frames, dtype=int).tolist())
    frames_phi, frames_psi, frames_t, frames_step = [], [], [], []
    frames_ref = [] if reference_fn is not None else None

    for k in range(euler_steps):
        t = torch.as_tensor(k * dt, device=device, dtype=x.dtype)
        x = step(t, x).detach()
        if k in record_steps:
            phi, psi = phi_psi_of(x, num_samples)
            frames_phi.append(phi)
            frames_psi.append(psi)
            t_next = (k + 1) * dt
            frames_t.append(t_next)
            frames_step.append(k + 1)
            if reference_fn is not None:
                frames_ref.append(reference_fn(t_next))

    fig, ax = plt.subplots(figsize=figsize)

    if background_density is not None:
        phi_bg, psi_bg = background_density
        ax.hist2d(phi_bg, psi_bg, 100, norm=LogNorm(), range=[_FULL_RANGE, _FULL_RANGE], alpha=0.4, zorder=1)
    elif background_grid is not None:
        grid_phi, grid_psi, cost_grid = background_grid
        ax.pcolormesh(
            grid_phi, grid_psi, cost_grid, cmap="viridis_r",
            norm=LogNorm(vmin=1e-4, vmax=max(cost_grid.max(), 1e-3)), shading="auto", zorder=1,
        )

    if draw_targets is not None:
        draw_targets(ax)

    ax.set_xlim(*phi_range)
    ax.set_ylim(*psi_range)
    ax.set_xlabel(r"$\varphi$", fontsize=14)
    ax.set_ylabel(r"$\psi$", fontsize=14)
    ax.set_aspect("equal")

    scatter = ax.scatter([], [], s=point_size, c=point_color, alpha=0.7, edgecolors="none", zorder=5)
    ref_marker = ax.scatter([], [], s=90, c="tab:blue", marker="*", edgecolors="black", zorder=6) if reference_fn is not None else None
    title = ax.set_title("")

    def update(frame_idx: int):
        scatter.set_offsets(np.column_stack([frames_phi[frame_idx], frames_psi[frame_idx]]))
        artists = [scatter, title]
        suffix = ""
        if ref_marker is not None:
            ref = frames_ref[frame_idx]
            ref_marker.set_offsets([ref] if ref is not None else np.empty((0, 2)))
            suffix = "  (tracking reference)" if ref is not None else "  (no reference)"
            artists.append(ref_marker)
        title.set_text(f"t={frames_t[frame_idx]:.2f}  (Euler step {frames_step[frame_idx]}/{euler_steps}){suffix}")
        return tuple(artists)

    anim = FuncAnimation(fig, update, frames=len(frames_t), interval=120, blit=False)
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return gif_path


# --------------------------------------------------------------------------------------
# This script's own standalone use of it (the GIF_OBJECTIVE/GIF_ZOOM/GIF_BACKGROUND demo)
# --------------------------------------------------------------------------------------


def _zoom_window() -> tuple[tuple[float, float], tuple[float, float]]:
    """(phi_range, psi_range) for GIF_ZOOM="zoom", per objective."""
    if GIF_OBJECTIVE in ("smiley", "smiley_reference"):
        m = 0.25 * FACE_RADIUS
        return (FACE_CENTER[0] - FACE_RADIUS - m, FACE_CENTER[0] + FACE_RADIUS + m), \
               (FACE_CENTER[1] - FACE_RADIUS - m, FACE_CENTER[1] + FACE_RADIUS + m)
    if GIF_OBJECTIVE == "phi_psi_target":
        m = 1.0
        return (PHI_PSI_TARGET[0] - m, PHI_PSI_TARGET[0] + m), (PHI_PSI_TARGET[1] - m, PHI_PSI_TARGET[1] + m)
    if GIF_OBJECTIVE == "phi_target":
        return (1.0 - 1.5, 1.0 + 1.5), _FULL_RANGE  # phi-only target -- no natural psi window
    if GIF_OBJECTIVE == "unguided":
        m = 0.6
        return (PHI_TARGET[0] - m, PHI_TARGET[1] + m), (PSI_TARGET[0] - m, PSI_TARGET[1] + m)
    # "pos_phi": target is an unbounded half-plane (phi > 0) -- no bounded zoom region.
    return _FULL_RANGE, _FULL_RANGE


def _smiley_cost_grid(grid_phi: torch.Tensor, grid_psi: torch.Tensor) -> torch.Tensor:
    dist_to_face = torus_distance(grid_phi, grid_psi, FACE_CENTER[0], FACE_CENTER[1])
    cost = within_radius_penalty(dist_to_face, FACE_RADIUS)
    for cx, cy in EYE_CENTERS:
        d = torus_distance(grid_phi, grid_psi, cx, cy)
        cost = cost + EYE_MOUTH_WEIGHT * repel_within_radius_penalty(d, EYE_RADIUS)
    for cx, cy in MOUTH_CENTERS:
        d = torus_distance(grid_phi, grid_psi, cx, cy)
        cost = cost + EYE_MOUTH_WEIGHT * repel_within_radius_penalty(d, MOUTH_RADIUS)
    return cost


def main() -> None:
    if GIF_BACKGROUND == "cost" and GIF_OBJECTIVE in ("smiley_reference", "unguided"):
        raise ValueError(f'GIF_BACKGROUND="cost" isn\'t defined for GIF_OBJECTIVE="{GIF_OBJECTIVE}".')
    if GIF_OBJECTIVE == "unguided" and GIF_BACKGROUND == "density":
        raise ValueError('GIF_BACKGROUND="density" is redundant for GIF_OBJECTIVE="unguided" -- the foreground already is that density; use "none".')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, datamodule, num_atoms = load_model_and_data()
    model = model.to(device).eval()

    eval_ctx = datamodule.prepare_eval(sequence=SEQUENCE, stage="test")
    phi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="phi")
    psi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="psi")
    chirality_checker = ChiralitySignChecker(eval_ctx.topology, eval_ctx.true_data.samples[:1])

    # -- Cost functions, batched convention (x1: (B, atoms, dims) -> (B,)) --

    def pos_phi_cost_fn(x1: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        phi = dihedrals(x1 * sign, phi_idx)
        return one_sided_quadratic_penalty(phi, threshold=0.0, penalize_below=True).sum(dim=-1)

    def phi_target_cost_fn(x1: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        phi = dihedrals(x1 * sign, phi_idx)
        return box_quadratic_penalty(phi, low=1.0, high=1.0).sum(dim=-1)

    def phi_psi_target_cost_fn(x1: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        x1_fixed = x1 * sign
        phi = dihedrals(x1_fixed, phi_idx).squeeze(-1)
        psi = dihedrals(x1_fixed, psi_idx).squeeze(-1)
        dist = torus_distance(phi, psi, PHI_PSI_TARGET[0], PHI_PSI_TARGET[1])
        return quadratic_target_penalty(dist)

    def smiley_cost_fn(x1: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        x1_fixed = x1 * sign
        phi = dihedrals(x1_fixed, phi_idx).squeeze(-1)
        psi = dihedrals(x1_fixed, psi_idx).squeeze(-1)
        dist_to_face = torus_distance(phi, psi, FACE_CENTER[0], FACE_CENTER[1])
        cost = within_radius_penalty(dist_to_face, FACE_RADIUS)
        for cx, cy in EYE_CENTERS:
            dist = torus_distance(phi, psi, cx, cy)
            cost = cost + EYE_MOUTH_WEIGHT * repel_within_radius_penalty(dist, EYE_RADIUS)
        for cx, cy in MOUTH_CENTERS:
            dist = torus_distance(phi, psi, cx, cy)
            cost = cost + EYE_MOUTH_WEIGHT * repel_within_radius_penalty(dist, MOUTH_RADIUS)
        return cost

    reference_phi_psi_at = None
    if GIF_OBJECTIVE == "smiley_reference":
        ref_data = torch.load(REFERENCE_TRAJECTORY_PATH, weights_only=False)
        x0_ref = ref_data["x0"].to(device)
        x1_ref = ref_data["x1"].to(device)

        def reference_phi_psi_at(t: torch.Tensor) -> tuple[float, float]:
            """The reference (noised smiley-center) point's own endpoint prediction at t."""
            with torch.no_grad():
                x_t_ref = (1.0 - t) * x0_ref + t * x1_ref
                v_ref = model.net(t.reshape(1), x_t_ref.reshape(1, -1), encodings=None).reshape_as(x_t_ref)
                x1_hat_ref = x_t_ref + (1.0 - t) * v_ref
                flip_ref = chirality_checker.flip_mask(x1_hat_ref)
                sign_ref = torch.where(flip_ref, -1.0, 1.0).to(x1_hat_ref)[:, None, None]
                x1_hat_ref_fixed = x1_hat_ref * sign_ref
                phi_ref = dihedrals(x1_hat_ref_fixed, phi_idx).squeeze(-1)
                psi_ref = dihedrals(x1_hat_ref_fixed, psi_idx).squeeze(-1)
            return phi_ref.item(), psi_ref.item()

        def smiley_reference_cost_fn(x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            if t > SMILEY_REFERENCE_T_SWITCH:
                return smiley_cost_fn(x1)
            with torch.no_grad():
                flip_mask = chirality_checker.flip_mask(x1)
            sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
            x1_fixed = x1 * sign
            phi_cur = dihedrals(x1_fixed, phi_idx).squeeze(-1)
            psi_cur = dihedrals(x1_fixed, psi_idx).squeeze(-1)
            phi_ref, psi_ref = reference_phi_psi_at(t)
            dist = torus_distance(phi_cur, psi_cur, phi_ref, psi_ref)
            return quadratic_target_penalty(dist)

    _cost_fn_by_objective = {
        "pos_phi": lambda x1, t: pos_phi_cost_fn(x1),
        "phi_target": lambda x1, t: phi_target_cost_fn(x1),
        "phi_psi_target": lambda x1, t: phi_psi_target_cost_fn(x1),
        "smiley": lambda x1, t: smiley_cost_fn(x1),
        "smiley_reference": (lambda x1, t: smiley_reference_cost_fn(x1, t)) if GIF_OBJECTIVE == "smiley_reference" else None,
        "unguided": lambda x1, t: torch.zeros(x1.shape[0], device=x1.device, dtype=x1.dtype),
    }
    cost_fn = _cost_fn_by_objective[GIF_OBJECTIVE]

    def phi_psi_of(x1_flat: torch.Tensor, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        x1 = x1_flat.reshape(batch_size, num_atoms, -1).detach()
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        x1_fixed = x1 * sign
        phi = dihedrals(x1_fixed, phi_idx).squeeze(-1)
        psi = dihedrals(x1_fixed, psi_idx).squeeze(-1)
        return phi.cpu().numpy(), psi.cpu().numpy()

    net = copy.deepcopy(model.net)
    net.requires_grad_(False)

    # -- Background --
    background_density = None
    background_grid = None

    if GIF_BACKGROUND == "density":
        print(f"Computing unguided background density ({NUM_BACKGROUND_SAMPLES} samples, {UNGUIDED_EULER_STEPS} steps)...")
        bg_step = make_guided_euler_step(
            net, None, lambda x1, t: torch.zeros(x1.shape[0], device=x1.device, dtype=x1.dtype),
            dt=1.0 / UNGUIDED_EULER_STEPS, gamma=0.0, alpha=0.0, n_inner=0, use_score_deviation=False,
        )
        torch.manual_seed(SEED)
        z_bg = model.prior.sample(NUM_BACKGROUND_SAMPLES, num_atoms, device=device)
        xt_bg = z_bg.reshape(NUM_BACKGROUND_SAMPLES, -1)
        with torch.no_grad():
            for i in range(UNGUIDED_EULER_STEPS):
                t_bg = torch.as_tensor(i / UNGUIDED_EULER_STEPS, device=device, dtype=xt_bg.dtype)
                xt_bg = bg_step(t_bg, xt_bg)
        background_density = phi_psi_of(xt_bg, NUM_BACKGROUND_SAMPLES)
        print("done.\n")
    elif GIF_BACKGROUND == "cost":
        grid_phi_range, grid_psi_range = _zoom_window() if GIF_ZOOM == "zoom" else (_FULL_RANGE, _FULL_RANGE)
        grid_phi = torch.linspace(*grid_phi_range, 400)
        grid_psi = torch.linspace(*grid_psi_range, 400)
        GRID_PHI, GRID_PSI = torch.meshgrid(grid_phi, grid_psi, indexing="xy")
        with torch.no_grad():
            cost_grid = {
                "pos_phi": lambda: one_sided_quadratic_penalty(GRID_PHI, threshold=0.0, penalize_below=True),
                "phi_target": lambda: box_quadratic_penalty(GRID_PHI, low=1.0, high=1.0),
                "phi_psi_target": lambda: quadratic_target_penalty(
                    torus_distance(GRID_PHI, GRID_PSI, PHI_PSI_TARGET[0], PHI_PSI_TARGET[1])
                ),
                "smiley": lambda: _smiley_cost_grid(GRID_PHI, GRID_PSI),
            }[GIF_OBJECTIVE]().numpy()
        background_grid = (GRID_PHI.numpy(), GRID_PSI.numpy(), cost_grid)

    print(f"=== Guidance hyperparameters (OBJECTIVE={GIF_OBJECTIVE}, from plot_guided_euler_ramachandran.py) ===")
    print(f"NUM_SAMPLES={NUM_SAMPLES} EULER_STEPS={EULER_STEPS} GUIDANCE_INNER_STEPS={GUIDANCE_INNER_STEPS} "
          f"GUIDANCE_GAMMA={GUIDANCE_GAMMA} GUIDANCE_LR={GUIDANCE_LR} GUIDANCE_W_TERMINAL={GUIDANCE_W_TERMINAL}")
    print("=" * 90 + "\n")

    is_unguided = GIF_OBJECTIVE == "unguided"

    def weighted_cost_fn(x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (0.0 if is_unguided else GUIDANCE_W_TERMINAL) * cost_fn(x1, t)

    reference_fn = None
    if GIF_OBJECTIVE == "smiley_reference":

        def reference_fn(t: float) -> tuple[float, float] | None:
            t_tensor = torch.as_tensor(t, device=device, dtype=torch.float32)
            if t_tensor > SMILEY_REFERENCE_T_SWITCH:
                return None
            return reference_phi_psi_at(t_tensor)

    # -- Target overlay --

    def draw_targets(ax: plt.Axes) -> None:
        if GIF_OBJECTIVE in ("smiley", "smiley_reference"):
            ax.add_patch(Circle(FACE_CENTER, FACE_RADIUS, fill=False, edgecolor="black", linewidth=1.2, zorder=3))
            for cx, cy in EYE_CENTERS:
                ax.add_patch(Circle((cx, cy), EYE_RADIUS, fill=False, edgecolor="black", linewidth=1.0, zorder=3))
            for cx, cy in MOUTH_CENTERS:
                ax.add_patch(Circle((cx, cy), MOUTH_RADIUS, fill=False, edgecolor="black", linewidth=1.0, zorder=3))
        elif GIF_OBJECTIVE == "phi_psi_target":
            ax.plot(*PHI_PSI_TARGET, marker="+", markersize=18, markeredgewidth=2.2, color="black", zorder=3)
            ax.plot(*PHI_PSI_TARGET, marker="o", markersize=10, markerfacecolor="none", markeredgecolor="black", zorder=3)
        elif GIF_OBJECTIVE == "phi_target":
            ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--", zorder=3)
        elif GIF_OBJECTIVE == "pos_phi":
            ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", zorder=3)
        elif GIF_OBJECTIVE == "unguided":
            rect = mpatches.Rectangle(
                (PHI_TARGET[0], PSI_TARGET[0]), PHI_TARGET[1] - PHI_TARGET[0], PSI_TARGET[1] - PSI_TARGET[0],
                fill=False, edgecolor="#E8543A", linewidth=1.8, linestyle="--", zorder=3,
            )
            ax.add_patch(rect)

    phi_range, psi_range = _zoom_window() if GIF_ZOOM == "zoom" else (_FULL_RANGE, _FULL_RANGE)
    point_color = "#1f5fa8" if is_unguided else "red"

    print(f"Integrating {EULER_STEPS} {'unguided' if is_unguided else 'guided'} Euler steps "
          f"({NUM_SAMPLES} samples), recording {NUM_FRAMES} frames...")
    result_path = make_guided_trajectory_gif(
        model, num_atoms, phi_idx, psi_idx, chirality_checker, weighted_cost_fn,
        gif_path=GIF_PATH,
        num_samples=NUM_SAMPLES,
        euler_steps=EULER_STEPS,
        num_frames=NUM_FRAMES,
        gamma=GUIDANCE_GAMMA,
        alpha=0.0 if is_unguided else GUIDANCE_LR,
        n_inner=0 if is_unguided else GUIDANCE_INNER_STEPS,
        seed=SEED,
        device=device,
        phi_range=phi_range,
        psi_range=psi_range,
        draw_targets=draw_targets,
        background_density=background_density,
        background_grid=background_grid,
        reference_fn=reference_fn,
        point_color=point_color,
    )
    print(f"saved {result_path}")


if __name__ == "__main__":
    main()
