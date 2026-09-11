"""Animated GIF of the guided sampler's endpoint prediction converging on a
chosen objective. Three independent switches at the top control everything:

    GIF_OBJECTIVE: "pos_phi", "phi_target", "phi_psi_target", "smiley",
        "smiley_reference", or "unguided" (no guidance at all -- the
        no-guidance baseline, alpha=0/n_inner=0).
    GIF_ZOOM: "full" (the whole [-pi,pi] x [-pi,pi] torus) or "zoom" (a
        window around the objective's own target region -- pos_phi has no
        bounded target region, so it's always full regardless).
    GIF_BACKGROUND: "none" (plain), "density" (a quick fresh unguided Euler
        pass's phi/psi density, hist2d/LogNorm/viridis, same style as
        plot_ramachandran.py -- shows how guided samples redistribute
        relative to where the model naturally puts mass; meaningless for
        GIF_OBJECTIVE="unguided", since the foreground already is that
        density) or "cost" (a static heatmap of the cost function's own
        values over the grid -- only defined for objectives with a
        time-independent closed-form cost: pos_phi, phi_target,
        phi_psi_target, smiley; not available for smiley_reference, whose
        cost depends on t, or unguided, which has none).

Replaces gif_guided_trajectory.py (old: smiley-only, zoomed, cost-landscape
background), gif_guided_trajectory_full_space.py (smiley/phi_target/pos_phi/
unguided, full-space, density background), and gif_guided_trajectory_simple.py
(phi_psi_target/smiley/smiley_reference, full-space, no background) -- three
scripts whose only real differences were exactly these three axes, now one
script with three parameters.

Also standardizes on generate_proposal_guided_euler/make_guided_euler_step
(euler_density_integrator.py) as the integration engine for EVERY objective,
including the ones the old full-space/zoomed scripts drove by hand-reimplementing
FlowMatchingModule._integrate_guided's loop -- confirmed functionally
equivalent earlier this session (same seed/params -> same results to ~1e-3).
Two things that reimplementation supported and this does not: GUIDANCE_OPTIMIZER
="adam" (only "gd"-style normalized-gradient inner steps here) and
GUIDANCE_INIT_CONTROL="causal_zero" (the control always resets to zero each
step here, never carried over) -- both match plot_guided_euler_ramachandran.py's
current active settings ("gd" / "zero"), so this is a simplification, not a
behavior change, for the configs actually in use.

At every recorded step, this plots the phi/psi of the actual running state
``x_t`` right after that step (what ``make_guided_euler_step`` returns) --
NOT the old full-space scripts' forward-looking *free endpoint estimate*
``x1_pred = cxt + (1-t)*v(cxt,t)`` (the same quantity the cost function
itself is evaluated on internally, which "sees" a cleaner picture of where
guidance is steering the sample than the still-noisy running state does,
especially early on) -- matching the "simple" family's convention instead,
since ``make_guided_euler_step`` doesn't expose that intermediate quantity.
One upside of this simplification: the last recorded frame IS the true final
sample already (t=1 exactly), so unlike the old scripts, no special-case
replacement of a stale forward-looking last frame is needed.

Hyperparameters (EULER_STEPS, GUIDANCE_GAMMA, GUIDANCE_LR, GUIDANCE_W_TERMINAL,
etc.) are imported from plot_guided_euler_ramachandran.py so this stays in
sync with whatever config that script is currently using.

Run with:
    uv run python tests/guidance/visualization/gif_guided_trajectory.py
"""

from __future__ import annotations

import copy
import math

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

GIF_OBJECTIVE = "smiley"  # "pos_phi", "phi_target", "phi_psi_target", "smiley", "smiley_reference", "unguided"
GIF_ZOOM = "full"  # "full" or "zoom"
GIF_BACKGROUND = "none"  # "none", "density", or "cost"

NUM_SAMPLES = 32  # trimmed for GPU headroom while other jobs share this GPU -- bump to 1000+ for a
# cleaner point cloud with GIF_BACKGROUND="density"/"smiley" (no drawn target) once the GPU is free
NUM_BACKGROUND_SAMPLES = 2000  # only used for GIF_BACKGROUND="density"
SEED = 42
NUM_FRAMES = 60  # subsampled from EULER_STEPS
GIF_PATH = f"{OUT_DIR}/guided_trajectory_{GIF_OBJECTIVE}_{GIF_ZOOM}_{GIF_BACKGROUND}.gif"

_FULL_RANGE = (-math.pi, math.pi)


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

    def single_sample_cost_fn(x1_flat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return cost_fn(x1_flat.view(1, -1, 3), t).squeeze()

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
    phi_bg = psi_bg = None
    cost_grid = grid_phi_range = grid_psi_range = None

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
        phi_bg, psi_bg = phi_psi_of(xt_bg, NUM_BACKGROUND_SAMPLES)
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
        grid_phi_np, grid_psi_np = GRID_PHI.numpy(), GRID_PSI.numpy()

    print(f"=== Guidance hyperparameters (OBJECTIVE={GIF_OBJECTIVE}, from plot_guided_euler_ramachandran.py) ===")
    print(f"NUM_SAMPLES={NUM_SAMPLES} EULER_STEPS={EULER_STEPS} GUIDANCE_INNER_STEPS={GUIDANCE_INNER_STEPS} "
          f"GUIDANCE_GAMMA={GUIDANCE_GAMMA} GUIDANCE_LR={GUIDANCE_LR} GUIDANCE_W_TERMINAL={GUIDANCE_W_TERMINAL}")
    print("=" * 90 + "\n")

    is_unguided = GIF_OBJECTIVE == "unguided"
    dt = 1.0 / EULER_STEPS
    step = make_guided_euler_step(
        net, None, lambda x1, t: (0.0 if is_unguided else GUIDANCE_W_TERMINAL) * single_sample_cost_fn(x1, t),
        dt=dt, gamma=GUIDANCE_GAMMA, alpha=0.0 if is_unguided else GUIDANCE_LR,
        n_inner=0 if is_unguided else GUIDANCE_INNER_STEPS, use_score_deviation=False,
    )

    torch.manual_seed(SEED)
    z = model.prior.sample(NUM_SAMPLES, num_atoms, device=device)
    x = z.reshape(NUM_SAMPLES, -1)

    record_steps = set(np.linspace(0, EULER_STEPS - 1, NUM_FRAMES, dtype=int).tolist())
    frames_phi, frames_psi, frames_t, frames_step, frames_ref = [], [], [], [], []

    print(f"Integrating {EULER_STEPS} {'unguided' if is_unguided else 'guided'} Euler steps "
          f"({NUM_SAMPLES} samples), recording {len(record_steps)} frames...")
    for k in range(EULER_STEPS):
        t = torch.as_tensor(k * dt, device=device, dtype=x.dtype)
        # Records the running state x_next itself, not a forward-looking free
        # endpoint estimate -- see module docstring.
        x = step(t, x).detach()
        if k in record_steps:
            phi, psi = phi_psi_of(x, NUM_SAMPLES)
            frames_phi.append(phi)
            frames_psi.append(psi)
            t_next = (k + 1) * dt
            frames_t.append(t_next)
            frames_step.append(k + 1)
            if reference_phi_psi_at is not None:
                t_next_tensor = torch.as_tensor(t_next, device=device, dtype=x.dtype)
                frames_ref.append(reference_phi_psi_at(t_next_tensor) if t_next_tensor <= SMILEY_REFERENCE_T_SWITCH else None)
        if (k + 1) % 50 == 0:
            print(f"  step {k + 1}/{EULER_STEPS}")

    print(f"done, {len(frames_t)} frames recorded.\n")

    # -- Plot --
    if GIF_ZOOM == "zoom":
        phi_range, psi_range = _zoom_window()
    else:
        phi_range, psi_range = _FULL_RANGE, _FULL_RANGE

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    if GIF_BACKGROUND == "density":
        ax.hist2d(phi_bg, psi_bg, 100, norm=LogNorm(), range=[_FULL_RANGE, _FULL_RANGE], alpha=0.4, zorder=1)
    elif GIF_BACKGROUND == "cost":
        ax.pcolormesh(
            grid_phi_np, grid_psi_np, cost_grid, cmap="viridis_r",
            norm=LogNorm(vmin=1e-4, vmax=max(cost_grid.max(), 1e-3)), shading="auto", zorder=1,
        )

    # Target overlay -- always drawn (cheap, and helps even with a density/cost background).
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

    ax.set_xlim(*phi_range)
    ax.set_ylim(*psi_range)
    ax.set_xlabel(r"$\varphi$", fontsize=14)
    ax.set_ylabel(r"$\psi$", fontsize=14)
    ax.set_aspect("equal")

    point_color = "#1f5fa8" if is_unguided else "red"
    scatter = ax.scatter([], [], s=12, c=point_color, alpha=0.7, edgecolors="none", zorder=5)
    ref_marker = ax.scatter([], [], s=90, c="tab:blue", marker="*", edgecolors="black", zorder=6) if frames_ref else None
    title = ax.set_title("")

    def update(frame_idx: int):
        scatter.set_offsets(np.column_stack([frames_phi[frame_idx], frames_psi[frame_idx]]))
        artists = [scatter, title]
        suffix = ""
        if ref_marker is not None:
            ref = frames_ref[frame_idx]
            ref_marker.set_offsets([ref] if ref is not None else np.empty((0, 2)))
            suffix = "  (tracking reference)" if ref is not None else "  (smiley shape)"
            artists.append(ref_marker)
        title.set_text(f"t={frames_t[frame_idx]:.2f}  (Euler step {frames_step[frame_idx]}/{EULER_STEPS}){suffix}")
        return tuple(artists)

    anim = FuncAnimation(fig, update, frames=len(frames_t), interval=120, blit=False)
    anim.save(GIF_PATH, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"saved {GIF_PATH}")


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


if __name__ == "__main__":
    main()
