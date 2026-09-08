"""Animate how the guided sampler's endpoint prediction wanders into the smiley
during a single guided Euler integration, using the current hyperparameters in
``plot_guided_euler_ramachandran.py``.

At every recorded step ``i`` (a subsample of the full ``EULER_STEPS``, to keep
the GIF a reasonable size), this plots the phi/psi of the *free endpoint
estimate* ``x1_pred = cxt + (1-t)*v(cxt,t)`` -- the same quantity
``guidance_cost_fn`` is evaluated on -- not the raw noisy running state ``x_t``,
since that's what "where does the model currently think this sample will end
up" actually means early in the trajectory.

Reimplements ``FlowMatchingModule._integrate_guided``'s loop inline (rather
than modifying it) purely to add this per-step recording hook.

Run with:
    uv run python tests/guidance/visualization/gif_guided_trajectory.py
"""

from __future__ import annotations

import copy
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm

from transferable_samplers.guidance.costs import repel_within_radius_penalty, within_radius_penalty
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
    GUIDANCE_INIT_CONTROL,
    GUIDANCE_INNER_STEPS,
    GUIDANCE_LR,
    GUIDANCE_OPTIMIZER,
    GUIDANCE_W_CONTROL,
    GUIDANCE_W_TERMINAL,
    GUIDANCE_W_VF,
    MOUTH_CENTERS,
    MOUTH_RADIUS,
    OBJECTIVE,
    OUT_DIR,
    PHI_TARGET,
    PSI_TARGET,
    SEQUENCE,
    load_model_and_data,
)

assert OBJECTIVE == "smiley", "This GIF is specific to the smiley cost landscape."

NUM_SAMPLES = 48
SEED = 42
NUM_FRAMES = 60  # subsampled from EULER_STEPS
GIF_PATH = f"{OUT_DIR}/guided_trajectory_smiley.gif"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, datamodule, num_atoms = load_model_and_data()
    model = model.to(device).eval()

    eval_ctx = datamodule.prepare_eval(sequence=SEQUENCE, stage="test")
    phi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="phi")
    psi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="psi")
    chirality_checker = ChiralitySignChecker(eval_ctx.topology, eval_ctx.true_data.samples[:1])

    def smiley_cost_fn(x1: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        x1_fixed = x1 * sign
        phi = dihedrals(x1_fixed, phi_idx).squeeze(-1)
        psi = dihedrals(x1_fixed, psi_idx).squeeze(-1)
        dist_to_face = torch.sqrt((phi - FACE_CENTER[0]) ** 2 + (psi - FACE_CENTER[1]) ** 2)
        cost = within_radius_penalty(dist_to_face, FACE_RADIUS)
        for cx, cy in EYE_CENTERS:
            dist = torch.sqrt((phi - cx) ** 2 + (psi - cy) ** 2)
            cost = cost + EYE_MOUTH_WEIGHT * repel_within_radius_penalty(dist, EYE_RADIUS)
        for cx, cy in MOUTH_CENTERS:
            dist = torch.sqrt((phi - cx) ** 2 + (psi - cy) ** 2)
            cost = cost + EYE_MOUTH_WEIGHT * repel_within_radius_penalty(dist, MOUTH_RADIUS)
        return cost

    def phi_psi_of(x1_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = x1_flat.reshape(NUM_SAMPLES, num_atoms, -1).detach()
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        x1_fixed = x1 * sign
        phi = dihedrals(x1_fixed, phi_idx).squeeze(-1)
        psi = dihedrals(x1_fixed, psi_idx).squeeze(-1)
        return phi.cpu(), psi.cpu()

    print("=== Guidance hyperparameters (from plot_guided_euler_ramachandran.py) ===")
    print(f"EULER_STEPS={EULER_STEPS} GUIDANCE_INNER_STEPS={GUIDANCE_INNER_STEPS} "
          f"GUIDANCE_GAMMA={GUIDANCE_GAMMA} GUIDANCE_LR={GUIDANCE_LR} "
          f"GUIDANCE_W_TERMINAL={GUIDANCE_W_TERMINAL} GUIDANCE_OPTIMIZER={GUIDANCE_OPTIMIZER}")
    print("==========================================================================\n")

    model.guidance_cost_fn = smiley_cost_fn
    model.guidance_num_steps = EULER_STEPS
    model.guidance_inner_steps = GUIDANCE_INNER_STEPS
    model.guidance_gamma = GUIDANCE_GAMMA
    model.guidance_lr = GUIDANCE_LR
    model.guidance_w_terminal = GUIDANCE_W_TERMINAL
    model.guidance_w_vf = GUIDANCE_W_VF
    model.guidance_w_control = GUIDANCE_W_CONTROL
    model.guidance_init_control = GUIDANCE_INIT_CONTROL
    model.guidance_optimizer = GUIDANCE_OPTIMIZER

    torch.manual_seed(SEED)
    z = model.prior.sample(NUM_SAMPLES, num_atoms, device=device)

    # -- Reimplementation of FlowMatchingModule._integrate_guided's loop, with a
    # per-step phi/psi recording hook (see module docstring for why not just
    # patch the real method).
    net_c = copy.deepcopy(model.net)
    net_c.requires_grad_(False)
    eval_fn = partial(net_c, encodings=None)

    ts = torch.linspace(0.0, 1.0, model.guidance_num_steps + 1, device=z.device)
    dt = ts[1] - ts[0]
    record_steps = set(np.linspace(0, model.guidance_num_steps - 1, NUM_FRAMES, dtype=int).tolist())

    xt = z.reshape(NUM_SAMPLES, -1)
    u_t_carry = None
    frames_phi, frames_psi, frames_t, frames_step = [], [], [], []

    print(f"Integrating {model.guidance_num_steps} guided Euler steps, recording {len(record_steps)} frames...")
    for i in range(model.guidance_num_steps):
        t_i = ts[i]
        gamma_t = model.guidance_gamma(t_i) if callable(model.guidance_gamma) else model.guidance_gamma

        with torch.enable_grad():
            xt_ = xt.detach()
            if model.guidance_init_control == "causal_zero" and i > 0:
                u_t = u_t_carry.clone().requires_grad_(True)
            else:
                u_t = torch.zeros_like(xt_, requires_grad=True)
            optimizer = torch.optim.Adam([u_t], lr=model.guidance_lr) if model.guidance_optimizer == "adam" else None

            vt_unguided = None
            if model.guidance_w_vf > 0:
                vt_unguided = eval_fn(t_i, xt_).detach()

            for _ in range(model.guidance_inner_steps):
                cxt = xt_ + gamma_t * u_t
                vt_control = eval_fn(t_i, cxt)
                x1_pred = cxt + (1.0 - t_i) * vt_control
                loss = model.guidance_w_terminal * model._guidance_terminal_cost(x1_pred, NUM_SAMPLES, num_atoms)

                if vt_unguided is not None:
                    vf_cost = (vt_control - vt_unguided).pow(2).reshape(NUM_SAMPLES, -1).sum(dim=-1)
                    loss = loss + model.guidance_w_vf * vf_cost
                if model.guidance_w_control > 0:
                    control_cost = u_t.pow(2).reshape(NUM_SAMPLES, -1).sum(dim=-1)
                    loss = loss + model.guidance_w_control * (gamma_t**2) * control_cost

                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.sum().backward()
                    optimizer.step()
                else:
                    (grad,) = torch.autograd.grad(loss.sum(), u_t)
                    grad_norm = grad.norm(dim=-1, keepdim=True)
                    u_t = (u_t - model.guidance_lr * grad / (grad_norm + 1e-8)).detach().requires_grad_(True)

            u_t_carry = u_t.detach()
            cxt = xt_ + gamma_t * u_t_carry
            vt_control = eval_fn(t_i, cxt)
            x1_pred = cxt + (1.0 - t_i) * vt_control

        xt = (cxt + dt * vt_control).detach()

        if i in record_steps:
            phi, psi = phi_psi_of(x1_pred)
            frames_phi.append(phi.numpy())
            frames_psi.append(psi.numpy())
            frames_t.append(float(t_i))
            frames_step.append(i + 1)  # 1-indexed Euler step number, out of guidance_num_steps

    # The loop's last recorded frame is x1_pred from the *second-to-last* Euler
    # step (i = num_steps - 1, t < 1) -- the model's own forward-looking
    # prediction, not the literal final integrated sample. Replace it with the
    # actual final `xt` (t = 1.0 exactly, what the real evaluation scripts use).
    phi_final, psi_final = phi_psi_of(xt)
    frames_phi[-1] = phi_final.numpy()
    frames_psi[-1] = psi_final.numpy()
    frames_t[-1] = 1.0
    frames_step[-1] = model.guidance_num_steps

    print(f"done, {len(frames_t)} frames recorded.\n")

    # -- Background: same smiley cost landscape as visualize_smiley_cost.py, zoomed.
    margin = 0.25 * FACE_RADIUS
    phi_range = (FACE_CENTER[0] - FACE_RADIUS - margin, FACE_CENTER[0] + FACE_RADIUS + margin)
    psi_range = (FACE_CENTER[1] - FACE_RADIUS - margin, FACE_CENTER[1] + FACE_RADIUS + margin)
    grid_phi = torch.linspace(*phi_range, 400)
    grid_psi = torch.linspace(*psi_range, 400)
    GRID_PHI, GRID_PSI = torch.meshgrid(grid_phi, grid_psi, indexing="xy")
    dist_to_face = torch.sqrt((GRID_PHI - FACE_CENTER[0]) ** 2 + (GRID_PSI - FACE_CENTER[1]) ** 2)
    cost_grid = within_radius_penalty(dist_to_face, FACE_RADIUS)
    for cx, cy in EYE_CENTERS:
        d = torch.sqrt((GRID_PHI - cx) ** 2 + (GRID_PSI - cy) ** 2)
        cost_grid = cost_grid + EYE_MOUTH_WEIGHT * repel_within_radius_penalty(d, EYE_RADIUS)
    for cx, cy in MOUTH_CENTERS:
        d = torch.sqrt((GRID_PHI - cx) ** 2 + (GRID_PSI - cy) ** 2)
        cost_grid = cost_grid + EYE_MOUTH_WEIGHT * repel_within_radius_penalty(d, MOUTH_RADIUS)
    cost_grid = cost_grid.numpy()

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.pcolormesh(
        GRID_PHI.numpy(), GRID_PSI.numpy(), cost_grid, cmap="viridis_r",
        norm=LogNorm(vmin=1e-4, vmax=max(cost_grid.max(), 1e-3)), shading="auto",
    )
    ax.set_xlim(*phi_range)
    ax.set_ylim(*psi_range)
    ax.set_xlabel(r"$\varphi$", fontsize=14)
    ax.set_ylabel(r"$\psi$", fontsize=14)
    ax.set_aspect("equal")
    scatter = ax.scatter([], [], s=18, c="#E8543A", edgecolors="white", linewidths=0.4, zorder=5)
    title = ax.set_title("")

    def update(frame_idx: int):
        scatter.set_offsets(np.column_stack([frames_phi[frame_idx], frames_psi[frame_idx]]))
        title.set_text(
            f"endpoint prediction $x_1$ at t={frames_t[frame_idx]:.2f}  "
            f"(Euler step {frames_step[frame_idx]}/{EULER_STEPS})"
        )
        return scatter, title

    anim = FuncAnimation(fig, update, frames=len(frames_t), interval=120, blit=False)
    anim.save(GIF_PATH, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"saved {GIF_PATH}")


if __name__ == "__main__":
    main()
