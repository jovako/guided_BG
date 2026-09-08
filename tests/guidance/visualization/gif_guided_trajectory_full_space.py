"""Animate how the guided sampler's endpoint prediction wanders across the FULL
phi/psi Ramachandran space into the smiley, using the current hyperparameters in
``plot_guided_euler_ramachandran.py``.

Unlike ``gif_guided_trajectory.py`` (zoomed to the smiley box, with the cost
landscape drawn as a background heatmap, small sample count for a clean
scatter), this shows the whole [-pi, pi] x [-pi, pi] torus -- with enough
samples (500-1000), the smiley shape should emerge purely from the
point-cloud density once the samples converge, not from any target overlay.
The background is a density plot of the *unguided* equilibrium distribution
(a fresh, quick unguided Euler pass with the same model, same step count as
``plot_unguided_euler_ramachandran.py``), drawn with the same style as
``plot_ramachandran.py``'s own density plots (hist2d, LogNorm, default
viridis colormap, at 40% alpha) -- so it's easy to see how the guided samples
redistribute relative to where the model naturally puts mass. Samples are
drawn in red for visibility against that backdrop.

At every recorded step, this plots the phi/psi of the *free endpoint
estimate* ``x1_pred = cxt + (1-t)*v(cxt,t)`` -- the same quantity
``guidance_cost_fn`` is evaluated on -- not the raw noisy running state
``x_t``, since that's what "where does the model currently think this sample
will end up" actually means early in the trajectory.

Reimplements ``FlowMatchingModule._integrate_guided``'s loop inline (rather
than modifying it) purely to add this per-step recording hook.

Run with:
    uv run python tests/guidance/visualization/gif_guided_trajectory_full_space.py
"""

from __future__ import annotations

import copy
import math
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm

from transferable_samplers.guidance.observables import dihedrals, get_dihedral_atom_indices
from transferable_samplers.utils.chirality import ChiralitySignChecker

from plot_guided_euler_ramachandran import (
    EULER_STEPS,
    GUIDANCE_GAMMA,
    GUIDANCE_INIT_CONTROL,
    GUIDANCE_INNER_STEPS,
    GUIDANCE_LR,
    GUIDANCE_OPTIMIZER,
    GUIDANCE_W_CONTROL,
    GUIDANCE_W_TERMINAL,
    GUIDANCE_W_VF,
    OUT_DIR,
    SEQUENCE,
    load_model_and_data,
)
from plot_unguided_euler_ramachandran import EULER_STEPS as UNGUIDED_EULER_STEPS

# Note: this GIF always uses its own locally-defined smiley_cost_fn below,
# regardless of plot_guided_euler_ramachandran.py's current OBJECTIVE setting
# -- only the guidance hyperparameters (gamma, lr, w_terminal, etc.) are
# shared from that file.

NUM_SAMPLES = 1000
NUM_UNGUIDED_SAMPLES = 2000  # for the background density only
SEED = 42
NUM_FRAMES = 60  # subsampled from EULER_STEPS
GIF_PATH = f"{OUT_DIR}/guided_trajectory_smiley_full_space.gif"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, datamodule, num_atoms = load_model_and_data()
    model = model.to(device).eval()

    eval_ctx = datamodule.prepare_eval(sequence=SEQUENCE, stage="test")
    phi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="phi")
    psi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="psi")
    chirality_checker = ChiralitySignChecker(eval_ctx.topology, eval_ctx.true_data.samples[:1])

    # Reuse the same smiley cost fn as plot_guided_euler_ramachandran.py --
    # imported constants aren't enough since the cost fn itself is defined
    # inside that script's main(), so it's reconstructed here identically.
    from plot_guided_euler_ramachandran import (
        EYE_CENTERS,
        EYE_MOUTH_WEIGHT,
        EYE_RADIUS,
        FACE_CENTER,
        FACE_RADIUS,
        MOUTH_CENTERS,
        MOUTH_RADIUS,
    )
    from transferable_samplers.guidance.costs import repel_within_radius_penalty, within_radius_penalty

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

    def phi_psi_of(x1_flat: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = x1_flat.reshape(batch_size, num_atoms, -1).detach()
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        x1_fixed = x1 * sign
        phi = dihedrals(x1_fixed, phi_idx).squeeze(-1)
        psi = dihedrals(x1_fixed, psi_idx).squeeze(-1)
        return phi.cpu(), psi.cpu()

    net_c = copy.deepcopy(model.net)
    net_c.requires_grad_(False)
    eval_fn = partial(net_c, encodings=None)

    # -- Background: a quick unguided Euler pass (same model, same step count as
    # plot_unguided_euler_ramachandran.py) to get the natural equilibrium
    # distribution's phi/psi density.
    print(f"Computing unguided background density ({NUM_UNGUIDED_SAMPLES} samples, "
          f"{UNGUIDED_EULER_STEPS} steps)...")
    torch.manual_seed(SEED)
    z_bg = model.prior.sample(NUM_UNGUIDED_SAMPLES, num_atoms, device=device)
    xt_bg = z_bg.reshape(NUM_UNGUIDED_SAMPLES, -1)
    ts_bg = torch.linspace(0.0, 1.0, UNGUIDED_EULER_STEPS + 1, device=device)
    dt_bg = ts_bg[1] - ts_bg[0]
    with torch.no_grad():
        for i in range(UNGUIDED_EULER_STEPS):
            xt_bg = xt_bg + dt_bg * eval_fn(ts_bg[i], xt_bg)
    phi_bg, psi_bg = phi_psi_of(xt_bg, NUM_UNGUIDED_SAMPLES)
    print("done.\n")

    print("=== Guidance hyperparameters (from plot_guided_euler_ramachandran.py) ===")
    print(f"NUM_SAMPLES={NUM_SAMPLES} EULER_STEPS={EULER_STEPS} GUIDANCE_INNER_STEPS={GUIDANCE_INNER_STEPS} "
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
    ts = torch.linspace(0.0, 1.0, model.guidance_num_steps + 1, device=z.device)
    dt = ts[1] - ts[0]
    record_steps = set(np.linspace(0, model.guidance_num_steps - 1, NUM_FRAMES, dtype=int).tolist())

    xt = z.reshape(NUM_SAMPLES, -1)
    u_t_carry = None
    frames_phi, frames_psi, frames_t, frames_step = [], [], [], []

    print(f"Integrating {model.guidance_num_steps} guided Euler steps ({NUM_SAMPLES} samples), "
          f"recording {len(record_steps)} frames...")
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
            phi, psi = phi_psi_of(x1_pred, NUM_SAMPLES)
            frames_phi.append(phi.numpy())
            frames_psi.append(psi.numpy())
            frames_t.append(float(t_i))
            frames_step.append(i + 1)  # 1-indexed Euler step number, out of guidance_num_steps

        if (i + 1) % 50 == 0:
            print(f"  step {i + 1}/{model.guidance_num_steps}")

    # The loop's last recorded frame is x1_pred from the *second-to-last* Euler
    # step (i = num_steps - 1, t < 1) -- the model's own forward-looking
    # prediction, not the literal final integrated sample. Replace it with the
    # actual final `xt` (t = 1.0 exactly, what the real evaluation scripts use).
    phi_final, psi_final = phi_psi_of(xt, NUM_SAMPLES)
    frames_phi[-1] = phi_final.numpy()
    frames_psi[-1] = psi_final.numpy()
    frames_t[-1] = 1.0
    frames_step[-1] = model.guidance_num_steps

    print(f"done, {len(frames_t)} frames recorded.\n")

    # -- Background: same style as plot_ramachandran.py's own density plots
    # (hist2d, LogNorm, default viridis colormap) built from the unguided
    # equilibrium distribution -- the smiley should be visible from the
    # guided point density alone, on top of that backdrop.
    full_range = (-math.pi, math.pi)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.hist2d(
        phi_bg.numpy(), psi_bg.numpy(), 100, norm=LogNorm(), range=[full_range, full_range], alpha=0.4, zorder=1,
    )
    ax.set_xlim(*full_range)
    ax.set_ylim(*full_range)
    ax.set_xlabel(r"$\varphi$", fontsize=14)
    ax.set_ylabel(r"$\psi$", fontsize=14)
    ax.set_aspect("equal")
    scatter = ax.scatter([], [], s=12, c="red", alpha=0.8, edgecolors="none", zorder=5)
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
