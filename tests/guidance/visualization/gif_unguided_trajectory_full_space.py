"""Animate how the *unguided* sampler's endpoint prediction wanders across the
FULL phi/psi Ramachandran space during a plain fixed-step Euler integration --
the no-guidance baseline counterpart to ``gif_guided_trajectory_full_space.py``
/ ``gif_guided_trajectory_full_space_pos_phi.py``, using the same EULER_STEPS
as ``plot_unguided_euler_ramachandran.py``.

Same visualization style: the whole [-pi, pi] x [-pi, pi] torus, plain
scatter -- since there's no guidance pulling samples anywhere, the point
cloud should spread out to fill the model's natural (unbiased) equilibrium
distribution rather than collapsing into any particular region. The only
drawn overlay is a box marking PHI_TARGET x PSI_TARGET, the same box the
smiley guidance geometry (in ``plot_guided_euler_ramachandran.py``) is scaled
and recentered to fit inside -- so it's easy to see how much of the natural
distribution already overlaps that region versus how much guidance has to
displace.

At every recorded step, this plots the phi/psi of the *free endpoint
estimate* ``x1_pred = x_t + (1-t)*v(x_t,t)`` -- the same quantity the guided
GIFs track -- not the raw noisy running state ``x_t`` itself. There's no
inner optimization loop here (no cost function, no control vector) since
there's no guidance: each step is just ``x_t += dt * v(x_t, t)``.

Run with:
    uv run python tests/guidance/visualization/gif_unguided_trajectory_full_space.py
"""

from __future__ import annotations

import copy
import math
from functools import partial

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from transferable_samplers.guidance.observables import dihedrals, get_dihedral_atom_indices
from transferable_samplers.utils.chirality import ChiralitySignChecker

from plot_guided_euler_ramachandran import PHI_TARGET, PSI_TARGET
from plot_unguided_euler_ramachandran import EULER_STEPS, OUT_DIR, SEQUENCE, load_model_and_data

NUM_SAMPLES = 2000
SEED = 42
NUM_FRAMES = 60  # subsampled from EULER_STEPS
GIF_PATH = f"{OUT_DIR}/unguided_trajectory_full_space.gif"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, datamodule, num_atoms = load_model_and_data()
    model = model.to(device).eval()

    eval_ctx = datamodule.prepare_eval(sequence=SEQUENCE, stage="test")
    phi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="phi")
    psi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="psi")
    chirality_checker = ChiralitySignChecker(eval_ctx.topology, eval_ctx.true_data.samples[:1])

    def phi_psi_of(x1_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = x1_flat.reshape(NUM_SAMPLES, num_atoms, -1).detach()
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        x1_fixed = x1 * sign
        phi = dihedrals(x1_fixed, phi_idx).squeeze(-1)
        psi = dihedrals(x1_fixed, psi_idx).squeeze(-1)
        return phi.cpu(), psi.cpu()

    print(f"=== Unguided Euler, no guidance: NUM_SAMPLES={NUM_SAMPLES} EULER_STEPS={EULER_STEPS} ===\n")

    torch.manual_seed(SEED)
    z = model.prior.sample(NUM_SAMPLES, num_atoms, device=device)

    net_c = copy.deepcopy(model.net)
    net_c.requires_grad_(False)
    eval_fn = partial(net_c, encodings=None)

    ts = torch.linspace(0.0, 1.0, EULER_STEPS + 1, device=z.device)
    dt = ts[1] - ts[0]
    record_steps = set(np.linspace(0, EULER_STEPS - 1, NUM_FRAMES, dtype=int).tolist())

    xt = z.reshape(NUM_SAMPLES, -1)
    frames_phi, frames_psi, frames_t, frames_step = [], [], [], []

    print(f"Integrating {EULER_STEPS} unguided Euler steps ({NUM_SAMPLES} samples), "
          f"recording {len(record_steps)} frames...")
    with torch.no_grad():
        for i in range(EULER_STEPS):
            t_i = ts[i]
            v = eval_fn(t_i, xt)
            x1_pred = xt + (1.0 - t_i) * v
            xt = xt + dt * v

            if i in record_steps:
                phi, psi = phi_psi_of(x1_pred)
                frames_phi.append(phi.numpy())
                frames_psi.append(psi.numpy())
                frames_t.append(float(t_i))
                frames_step.append(i + 1)  # 1-indexed Euler step number, out of EULER_STEPS

            if (i + 1) % 50 == 0:
                print(f"  step {i + 1}/{EULER_STEPS}")

    # The loop's last recorded frame is x1_pred from the *second-to-last* Euler
    # step (i = EULER_STEPS - 1, t < 1) -- the model's own forward-looking
    # prediction, not the literal final integrated sample. Replace it with the
    # actual final `xt` (t = 1.0 exactly, what the real evaluation scripts use).
    phi_final, psi_final = phi_psi_of(xt)
    frames_phi[-1] = phi_final.numpy()
    frames_psi[-1] = psi_final.numpy()
    frames_t[-1] = 1.0
    frames_step[-1] = EULER_STEPS

    print(f"done, {len(frames_t)} frames recorded.\n")

    # -- No background at all: the full [-pi, pi] x [-pi, pi] Ramachandran space.
    full_range = (-math.pi, math.pi)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_xlim(*full_range)
    ax.set_ylim(*full_range)
    ax.set_xlabel(r"$\varphi$", fontsize=14)
    ax.set_ylabel(r"$\psi$", fontsize=14)
    ax.set_aspect("equal")
    rect = mpatches.Rectangle(
        (PHI_TARGET[0], PSI_TARGET[0]),
        PHI_TARGET[1] - PHI_TARGET[0],
        PSI_TARGET[1] - PSI_TARGET[0],
        fill=False,
        edgecolor="#E8543A",
        linewidth=1.8,
        linestyle="--",
        zorder=4,
    )
    ax.add_patch(rect)
    ax.axvline(1.0, color="green", linewidth=1.5, linestyle="--", zorder=4)
    scatter = ax.scatter([], [], s=4, c="#1f5fa8", alpha=0.5, edgecolors="none", zorder=5)
    title = ax.set_title("")

    def update(frame_idx: int):
        scatter.set_offsets(np.column_stack([frames_phi[frame_idx], frames_psi[frame_idx]]))
        title.set_text(
            f"unguided endpoint prediction $x_1$ at t={frames_t[frame_idx]:.2f}  "
            f"(Euler step {frames_step[frame_idx]}/{EULER_STEPS})"
        )
        return scatter, title

    anim = FuncAnimation(fig, update, frames=len(frames_t), interval=120, blit=False)
    anim.save(GIF_PATH, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"saved {GIF_PATH}")


if __name__ == "__main__":
    main()
