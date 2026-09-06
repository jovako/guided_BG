"""Unguided rejection-sampling baseline for the smiley constraint.

Same idea as rejection_baseline_positive_phi.py, generalized to the smiley
region: a sample is accepted if its (phi, psi) falls inside the face circle
and outside every eye/mouth exclusion circle -- i.e. exactly the
"frac_in_face_circle and not frac_in_eye_or_mouth" success condition used by
the guided smiley script (plot_guided_euler_ramachandran.py), just applied as
a hard accept/reject filter on unguided dopri5 samples instead of as a soft
guidance cost.

The smiley region is a small box (phi in PHI_TARGET, psi in PSI_TARGET)
overlapping only part of the model's natural density, further restricted to a
circle minus several small holes -- so expect a much lower acceptance rate
than the positive-phi baseline (that was ~50%; this could be well under 5%).
MAX_BATCHES is set high accordingly since each batch is cheap regardless of
how many samples in it are accepted; watch the running acceptance rate printed
per batch and lower TARGET_ACCEPTED (or raise MAX_BATCHES) if it's too rare.

Same Wasserstein/chirality conventions as rejection_baseline_positive_phi.py:
accepted samples' coordinates + energies are saved (not just summary stats),
since torus-w2/energy-w2 against a guided run need the actual per-sample
values from both sides.

Run with:
    uv run python tests/guidance/rejection_baseline_smiley.py
"""

from __future__ import annotations

import csv
import os

import hydra
import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from bootstrap import bootstrap_mean_ci

from transferable_samplers.evaluation.metrics.wasserstein_distances import energy_wasserstein, torus_wasserstein
from transferable_samplers.evaluation.plots.plot_ramachandran import plot_ramachandran
from transferable_samplers.guidance.observables import dihedrals, get_dihedral_atom_indices
from transferable_samplers.utils.chirality import ChiralitySignChecker
from transferable_samplers.utils.init_resume_utils import resolve_init
from transferable_samplers.utils.standardization import destandardize_coords

TARGET_ACCEPTED = 500
BATCH_SIZE = 512
MAX_BATCHES = 200  # smiley region is much rarer than positive-phi; batches are cheap so cap generously
SEED = 42
SEQUENCE = "Ace-A-Nme"
OUT_DIR = "tests/guidance/out"
PREFIX = "rejection_smiley"
SAMPLES_PATH = f"{OUT_DIR}/{PREFIX}_samples.pt"

# Same box target and derived smiley geometry as plot_guided_euler_ramachandran.py.
PHI_TARGET = (-2.0, -1.0)
PSI_TARGET = (-0.5, 0.5)
_BOX_CENTER = ((PHI_TARGET[0] + PHI_TARGET[1]) / 2, (PSI_TARGET[0] + PSI_TARGET[1]) / 2)
_BOX_HALF_EXTENT = min(PHI_TARGET[1] - PHI_TARGET[0], PSI_TARGET[1] - PSI_TARGET[0]) / 2
_SMILEY_SCALE = 0.9 * _BOX_HALF_EXTENT / 2.2
_INNER_SCALE = 0.5

FACE_CENTER = _BOX_CENTER
FACE_RADIUS = 2.2 * _SMILEY_SCALE
EYE_CENTERS = [
    (_BOX_CENTER[0] + dx * _INNER_SCALE * _SMILEY_SCALE, _BOX_CENTER[1] + dy * _INNER_SCALE * _SMILEY_SCALE)
    for dx, dy in [(-1.0, 1.0), (1.0, 1.0)]
]
EYE_RADIUS = 0.3 * _SMILEY_SCALE
MOUTH_PHIS = [-1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2]
MOUTH_CENTERS = [
    (
        _BOX_CENTER[0] + p * _INNER_SCALE * _SMILEY_SCALE,
        _BOX_CENTER[1] + (-1.3 + 0.2 * p**2) * _INNER_SCALE * _SMILEY_SCALE,
    )
    for p in MOUTH_PHIS
]
MOUTH_RADIUS = 0.2 * _SMILEY_SCALE


def load_model_and_data():
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="eval",
            overrides=["experiment=single_system/eval/ecnf++_Ace-A-Nme_snis"],
        )
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    model = hydra.utils.instantiate(cfg.model)
    state_dict = resolve_init(
        init_ckpt_path=cfg.get("ckpt_path"),
        init_hf_state_dict_path=cfg.get("hf_state_dict_path"),
        scratch_dir=cfg.paths.scratch_dir,
    )
    model.load_state_dict(state_dict)
    return model, datamodule, cfg.data.num_atoms


def in_smiley_region(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """True where (phi, psi) is inside the face circle and outside every eye/mouth hole."""
    dist_to_face = torch.sqrt((phi - FACE_CENTER[0]) ** 2 + (psi - FACE_CENTER[1]) ** 2)
    accept = dist_to_face <= FACE_RADIUS
    for cx, cy in EYE_CENTERS:
        d = torch.sqrt((phi - cx) ** 2 + (psi - cy) ** 2)
        accept = accept & (d > EYE_RADIUS)
    for cx, cy in MOUTH_CENTERS:
        d = torch.sqrt((phi - cx) ** 2 + (psi - cy) ** 2)
        accept = accept & (d > MOUTH_RADIUS)
    return accept


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, datamodule, num_atoms = load_model_and_data()
    model = model.to(device).eval()

    eval_ctx = datamodule.prepare_eval(sequence=SEQUENCE, stage="test")
    phi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="phi")
    psi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="psi")
    chirality_checker = ChiralitySignChecker(eval_ctx.topology, eval_ctx.true_data.samples[:1])

    torch.manual_seed(SEED)

    accepted_energy = []
    accepted_samples_physical = []
    total_accepted = 0
    total_drawn = 0
    batch_i = 0
    while total_accepted < TARGET_ACCEPTED and batch_i < MAX_BATCHES:
        z = model.prior.sample(BATCH_SIZE, num_atoms, device=device)
        with torch.no_grad():
            x, _ = model._integrate(model.net, z, encodings=None, reverse=False, compute_dlogp=False)

        x_phys = destandardize_coords(x.cpu(), eval_ctx.normalization_std)
        flip_mask = chirality_checker.flip_mask(x_phys)
        x_phys_fixed = x_phys.clone()
        x_phys_fixed[flip_mask] *= -1

        phi = dihedrals(x_phys_fixed, phi_idx).squeeze(-1)
        psi = dihedrals(x_phys_fixed, psi_idx).squeeze(-1)
        accept_mask = in_smiley_region(phi, psi)

        with torch.no_grad():
            e_batch = eval_ctx.target_energy.energy(x)

        accepted_energy.append(e_batch[accept_mask].cpu())
        accepted_samples_physical.append(x_phys_fixed[accept_mask])
        total_accepted += accept_mask.sum().item()
        total_drawn += BATCH_SIZE
        batch_i += 1
        print(
            f"batch {batch_i}: drawn={total_drawn} accepted={total_accepted} "
            f"(this batch: {accept_mask.float().mean():.2%}, running: {total_accepted / total_drawn:.2%})",
            flush=True,
        )

    acceptance_rate = total_accepted / total_drawn

    accepted_energy = torch.cat(accepted_energy, dim=0)[:TARGET_ACCEPTED]
    accepted_samples_physical = torch.cat(accepted_samples_physical, dim=0)[:TARGET_ACCEPTED]

    metrics = {
        f"{PREFIX}/num-drawn": total_drawn,
        f"{PREFIX}/num-accepted": len(accepted_energy),
        f"{PREFIX}/acceptance-rate": acceptance_rate,
    }
    if len(accepted_energy) > 0:
        metrics[f"{PREFIX}/mean-energy"] = accepted_energy.mean().item()
        metrics[f"{PREFIX}/median-energy"] = accepted_energy.median().item()
        if len(accepted_energy) >= 10:
            energy_ci = bootstrap_mean_ci(accepted_energy)
            metrics[f"{PREFIX}/mean-energy-se"] = energy_ci["se"]
            metrics[f"{PREFIX}/mean-energy-analytic-se"] = energy_ci["analytic_se"]
            metrics[f"{PREFIX}/mean-energy-ci-low"] = energy_ci["ci_low"]
            metrics[f"{PREFIX}/mean-energy-ci-high"] = energy_ci["ci_high"]
        else:
            print(f"Only {len(accepted_energy)} accepted samples -- skipping bootstrap CI (need >= 10).")
        metrics.update(
            energy_wasserstein(pred_energy=accepted_energy, true_energy=eval_ctx.true_data.E_target, prefix=PREFIX)
        )
        metrics.update(
            torus_wasserstein(eval_ctx.true_data.samples, accepted_samples_physical, eval_ctx.topology, prefix=PREFIX)
        )

    print()
    print("\n".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()))

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = f"{OUT_DIR}/{PREFIX}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())
    print(f"\nsaved {csv_path}")

    if len(accepted_energy) == 0:
        print("No samples accepted -- skipping sample save and plot.")
        return

    torch.save(
        {"samples_physical": accepted_samples_physical, "energy": accepted_energy, "acceptance_rate": acceptance_rate},
        SAMPLES_PATH,
    )
    print(f"saved {SAMPLES_PATH}")

    def log_image_fn(fig, name: str) -> None:
        path = f"{OUT_DIR}/{name.replace('/', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"saved {path}")
        plt.close(fig)

    plot_ramachandran(log_image_fn, accepted_samples_physical, eval_ctx.topology, prefix=PREFIX)

    if len(accepted_energy) < TARGET_ACCEPTED:
        print(
            f"\nWARNING: only accepted {len(accepted_energy)}/{TARGET_ACCEPTED} target samples "
            f"after {MAX_BATCHES} batches (acceptance rate {acceptance_rate:.2%}). "
            "Raise MAX_BATCHES if you need the full target count."
        )


if __name__ == "__main__":
    main()
