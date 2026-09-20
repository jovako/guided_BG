"""Unguided rejection-sampling baseline for the "positive phi" constraint.

Reference point for judging guided sampling: draw plain (unguided) samples
via dopri5, keep only the ones satisfying phi > 0, report mean target energy
of that accepted subset.

Saves accepted samples' coordinates + energies (not just summary stats) so a
Wasserstein distance can later be computed directly against a guided run.

Energy is computed on the samples as generated; chirality-corrected phi is
only used to decide accept/reject. Saved/plotted coordinates ARE
chirality-corrected (torus-w2/Ramachandran geometry needs that).

Run with:
    uv run python tests/guidance/rejection_baselines/rejection_baseline_positive_phi.py
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

TARGET_ACCEPTED = 5000
BATCH_SIZE = 512
MAX_BATCHES = 40
SEED = 42
SEQUENCE = "Ace-A-Nme"
OUT_DIR = "tests/guidance/out"
PREFIX = "rejection_positive_phi"
SAMPLES_PATH = f"{OUT_DIR}/{PREFIX}_samples.pt"


def load_model_and_data():
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../../configs"):
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


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, datamodule, num_atoms = load_model_and_data()
    model = model.to(device).eval()

    eval_ctx = datamodule.prepare_eval(sequence=SEQUENCE, stage="test")
    phi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="phi")
    chirality_checker = ChiralitySignChecker(eval_ctx.topology, eval_ctx.true_data.samples[:1])

    torch.manual_seed(SEED)

    accepted_energy = []
    accepted_samples_physical = []  # chirality-corrected
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
        accept_mask = phi > 0

        with torch.no_grad():
            e_batch = eval_ctx.target_energy.energy(x)

        accepted_energy.append(e_batch[accept_mask].cpu())
        accepted_samples_physical.append(x_phys_fixed[accept_mask])
        total_accepted += accept_mask.sum().item()
        total_drawn += BATCH_SIZE
        batch_i += 1
        print(
            f"batch {batch_i}: drawn={total_drawn} accepted={total_accepted} "
            f"(this batch: {accept_mask.float().mean():.1%})",
            flush=True,
        )

    acceptance_rate = total_accepted / total_drawn

    accepted_energy = torch.cat(accepted_energy, dim=0)[:TARGET_ACCEPTED]
    accepted_samples_physical = torch.cat(accepted_samples_physical, dim=0)[:TARGET_ACCEPTED]

    energy_ci = bootstrap_mean_ci(accepted_energy)
    metrics = {
        f"{PREFIX}/num-drawn": total_drawn,
        f"{PREFIX}/num-accepted": len(accepted_energy),
        f"{PREFIX}/acceptance-rate": acceptance_rate,
        f"{PREFIX}/mean-energy": accepted_energy.mean().item(),
        f"{PREFIX}/mean-energy-se": energy_ci["se"],
        f"{PREFIX}/mean-energy-analytic-se": energy_ci["analytic_se"],
        f"{PREFIX}/mean-energy-ci-low": energy_ci["ci_low"],
        f"{PREFIX}/mean-energy-ci-high": energy_ci["ci_high"],
        f"{PREFIX}/median-energy": accepted_energy.median().item(),
    }
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
            f"after {MAX_BATCHES} batches (acceptance rate {acceptance_rate:.1%}). "
            "Raise MAX_BATCHES if you need the full target count."
        )


if __name__ == "__main__":
    main()
