"""Sanity check for the unguided fixed-step Euler integrator on alanine dipeptide.

Checks that ``generate_proposal_guided_euler`` with guidance off (alpha=0,
n_inner=0) produces a physically sensible equilibrium distribution: no
SNIS/importance-weighting, just raw proposal samples off the Euler solver.

Reports mean target energy, energy-W2 and torus-W2 to the true trajectory,
and a Ramachandran plot.

Run with:
    uv run python tests/guidance/visualization/plot_unguided_euler_ramachandran.py
"""

from __future__ import annotations

import csv
import os

import hydra
import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from transferable_samplers.evaluation.metrics.wasserstein_distances import energy_wasserstein, torus_wasserstein
from transferable_samplers.evaluation.plots.plot_ramachandran import plot_ramachandran
from transferable_samplers.guidance.euler_density_integrator import generate_proposal_guided_euler
from transferable_samplers.utils.chirality import get_symmetry_change
from transferable_samplers.utils.init_resume_utils import resolve_init
from transferable_samplers.utils.standardization import destandardize_coords

NUM_SAMPLES = 10000
BATCH_SIZE = 512
SEED = 42
EULER_STEPS = 200
SEQUENCE = "Ace-A-Nme"
OUT_DIR = "tests/guidance/out"
PREFIX = "euler_unguided"


def load_model_and_data():
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(
            config_name="eval",
            overrides=["experiment=single_system/eval/ecnf++_Ace-A-Nme_snis"],
        )

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()  # no-op if already downloaded

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

    torch.manual_seed(SEED)
    samples = []
    num_batches = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(num_batches):
        n = min(BATCH_SIZE, NUM_SAMPLES - i * BATCH_SIZE)
        x, _, _ = generate_proposal_guided_euler(
            model, n, num_atoms, lambda x1, t: (x1 * 0.0).sum(),
            alpha=0.0, n_inner=0, use_score_deviation=False, beta=0.0, lam=0.0,
            n_steps=EULER_STEPS, device=device, track_density=False,
        )
        x = x.detach()
        samples.append(x.cpu())
        print(f"batch {i + 1}/{num_batches} done ({sum(s.shape[0] for s in samples)}/{NUM_SAMPLES} samples)")

    samples = torch.cat(samples, dim=0)  # normalized space

    print("Computing target (OpenMM) energy of generated samples...")
    with torch.no_grad():
        e_generated = eval_ctx.target_energy.energy(samples)

    samples_physical = destandardize_coords(samples, eval_ctx.normalization_std)

    # Flip mirror-image (wrong-chirality) samples to match the true reference.
    flip_mask = get_symmetry_change(eval_ctx.true_data.samples, samples_physical, eval_ctx.topology)
    print(f"chirality: flipped {flip_mask.float().mean():.1%} of samples to match the true reference")
    samples_physical = samples_physical.clone()
    samples_physical[flip_mask] *= -1

    metrics = {f"{PREFIX}/mean-energy": e_generated.mean().item()}
    metrics[f"{PREFIX}/correct-chirality-rate"] = 1 - flip_mask.float().mean().item()
    metrics.update(energy_wasserstein(pred_energy=e_generated, true_energy=eval_ctx.true_data.E_target, prefix=PREFIX))
    metrics.update(
        torus_wasserstein(eval_ctx.true_data.samples, samples_physical, eval_ctx.topology, prefix=PREFIX)
    )

    print("\n".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = f"{OUT_DIR}/{PREFIX}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())
    print(f"saved {csv_path}")

    def log_image_fn(fig, name: str) -> None:
        path = f"{OUT_DIR}/{name.replace('/', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"saved {path}")
        plt.close(fig)

    plot_ramachandran(log_image_fn, samples_physical, eval_ctx.topology, prefix=PREFIX)


if __name__ == "__main__":
    main()
