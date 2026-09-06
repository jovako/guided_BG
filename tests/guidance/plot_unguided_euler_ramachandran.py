"""Sanity check for the unguided fixed-step Euler integrator on alanine dipeptide.

Checks that the new Euler integration path used for guidance
(``FlowMatchingModule._integrate_guided`` with ``guidance_inner_steps=0``, i.e.
no guidance) produces a physically sensible equilibrium distribution on its
own -- no SNIS/importance-weighting (no ``logw``, no effective-sample-size),
just raw proposal samples from the ECNF++ model straight off the Euler solver.

Reports:
    - mean target energy (OpenMM force-field energy of the generated
      conformations -- independent of the model's own log-density, which we
      don't track for the guided/Euler path; only importance-weighted metrics
      like ESS would need that)
    - energy-W2 to the true trajectory's energy distribution
    - torus-W2 on phi/psi to the true trajectory
    - a Ramachandran plot

Run with:
    uv run python scripts/plot_unguided_euler_ramachandran.py
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
    with initialize(version_base="1.3", config_path="../../configs"):
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

    model.guidance_num_steps = EULER_STEPS
    model.guidance_inner_steps = 0  # no guidance -- reduces to plain Euler

    torch.manual_seed(SEED)
    samples = []
    num_batches = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(num_batches):
        n = min(BATCH_SIZE, NUM_SAMPLES - i * BATCH_SIZE)
        z = model.prior.sample(n, num_atoms, device=device)
        with torch.no_grad():
            #x = model._integrate(model.net, z, encodings=None, compute_dlogp=False)[0]
            x = model._integrate_guided(model.net, z, encodings=None)
        samples.append(x.cpu())
        print(f"batch {i + 1}/{num_batches} done ({sum(s.shape[0] for s in samples)}/{NUM_SAMPLES} samples)")

    samples = torch.cat(samples, dim=0)  # normalized space

    print("Computing target (OpenMM) energy of generated samples...")
    with torch.no_grad():
        e_generated = eval_ctx.target_energy.energy(samples)

    samples_physical = destandardize_coords(samples, eval_ctx.normalization_std)

    # EGNN is reflection-equivariant, so some samples come out as the wrong
    # (mirror-image) enantiomer -- flip them back to match the true chirality,
    # same as PeptideEnsembleEvaluator._fix_chirality does. Otherwise the
    # Ramachandran plot shows a spurious point-reflected population.
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
