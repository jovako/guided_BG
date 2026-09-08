"""Unguided Euler vs dopri5: energy-w2/w1 and torus-w2, directly between the two
sample sets (not against the true MD reference) -- isolates discretization
error from model error.

Complements ``compare_euler_vs_dopri5.py`` (which reports raw per-sample RMSE
between the two trajectories): this instead reports how much the *distribution*
of physical (OpenMM) energies and phi/psi shifts when swapping dopri5 for a
fixed-step Euler solver, with guidance off in both cases (same z, same model).

Run with:
    uv run python tests/guidance/integrator_checks/compare_euler_vs_dopri5_metrics.py
"""

from __future__ import annotations

import hydra
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from transferable_samplers.evaluation.metrics.wasserstein_distances import energy_wasserstein, torus_wasserstein
from transferable_samplers.utils.chirality import get_symmetry_change
from transferable_samplers.utils.init_resume_utils import resolve_init
from transferable_samplers.utils.standardization import destandardize_coords

NUM_SAMPLES = 500
SEED = 42
EULER_STEPS = 200
SEQUENCE = "Ace-A-Nme"


def main() -> None:
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(config_name="eval", overrides=["experiment=single_system/eval/ecnf++_Ace-A-Nme_snis"])
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    model = hydra.utils.instantiate(cfg.model)
    state_dict = resolve_init(
        init_ckpt_path=cfg.get("ckpt_path"),
        init_hf_state_dict_path=cfg.get("hf_state_dict_path"),
        scratch_dir=cfg.paths.scratch_dir,
    )
    model.load_state_dict(state_dict)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    num_atoms = cfg.data.num_atoms

    eval_ctx = datamodule.prepare_eval(sequence=SEQUENCE, stage="test")

    torch.manual_seed(SEED)
    z = model.prior.sample(NUM_SAMPLES, num_atoms, device=device)

    print("Integrating dopri5 (reference)...")
    with torch.no_grad():
        x_dopri, _ = model._integrate(model.net, z.clone(), encodings=None, reverse=False, compute_dlogp=False)

    print(f"Integrating {EULER_STEPS}-step unguided Euler...")
    model.guidance_num_steps = EULER_STEPS
    model.guidance_inner_steps = 0  # no guidance -- reduces to plain Euler
    with torch.no_grad():
        x_euler = model._integrate_guided(model.net, z.clone(), encodings=None)

    print("Computing target (OpenMM) energies...")
    with torch.no_grad():
        e_dopri = eval_ctx.target_energy.energy(x_dopri)
        e_euler = eval_ctx.target_energy.energy(x_euler)

    dopri_physical = destandardize_coords(x_dopri.cpu(), eval_ctx.normalization_std)
    euler_physical = destandardize_coords(x_euler.cpu(), eval_ctx.normalization_std)

    flip_dopri = get_symmetry_change(eval_ctx.true_data.samples, dopri_physical, eval_ctx.topology)
    flip_euler = get_symmetry_change(eval_ctx.true_data.samples, euler_physical, eval_ctx.topology)
    dopri_physical = dopri_physical.clone()
    dopri_physical[flip_dopri] *= -1
    euler_physical = euler_physical.clone()
    euler_physical[flip_euler] *= -1
    print(f"chirality flips: dopri5={flip_dopri.float().mean():.1%}  euler={flip_euler.float().mean():.1%}")

    metrics = {}
    metrics["dopri5/mean-energy"] = e_dopri.mean().item()
    metrics[f"euler_{EULER_STEPS}/mean-energy"] = e_euler.mean().item()
    metrics.update(
        energy_wasserstein(pred_energy=e_euler.cpu(), true_energy=e_dopri.cpu(), prefix=f"euler_{EULER_STEPS}-vs-dopri5")
    )
    metrics.update(
        torus_wasserstein(dopri_physical, euler_physical, eval_ctx.topology, prefix=f"euler_{EULER_STEPS}-vs-dopri5")
    )

    print()
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
