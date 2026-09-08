"""How loose can dopri5's atol/rtol go while still matching the accuracy of the
200-step fixed unguided Euler baseline (energy-w2=8.647 vs the true reference,
see euler_unguided_metrics.csv)?

For each candidate tolerance, integrates unguided dopri5 (no dlogp -- just
samples, matching how the Euler baseline itself was evaluated), records the
resulting model.nfe, computes target (OpenMM) energy, and reports energy-w2
against the true reference distribution. Answers, concretely, "how many NFEs
does dopri5 actually need for the same accuracy" instead of assuming its
default tight tolerance (atol=rtol=1e-5) is the right comparison point.

Run with:
    uv run python tests/guidance/integrator_checks/sweep_dopri5_tolerance.py
"""

from __future__ import annotations

import hydra
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from transferable_samplers.evaluation.metrics.wasserstein_distances import energy_wasserstein
from transferable_samplers.utils.init_resume_utils import resolve_init

NUM_SAMPLES = 500
SEED = 42
SEQUENCE = "Ace-A-Nme"
# From loose to the current tight default.
TOLERANCES = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
TARGET_ENERGY_W2 = 8.647  # euler_unguided_metrics.csv, 200-step fixed Euler


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

    print(f"Target: energy-w2 <= {TARGET_ENERGY_W2:.3f} (200-step fixed unguided Euler)\n")
    print(f"{'atol=rtol':>10} | {'nfe':>6} | {'mean-energy':>12} | {'energy-w2':>10}")

    for tol in TOLERANCES:
        model.atol = tol
        model.rtol = tol
        model.nfe = 0

        torch.manual_seed(SEED)
        z = model.prior.sample(NUM_SAMPLES, num_atoms, device=device)

        with torch.no_grad():
            x, _ = model._integrate(model.net, z, encodings=None, reverse=False, compute_dlogp=False)
            e_generated = eval_ctx.target_energy.energy(x)

        w2 = energy_wasserstein(pred_energy=e_generated.cpu(), true_energy=eval_ctx.true_data.E_target, prefix="x")[
            "x/energy-w2"
        ]

        flag = " <= target!" if w2 <= TARGET_ENERGY_W2 else ""
        print(f"{tol:>10.0e} | {model.nfe:>6} | {e_generated.mean().item():>12.4f} | {w2:>10.4f}{flag}")


if __name__ == "__main__":
    main()
