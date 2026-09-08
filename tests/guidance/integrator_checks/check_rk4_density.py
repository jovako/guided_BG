"""Step 1 of the RK4-density validation protocol: alpha=0 must match stock dopri5 log q.

This is the master correctness check for rk4_density_integrator.py -- with guidance
switched off (alpha=0, n_inner=0), the RK4 integrator's log q should converge to
dopri5's as n_steps grows. Any discrepancy here is purely about the integrator
(step function, exact log-det convention, sign), decoupled from whether guidance
itself is implemented correctly.

Deliberately small (batch of a handful, and moderate step counts): the per-step
cost is a full (66, 66) Jacobian per sample via jacrev+vmap, which is far more
expensive than the existing Euler/dopri5 paths -- see rk4_density_integrator.py's
module docstring for the rough cost estimate.

Run with:
    uv run python tests/guidance/integrator_checks/check_rk4_density.py
"""

from __future__ import annotations

import time

import hydra
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from transferable_samplers.guidance.rk4_density_integrator import check_unguided_consistency
from transferable_samplers.utils.init_resume_utils import resolve_init

BATCH = 4
N_STEPS_LIST = (20, 40, 100, 200)
SEED = 42


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

    torch.manual_seed(SEED)
    z = model.prior.sample(BATCH, num_atoms, device=device)

    print(f"Running check_unguided_consistency: batch={BATCH}, n_steps={N_STEPS_LIST}, d={num_atoms * 3}")
    t0 = time.time()
    out = check_unguided_consistency(model, z, n_steps_list=N_STEPS_LIST)
    print(f"done in {time.time() - t0:.1f}s\n")

    for key, stats in out.items():
        print(f"{key!s:>10}: logq_mean={stats['logq_mean']:.4f}  logq_std={stats['logq_std']:.4f}")


if __name__ == "__main__":
    main()
