"""Check that the guided Euler integrator, with guidance off, reproduces the
same ODE solution as the adaptive dopri5 solver as the step count grows.

Run with:
    uv run python tests/guidance/integrator_checks/compare_euler_vs_dopri5.py
"""

from __future__ import annotations

import hydra
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from transferable_samplers.guidance.euler_density_integrator import generate_proposal_guided_euler
from transferable_samplers.utils.init_resume_utils import resolve_init

NUM_SAMPLES = 128
SEED = 42
EULER_STEP_COUNTS = [10, 25, 50, 100, 200, 400, 800]


def load_model() -> tuple[torch.nn.Module, int]:
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(
            config_name="eval",
            overrides=["experiment=single_system/eval/ecnf++_Ace-A-Nme_snis"],
        )

    model = hydra.utils.instantiate(cfg.model)
    state_dict = resolve_init(
        init_ckpt_path=cfg.get("ckpt_path"),
        init_hf_state_dict_path=cfg.get("hf_state_dict_path"),
        scratch_dir=cfg.paths.scratch_dir,
    )
    model.load_state_dict(state_dict)
    return model, cfg.data.num_atoms


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, num_atoms = load_model()
    model = model.to(device).eval()

    torch.manual_seed(SEED)
    z = model.prior.sample(NUM_SAMPLES, num_atoms, device=device)

    with torch.no_grad():
        x_ref, _ = model._integrate(model.net, z.clone(), encodings=None, reverse=False, compute_dlogp=False)

    print(f"reference: dopri5, atol=rtol={model.atol:g}, nfe={model.nfe}\n")
    print(f"{'euler steps':>12} | {'rmse':>12} | {'max abs diff':>14}")
    for num_steps in EULER_STEP_COUNTS:
        torch.manual_seed(SEED)  # reproduce the same z as above via model.prior.sample(...) internally
        x_new, _, _ = generate_proposal_guided_euler(
            model, NUM_SAMPLES, num_atoms, lambda x1, t: (x1 * 0.0).sum(),
            alpha=0.0, n_inner=0, use_score_deviation=False, beta=0.0, lam=0.0,
            n_steps=num_steps, device=device, track_density=False,
        )
        x_new = x_new.detach()

        diff = x_new - x_ref
        rmse = diff.pow(2).mean().sqrt().item()
        max_abs = diff.abs().max().item()
        print(f"{num_steps:>12} | {rmse:>12.6f} | {max_abs:>14.6f}")


if __name__ == "__main__":
    main()
