"""Check that the new fixed-step Euler integrator (used for observable guidance)
reproduces the same ODE solution as the original adaptive dopri5 solver when
guidance is turned off.

``FlowMatchingModule._integrate_guided`` replaces the adaptive dopri5 solver
with a fixed-step Euler loop so a control vector can be optimized at every
step (see ``guidance_*`` hyperparameters). With ``guidance_inner_steps=0`` the
inner optimization never runs, so it reduces to a plain Euler integration of
the *unguided* velocity field -- this script checks that it converges to the
same trajectory as the original ``_integrate`` (dopri5, atol/rtol=1e-4) as the
number of Euler steps grows, using the trained ECNF++ checkpoint for alanine
dipeptide (Ace-A-Nme) and a fixed batch of prior samples.

Run with:
    uv run python tests/guidance/integrator_checks/compare_euler_vs_dopri5.py
"""

from __future__ import annotations

import hydra
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

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
        model.guidance_num_steps = num_steps
        model.guidance_inner_steps = 0  # no guidance -- reduces to plain Euler
        with torch.no_grad():
            x_new = model._integrate_guided(model.net, z.clone(), encodings=None)

        diff = x_new - x_ref
        rmse = diff.pow(2).mean().sqrt().item()
        max_abs = diff.abs().max().item()
        print(f"{num_steps:>12} | {rmse:>12.6f} | {max_abs:>14.6f}")


if __name__ == "__main__":
    main()
