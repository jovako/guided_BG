"""Does guided fixed-step Euler with EXACT density (euler_density_integrator.py)
actually converge samples toward the phi_target objective (see OBJECTIVE
docs in plot_guided_euler_ramachandran.py)? Sibling of
check_guided_euler_density.py (same structure, smiley -> phi_target cost).

Single batch (BATCH_SIZE=32), fixed alpha=6e-3 (no sweep). Also reports
orientation-preservation diagnostics: since euler_density_integrator.py's
``valid`` mask only tells you whether a sample stayed orientation-preserving
across ALL steps, not when it first failed, this passes an ``on_step``
callback to record, per step, how many samples have a non-positive
determinant that step -- printed as a per-step flip log plus a summary of
first-failure steps.

Run with:
    uv run python tests/guidance/integrator_checks/check_guided_euler_density_phi_target.py
"""

from __future__ import annotations

import time

import hydra
import numpy as np
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from transferable_samplers.guidance.costs import box_quadratic_penalty
from transferable_samplers.guidance.euler_density_integrator import generate_proposal_guided_euler
from transferable_samplers.guidance.observables import dihedrals, get_dihedral_atom_indices
from transferable_samplers.utils.chirality import ChiralitySignChecker
from transferable_samplers.utils.init_resume_utils import resolve_init
from transferable_samplers.utils.standardization import destandardize_coords

BATCH_SIZE = 32
N_STEPS = 200
SEED = 42
SEQUENCE = "Ace-A-Nme"

GAMMA = 2.0
ALPHA = 6e-3
W_TERMINAL = 20.0
LAST_STEP_GAMMA_SCALE = 0.  # scale gamma down on the final Euler step


def gamma_fn(t: torch.Tensor) -> float:
    last_step_t = (N_STEPS - 1) / N_STEPS
    return GAMMA * LAST_STEP_GAMMA_SCALE if float(t) >= last_step_t - 1e-9 else GAMMA

# Same bin-center nudge as plot_guided_euler_ramachandran.py's PHI_TARGET_ANGLE,
# so the "in target bin" metric below matches that script's convention.
_RAMA_BIN_EDGES = np.linspace(-np.pi, np.pi, 101)
_RAMA_BIN_CENTERS = (_RAMA_BIN_EDGES[:-1] + _RAMA_BIN_EDGES[1:]) / 2
PHI_TARGET_ANGLE = float(_RAMA_BIN_CENTERS[np.argmin(np.abs(_RAMA_BIN_CENTERS - 1.0))])


def _target_bin_range() -> tuple[float, float]:
    idx = int(np.searchsorted(_RAMA_BIN_EDGES, PHI_TARGET_ANGLE)) - 1
    return float(_RAMA_BIN_EDGES[idx]), float(_RAMA_BIN_EDGES[idx + 1])


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

    eval_ctx = datamodule.prepare_eval(sequence=SEQUENCE, stage="test")
    phi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="phi")
    chirality_checker = ChiralitySignChecker(eval_ctx.topology, eval_ctx.true_data.samples[:1])

    def terminal_cost(x1_flat: torch.Tensor) -> torch.Tensor:
        # Single-sample convention (see make_guided_field): x1_flat is (d,), not batched.
        x1 = x1_flat.view(1, -1, 3)
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        x1_fixed = x1 * sign
        phi = dihedrals(x1_fixed, phi_idx)  # (1, num_phi)
        # low=high=PHI_TARGET_ANGLE collapses box_quadratic_penalty to exactly
        # (phi - PHI_TARGET_ANGLE)**2 -- zero only exactly at the target.
        return box_quadratic_penalty(phi, low=PHI_TARGET_ANGLE, high=PHI_TARGET_ANGLE).sum(dim=-1).squeeze()

    bin_lo, bin_hi = _target_bin_range()

    # Per-step orientation-flip log: euler_density_integrator.py's own `valid`
    # mask only reports whether a sample stayed orientation-preserving across
    # ALL steps, not when it first failed -- on_step records that.
    first_failure_step = torch.full((BATCH_SIZE,), -1, dtype=torch.long)
    flip_events: list[tuple[int, int]] = []  # (step, num_newly_failed_this_step)

    def on_step(k: int, step_valid: torch.Tensor) -> None:
        newly_failed = (~step_valid.cpu()) & (first_failure_step < 0)
        n_new = int(newly_failed.sum().item())
        if n_new:
            first_failure_step[newly_failed] = k
            flip_events.append((k, n_new))

    torch.manual_seed(SEED)
    t0 = time.time()
    x, neg_logq, valid = generate_proposal_guided_euler(
        model, BATCH_SIZE, cfg.data.num_atoms, lambda x1, t: W_TERMINAL * terminal_cost(x1),
        gamma=gamma_fn, alpha=ALPHA, n_inner=1, n_steps=N_STEPS, use_score_deviation=False, beta=0.0,
        device=device, check_orientation=True, raise_on_orientation_failure=False, on_step=on_step,
    )
    elapsed = time.time() - t0

    x_phys = destandardize_coords(x.detach().cpu(), eval_ctx.normalization_std)
    chirality_flip_mask = chirality_checker.flip_mask(x_phys)
    x_phys = x_phys.clone()
    x_phys[chirality_flip_mask] *= -1
    phi = dihedrals(x_phys, phi_idx).squeeze(-1)
    mean_abs_phi_dev = (phi - PHI_TARGET_ANGLE).abs().mean().item()
    frac_in_target_bin = ((phi >= bin_lo) & (phi < bin_hi)).float().mean().item()

    with torch.no_grad():
        e = eval_ctx.target_energy.energy(x.to(device))

    print(f"gamma={GAMMA} (x{LAST_STEP_GAMMA_SCALE} at last step) alpha={ALPHA} w_term={W_TERMINAL} "
          f"n_steps={N_STEPS} batch={BATCH_SIZE} -> "
          f"elapsed={elapsed:.1f}s frac_in_target_bin={frac_in_target_bin:.3f} "
          f"mean_abs_phi_dev={mean_abs_phi_dev:.3g} mean_e={e.mean().item():.3g} "
          f"neg_logq_mean={neg_logq.mean().item():.3g}", flush=True)

    n_failed = int((~valid).sum().item())
    print(f"\nOrientation: {BATCH_SIZE - n_failed}/{BATCH_SIZE} stayed orientation-preserving for all {N_STEPS} steps")
    if flip_events:
        print(f"{n_failed} sample(s) failed at some point, across {len(flip_events)} distinct step(s):")
        for k, n_new in flip_events:
            print(f"  step {k:3d}/{N_STEPS}: {n_new} sample(s) newly failed orientation")
    else:
        print("No orientation flips occurred.")


if __name__ == "__main__":
    main()
