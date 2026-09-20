"""Bulk sampling run: draws CHUNKS batches of BATCH exact-density guided-Euler
samples (smiley objective) and saves each chunk to disk as it completes, to
build a pool for SNIS reweighting later.

Each chunk needs the full exact-density Jacobian pass, so CHUNKS is sized to
fit a multi-hour budget. Resumable: chunks already saved on disk are skipped.

Only saves x / neg_logq / valid / the original (unconstrained) target energy,
not SNIS weights -- combining the energy with the guidance cost function
(needed so reweighting doesn't just undo the guidance) is deferred to a
later script; this one only builds the raw sample pool.

Run with:
    uv run python tests/guidance/sampling/sample_smiley_guided_snis_pool.py
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from transferable_samplers.guidance.costs import repel_within_radius_penalty, torus_distance, within_radius_penalty
from transferable_samplers.guidance.euler_density_integrator import generate_proposal_guided_euler
from transferable_samplers.guidance.observables import dihedrals, get_dihedral_atom_indices
from transferable_samplers.utils.chirality import ChiralitySignChecker
from transferable_samplers.utils.init_resume_utils import resolve_init
from transferable_samplers.utils.standardization import destandardize_coords

BATCH = 128
CHUNKS = 18
SEED_BASE = 10_000  # chunk i uses seed SEED_BASE + i -- distinct from SEED=42 used elsewhere
N_STEPS = 250
SEQUENCE = "Ace-A-Nme"
OUT_DIR = Path("tests/guidance/out/snis_pool_smiley")

# The 66%-valid config.
GAMMA_HI = 1.91
GAMMA_LO_SLOPE = 4.7
GAMMA_THRESHOLD = 0.408
GAMMA_DECAY_THRESHOLD = 0.805
GAMMA_DECAY_SLOPE_MAG = 7.18
ALPHA = 0.08863
W_TERMINAL = 22.4
EYE_MOUTH_WEIGHT = 5.0


def gamma_fn(t: torch.Tensor) -> float:
    if t <= GAMMA_THRESHOLD:
        return GAMMA_LO_SLOPE * t
    if t <= GAMMA_DECAY_THRESHOLD:
        return GAMMA_HI
    return max(0.0, GAMMA_HI - GAMMA_DECAY_SLOPE_MAG * (t - GAMMA_DECAY_THRESHOLD))


PHI_TARGET, PSI_TARGET = (-2.0, -1.0), (-0.5, 0.5)
_BOX_CENTER = ((PHI_TARGET[0] + PHI_TARGET[1]) / 2, (PSI_TARGET[0] + PSI_TARGET[1]) / 2)
_BOX_HALF = min(PHI_TARGET[1] - PHI_TARGET[0], PSI_TARGET[1] - PSI_TARGET[0]) / 2
_SCALE = 0.9 * _BOX_HALF / 2.2
_INNER = 0.75
FACE_CENTER = _BOX_CENTER
FACE_RADIUS = 2.2 * _SCALE
EYE_CENTERS = [(_BOX_CENTER[0] + dx * _INNER * _SCALE, _BOX_CENTER[1] + dy * _INNER * _SCALE) for dx, dy in [(-1, 1), (1, 1)]]
EYE_RADIUS = 0.4 * _SCALE
MOUTH_PHIS = [-1.5, -1.125, -0.75, -0.375, 0.0, 0.375, 0.75, 1.125, 1.5]
MOUTH_CENTERS = [
    (_BOX_CENTER[0] + p * _INNER * _SCALE, _BOX_CENTER[1] + (-1.5 + 0.35 * p**2) * _INNER * _SCALE) for p in MOUTH_PHIS
]
MOUTH_RADIUS = 0.35 * _SCALE


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    psi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="psi")
    chirality_checker = ChiralitySignChecker(eval_ctx.topology, eval_ctx.true_data.samples[:1])

    def terminal_cost(x1_flat: torch.Tensor) -> torch.Tensor:
        x1 = x1_flat.view(1, -1, 3)
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        x1_fixed = x1 * sign
        phi = dihedrals(x1_fixed, phi_idx).squeeze(-1)
        psi = dihedrals(x1_fixed, psi_idx).squeeze(-1)
        dist_to_face = torus_distance(phi, psi, FACE_CENTER[0], FACE_CENTER[1])
        cost = within_radius_penalty(dist_to_face, FACE_RADIUS)
        for cx, cy in EYE_CENTERS:
            dist = torus_distance(phi, psi, cx, cy)
            cost = cost + EYE_MOUTH_WEIGHT * repel_within_radius_penalty(dist, EYE_RADIUS)
        for cx, cy in MOUTH_CENTERS:
            dist = torus_distance(phi, psi, cx, cy)
            cost = cost + EYE_MOUTH_WEIGHT * repel_within_radius_penalty(dist, MOUTH_RADIUS)
        return cost.squeeze()

    run_config = dict(
        batch=BATCH, chunks=CHUNKS, seed_base=SEED_BASE, n_steps=N_STEPS,
        gamma_hi=GAMMA_HI, gamma_lo_slope=GAMMA_LO_SLOPE, gamma_threshold=GAMMA_THRESHOLD,
        gamma_decay_threshold=GAMMA_DECAY_THRESHOLD, gamma_decay_slope_mag=GAMMA_DECAY_SLOPE_MAG,
        alpha=ALPHA, w_terminal=W_TERMINAL, eye_mouth_weight=EYE_MOUTH_WEIGHT,
        face_center=FACE_CENTER, face_radius=FACE_RADIUS,
    )
    torch.save(run_config, OUT_DIR / "run_config.pt")

    total_valid, total_raw = 0, 0
    run_t0 = time.time()
    for i in range(CHUNKS):
        chunk_path = OUT_DIR / f"chunk_{i:03d}.pt"
        if chunk_path.exists():
            existing = torch.load(chunk_path, weights_only=False)
            n_valid = existing["valid"].sum().item()
            total_valid += n_valid
            total_raw += BATCH
            print(f"[chunk {i:02d}/{CHUNKS}] already on disk, skipping (valid={n_valid}/{BATCH})", flush=True)
            continue

        seed = SEED_BASE + i
        torch.manual_seed(seed)
        t0 = time.time()
        x, neg_logq, valid = generate_proposal_guided_euler(
            model, BATCH, cfg.data.num_atoms, lambda x1, t: W_TERMINAL * terminal_cost(x1),
            gamma=gamma_fn, alpha=ALPHA, n_inner=1, n_steps=N_STEPS,
            use_score_deviation=False, beta=0.0, lam=0.0,
            device=device, track_density=True, raise_on_orientation_failure=False,
        )
        elapsed = time.time() - t0
        x = x.detach().cpu()
        neg_logq = neg_logq.detach().cpu()
        valid = valid.cpu()

        x_phys = destandardize_coords(x, eval_ctx.normalization_std)
        flip_mask = chirality_checker.flip_mask(x_phys)
        x_phys = x_phys.clone()
        x_phys[flip_mask] *= -1

        with torch.no_grad():
            e_generated = eval_ctx.target_energy.energy(x.to(device)).cpu()

        n_valid = valid.sum().item()
        total_valid += n_valid
        total_raw += BATCH
        elapsed_total_h = (time.time() - run_t0) / 3600

        # Write to a temp path then rename -- atomic, so a kill mid-save never
        # leaves a corrupt chunk file that a resumed run would choke on.
        tmp_path = chunk_path.with_suffix(".tmp")
        torch.save(
            {
                "x": x, "x_phys": x_phys, "neg_logq": neg_logq, "valid": valid,
                "energy": e_generated,  # ORIGINAL (unconstrained) target energy -- not yet combined with the cost fn
                "seed": seed, "elapsed_s": elapsed,
            },
            tmp_path,
        )
        tmp_path.rename(chunk_path)

        print(
            f"[chunk {i:02d}/{CHUNKS}] elapsed={elapsed:.1f}s ({elapsed / N_STEPS:.2f}s/step) "
            f"valid={n_valid}/{BATCH} | running total: {total_valid}/{total_raw} valid "
            f"| {elapsed_total_h:.2f}h elapsed",
            flush=True,
        )
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nDone. {total_valid}/{total_raw} valid samples across {CHUNKS} chunks, saved to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
