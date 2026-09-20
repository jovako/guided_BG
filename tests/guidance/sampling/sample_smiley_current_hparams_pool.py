"""Bulk sampling run: draws CHUNKS batches of BATCH exact-density guided-Euler
samples (smiley objective), saving each chunk to disk as it completes, using
whichever guidance hyperparameters are currently live in
plot_guided_euler_ramachandran.py (imported directly, not a frozen snapshot,
unlike sample_smiley_guided_snis_pool.py).

Writes to a separate pool directory (snis_pool_smiley_current_hparams/) so
this never mixes with sample_smiley_guided_snis_pool.py's pool.

Resumable: chunks already saved on disk are skipped.

Run with:
    uv run python tests/guidance/sampling/sample_smiley_current_hparams_pool.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "tests/guidance/visualization")

from transferable_samplers.guidance.costs import repel_within_radius_penalty, torus_distance, within_radius_penalty
from transferable_samplers.guidance.euler_density_integrator import generate_proposal_guided_euler
from transferable_samplers.guidance.observables import dihedrals, get_dihedral_atom_indices
from transferable_samplers.utils.chirality import ChiralitySignChecker
from transferable_samplers.utils.standardization import destandardize_coords

from plot_guided_euler_ramachandran import (
    EULER_STEPS,
    EYE_CENTERS,
    EYE_MOUTH_WEIGHT,
    EYE_RADIUS,
    FACE_CENTER,
    FACE_RADIUS,
    GUIDANCE_GAMMA,
    GUIDANCE_INNER_STEPS,
    GUIDANCE_LR,
    GUIDANCE_W_TERMINAL,
    MOUTH_CENTERS,
    MOUTH_RADIUS,
    OUT_DIR,
    SEQUENCE,
    load_model_and_data,
)

BATCH = 128
CHUNKS = 17
SEED_BASE = 20_000  # distinct from sample_smiley_guided_snis_pool.py's 10_000
POOL_DIR = Path(f"{OUT_DIR}/snis_pool_smiley_current_hparams")


def main() -> None:
    POOL_DIR.mkdir(parents=True, exist_ok=True)

    model, datamodule, num_atoms = load_model_and_data()
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
        batch=BATCH, chunks=CHUNKS, seed_base=SEED_BASE, n_steps=EULER_STEPS,
        guidance_gamma_repr=repr(GUIDANCE_GAMMA), alpha=GUIDANCE_LR, w_terminal=GUIDANCE_W_TERMINAL,
        n_inner=GUIDANCE_INNER_STEPS, face_center=FACE_CENTER, face_radius=FACE_RADIUS,
    )
    torch.save(run_config, POOL_DIR / "run_config.pt")

    total_valid, total_raw = 0, 0
    run_t0 = time.time()
    for i in range(CHUNKS):
        chunk_path = POOL_DIR / f"chunk_{i:03d}.pt"
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
            model, BATCH, num_atoms, lambda x1, t: GUIDANCE_W_TERMINAL * terminal_cost(x1),
            gamma=GUIDANCE_GAMMA, alpha=GUIDANCE_LR, n_inner=GUIDANCE_INNER_STEPS,
            use_score_deviation=False, beta=0.0, lam=0.0, n_steps=EULER_STEPS,
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

        tmp_path = chunk_path.with_suffix(".tmp")
        torch.save(
            {
                "x": x, "x_phys": x_phys, "neg_logq": neg_logq, "valid": valid,
                "energy": e_generated, "seed": seed, "elapsed_s": elapsed,
            },
            tmp_path,
        )
        tmp_path.rename(chunk_path)

        print(
            f"[chunk {i:02d}/{CHUNKS}] elapsed={elapsed:.1f}s ({elapsed / EULER_STEPS:.2f}s/step) "
            f"valid={n_valid}/{BATCH} | running total: {total_valid}/{total_raw} valid "
            f"| {elapsed_total_h:.2f}h elapsed",
            flush=True,
        )
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nDone. {total_valid}/{total_raw} valid samples across {CHUNKS} chunks, saved to {POOL_DIR}/", flush=True)


if __name__ == "__main__":
    main()
