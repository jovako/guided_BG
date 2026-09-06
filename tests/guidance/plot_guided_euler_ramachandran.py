"""Guidance sanity check: steer alanine dipeptide samples into a phi/psi "smiley" shape.

Same setup as ``plot_unguided_euler_ramachandran.py`` (fixed-step Euler
integration, ECNF++ on Ace-A-Nme, no SNIS/importance-weighting), but with
``use_guidance=True`` and a terminal cost that (1) pulls samples to stay
within ``FACE_RADIUS`` of ``FACE_CENTER`` (``within_radius_penalty`` -- the
face outline) and (2) strongly repels them from small circles marking the
eyes and mouth (``repel_within_radius_penalty``, weighted by
``EYE_MOUTH_WEIGHT``) -- so density fills the disk everywhere except
smiley-shaped holes.

Two previous objectives (guide toward positive phi; a single box target) are
kept commented out in ``guidance_cost_fn`` for easy reuse.

Note on interpreting energy-w2/torus-w2 here: guidance deliberately biases the
distribution away from the true (unbiased) equilibrium ensemble toward the
face disk, so these are expected to get *worse* relative to the unguided
run -- they're reported as a diagnostic of how far guidance pushes the
distribution, not as a "is guidance good" score. The metrics that matter for
guidance itself are frac-in-face-circle and frac-in-eye-or-mouth below.

Run with:
    uv run python tests/guidance/plot_guided_euler_ramachandran.py
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
from transferable_samplers.guidance.costs import (  # noqa: F401
    box_quadratic_penalty,
    one_sided_quadratic_penalty,
    repel_within_radius_penalty,
    within_radius_penalty,
)
from transferable_samplers.guidance.observables import dihedrals, get_dihedral_atom_indices
from transferable_samplers.utils.chirality import ChiralitySignChecker
from transferable_samplers.utils.init_resume_utils import resolve_init
from transferable_samplers.utils.standardization import destandardize_coords

NUM_SAMPLES = 64
BATCH_SIZE = 64
SEED = 42
EULER_STEPS = 200
GUIDANCE_INNER_STEPS = 2
GUIDANCE_GAMMA = lambda t: 3.0 if t > 0.6 else 2.0 * t  # noqa: E731
GUIDANCE_LR = 1e-2
GUIDANCE_W_TERMINAL = 50.0
GUIDANCE_W_VF = 0.0
GUIDANCE_W_CONTROL = 0.01
GUIDANCE_INIT_CONTROL = "zero"  # zero at step 0, carried over from the previous step after
SEQUENCE = "Ace-A-Nme"
OUT_DIR = "tests/guidance/out"
PREFIX = "guided_smiley"

# Old box-target attempt (one blob at phi in [-2, -1], psi around center),
# kept for reuse by the commented-out guidance_cost_fn below.
PHI_TARGET = (-2.0, -1.0)
PSI_TARGET = (-0.5, 0.5)

# Smiley face: samples are pulled to stay inside FACE_RADIUS of FACE_CENTER
# (the face outline), and strongly repelled from small circles marking the
# eyes and mouth -- so density fills the disk except for smiley-shaped holes.
# Scaled + recentered to fit entirely inside the old box target above (same
# relative geometry as a full-size face with FACE_RADIUS=2.2, just shrunk).
_BOX_CENTER = ((PHI_TARGET[0] + PHI_TARGET[1]) / 2, (PSI_TARGET[0] + PSI_TARGET[1]) / 2)
_BOX_HALF_EXTENT = min(PHI_TARGET[1] - PHI_TARGET[0], PSI_TARGET[1] - PSI_TARGET[0]) / 2
_SMILEY_SCALE = 0.9 * _BOX_HALF_EXTENT / 2.2  # 2.2 was the original (unscaled) face radius

FACE_CENTER = _BOX_CENTER
FACE_RADIUS = 2.2 * _SMILEY_SCALE

# Eyes/mouth were sitting too close to the face's outer edge -- pull their
# offsets from FACE_CENTER inward (uniformly shrunk by _INNER_SCALE) before
# applying the overall smiley scale/recenter, so they land closer to the
# middle of the face. Radii are left alone -- only positions move.
_INNER_SCALE = 0.5

EYE_CENTERS = [
    (_BOX_CENTER[0] + dx * _INNER_SCALE * _SMILEY_SCALE, _BOX_CENTER[1] + dy * _INNER_SCALE * _SMILEY_SCALE)
    for dx, dy in [(-1.0, 1.0), (1.0, 1.0)]
]
EYE_RADIUS = 0.3 * _SMILEY_SCALE
# A shallow upward-curving arc (a "U") for the mouth, sampled as several
# small exclusion circles along the curve psi = -1.3 + 0.2 * phi**2
# (in coordinates relative to FACE_CENTER, before scaling).
MOUTH_PHIS = [-1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2]
MOUTH_CENTERS = [
    (
        _BOX_CENTER[0] + p * _INNER_SCALE * _SMILEY_SCALE,
        _BOX_CENTER[1] + (-1.3 + 0.2 * p**2) * _INNER_SCALE * _SMILEY_SCALE,
    )
    for p in MOUTH_PHIS
]
MOUTH_RADIUS = 0.2 * _SMILEY_SCALE
EYE_MOUTH_WEIGHT = 5.0


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
    phi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="phi")
    psi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="psi")
    # Precomputed once: get_symmetry_change rebuilds the bond adjacency list
    # from the mdtraj topology (a Python-level loop) on every call, which is
    # too slow to redo on every guidance step (hundreds of calls per batch).
    # ChiralitySignChecker does that topology parsing once; any single
    # correct-chirality frame works as the reference (chirality is a discrete
    # per-stereocenter invariant, not a continuous per-conformation quantity).
    # Sign comparisons are scale-invariant, so it's fine that the reference is
    # in physical units while x1 below is normalized.
    chirality_checker = ChiralitySignChecker(eval_ctx.topology, eval_ctx.true_data.samples[:1])

    """
    # --- previous objective: guide toward positive phi (kept for reuse) ---
    def guidance_cost_fn(x1: torch.Tensor) -> torch.Tensor:
    #     # EGNN is reflection-equivariant: some samples mid-generation are the
    #     # wrong (mirror-image) enantiomer, for which raw phi has the opposite
    #     # sign of the true, chirality-corrected phi. Without this correction,
    #     # guidance would push those samples' phi the wrong way.
         with torch.no_grad():
             flip_mask = chirality_checker.flip_mask(x1)
         sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
         phi = dihedrals(x1 * sign, phi_idx)  # (batch, num_phi)
         return one_sided_quadratic_penalty(phi, threshold=0.0, penalize_below=True).sum(dim=-1)

    """
    def guidance_cost_fn(x1: torch.Tensor) -> torch.Tensor:
        # EGNN is reflection-equivariant: some samples mid-generation are the
        # wrong (mirror-image) enantiomer, for which raw phi/psi have the
        # opposite sign of the true, chirality-corrected values. Without this
        # correction, guidance would push those samples the wrong way.
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        x1_fixed = x1 * sign

        phi = dihedrals(x1_fixed, phi_idx).squeeze(-1)
        psi = dihedrals(x1_fixed, psi_idx).squeeze(-1)

        dist_to_face = torch.sqrt((phi - FACE_CENTER[0]) ** 2 + (psi - FACE_CENTER[1]) ** 2)
        cost = within_radius_penalty(dist_to_face, FACE_RADIUS)

        for cx, cy in EYE_CENTERS:
            dist = torch.sqrt((phi - cx) ** 2 + (psi - cy) ** 2)
            cost = cost + EYE_MOUTH_WEIGHT * repel_within_radius_penalty(dist, EYE_RADIUS)

        for cx, cy in MOUTH_CENTERS:
            dist = torch.sqrt((phi - cx) ** 2 + (psi - cy) ** 2)
            cost = cost + EYE_MOUTH_WEIGHT * repel_within_radius_penalty(dist, MOUTH_RADIUS)

        return cost

    model.use_guidance = True
    model.guidance_cost_fn = guidance_cost_fn
    model.guidance_num_steps = EULER_STEPS
    model.guidance_inner_steps = GUIDANCE_INNER_STEPS
    model.guidance_gamma = GUIDANCE_GAMMA
    model.guidance_lr = GUIDANCE_LR
    model.guidance_w_terminal = GUIDANCE_W_TERMINAL
    model.guidance_w_vf = GUIDANCE_W_VF
    model.guidance_w_control = GUIDANCE_W_CONTROL
    model.guidance_init_control = GUIDANCE_INIT_CONTROL

    torch.manual_seed(SEED)
    samples = []
    num_batches = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(num_batches):
        n = min(BATCH_SIZE, NUM_SAMPLES - i * BATCH_SIZE)
        z = model.prior.sample(n, num_atoms, device=device)
        with torch.no_grad():
            x = model._integrate_guided(model.net, z, encodings=None)
        samples.append(x.cpu())
        print(f"batch {i + 1}/{num_batches} done ({sum(s.shape[0] for s in samples)}/{NUM_SAMPLES} samples)")

    samples = torch.cat(samples, dim=0)  # normalized space

    print("Computing target (OpenMM) energy of generated samples...")
    samples_physical = destandardize_coords(samples, eval_ctx.normalization_std)

    # EGNN is reflection-equivariant, so some samples come out as the wrong
    # (mirror-image) enantiomer -- flip them back to match the true chirality,
    # same as PeptideEnsembleEvaluator._fix_chirality does. Otherwise the
    # Ramachandran plot shows a spurious point-reflected population.
    flip_mask = chirality_checker.flip_mask(samples_physical)
    print(f"chirality: flipped {flip_mask.float().mean():.1%} of samples to match the true reference")
    samples_physical = samples_physical.clone()
    samples_physical[flip_mask] *= -1
    
    def log_image_fn(fig, name: str) -> None:
        path = f"{OUT_DIR}/{name.replace('/', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"saved {path}")
        plt.close(fig)

    plot_ramachandran(log_image_fn, samples_physical, eval_ctx.topology, prefix=PREFIX)

    with torch.no_grad():
        e_generated = eval_ctx.target_energy.energy(samples)

    phi_generated = dihedrals(samples_physical, phi_idx).squeeze(-1)
    psi_generated = dihedrals(samples_physical, psi_idx).squeeze(-1)

    dist_to_face = torch.sqrt((phi_generated - FACE_CENTER[0]) ** 2 + (psi_generated - FACE_CENTER[1]) ** 2)
    in_face_circle = dist_to_face <= FACE_RADIUS

    in_eye_or_mouth = torch.zeros_like(in_face_circle)
    for cx, cy in EYE_CENTERS:
        dist = torch.sqrt((phi_generated - cx) ** 2 + (psi_generated - cy) ** 2)
        in_eye_or_mouth = in_eye_or_mouth | (dist <= EYE_RADIUS)
    for cx, cy in MOUTH_CENTERS:
        dist = torch.sqrt((phi_generated - cx) ** 2 + (psi_generated - cy) ** 2)
        in_eye_or_mouth = in_eye_or_mouth | (dist <= MOUTH_RADIUS)


    metrics = {
        f"{PREFIX}/mean-energy": e_generated.mean().item(),
        f"{PREFIX}/correct-chirality-rate": 1 - flip_mask.float().mean().item(),
        f"{PREFIX}/frac-in-face-circle": in_face_circle.float().mean().item(),
        f"{PREFIX}/frac-in-eye-or-mouth": in_eye_or_mouth.float().mean().item(),
        f"{PREFIX}/mean-phi": phi_generated.mean().item(),
        f"{PREFIX}/mean-psi": psi_generated.mean().item(),
    }
    metrics.update(energy_wasserstein(pred_energy=e_generated, true_energy=eval_ctx.true_data.E_target, prefix=PREFIX))
    metrics.update(torus_wasserstein(eval_ctx.true_data.samples, samples_physical, eval_ctx.topology, prefix=PREFIX))

    print("\n".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

    baseline_csv = f"{OUT_DIR}/euler_unguided_metrics.csv"
    if os.path.exists(baseline_csv):
        with open(baseline_csv) as f:
            baseline = {row["metric"]: float(row["value"]) for row in csv.DictReader(f)}
        print("\nvs. unguided baseline:")
        for name in ["mean-energy", "energy-w2", "torus-w2"]:
            base_key, guided_key = f"euler_unguided/{name}", f"{PREFIX}/{name}"
            if base_key in baseline and guided_key in metrics:
                print(f"  {name}: {baseline[base_key]:.4f} -> {metrics[guided_key]:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = f"{OUT_DIR}/{PREFIX}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())
    print(f"saved {csv_path}")

if __name__ == "__main__":
    main()
