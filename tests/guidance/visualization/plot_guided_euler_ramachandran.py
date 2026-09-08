"""Guidance sanity check: steer alanine dipeptide samples toward a chosen objective.

Same setup as ``plot_unguided_euler_ramachandran.py`` (fixed-step Euler
integration, ECNF++ on Ace-A-Nme, no SNIS/importance-weighting), but with
``use_guidance=True``. Set OBJECTIVE below to switch the terminal cost:
    - "pos_phi": one-sided penalty, zero cost once phi > 0.
    - "phi_target": quadratic penalty, zero cost only exactly at phi=1.
    - "smiley": pulls samples to stay within FACE_RADIUS of FACE_CENTER (the
      face outline) and strongly repels them from small circles marking the
      eyes and mouth -- so density fills the disk everywhere except
      smiley-shaped holes.

Besides the usual energy-w2/torus-w2 to the *true* trajectory (expected to
get worse under guidance -- see note below), this also reports the
Wasserstein distance to the matching unguided rejection-sampling baseline
(rejection_baseline_positive_phi.py / rejection_baseline_smiley.py's saved
.pt of accepted samples+energies) -- i.e. how far the guided distribution is
from "what the model's own constrained distribution actually looks like".
That comparison works fine with different sample counts on each side:
energy_wasserstein/torus_wasserstein build a separate uniform weight vector
sized to *each* input's own count (see wasserstein_distances.py), so exact
optimal transport between differently-sized empirical distributions is just
what they already do -- no matching/subsampling needed.

Note on interpreting energy-w2/torus-w2 to the *true* trajectory: guidance
deliberately biases the distribution away from the true (unbiased)
equilibrium ensemble toward the objective region, so these are expected to
get *worse* relative to the unguided run -- they're a diagnostic of how far
guidance pushes the distribution, not a "is guidance good" score. The
distance to the rejection baseline is the more direct "is guidance good"
comparison, since both sides already satisfy the same constraint.

Run with:
    uv run python tests/guidance/visualization/plot_guided_euler_ramachandran.py
"""

from __future__ import annotations

import csv
import inspect
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

OBJECTIVE = "pos_phi"  # "pos_phi", "phi_target", or "smiley"

NUM_SAMPLES = 256
BATCH_SIZE = 64
SEED = 42
EULER_STEPS = 200
GUIDANCE_INNER_STEPS = 1
GUIDANCE_GAMMA = 2.
#GUIDANCE_GAMMA = 1.6
GUIDANCE_LR = 1.e-2    # should scale antiproportionally to EULER_STEPS, so 2e-3 for 600 steps is roughly equivalent to 1e-2 for 120 steps in plot_guided_euler_ramachandran.py
GUIDANCE_W_TERMINAL = 20.
GUIDANCE_W_VF = 0.0
GUIDANCE_W_CONTROL = 0.0
GUIDANCE_INIT_CONTROL = "zero"  # zero at step 0, carried over from the previous step after
GUIDANCE_OPTIMIZER = "gd"  # "adam" or "sgd" (for the inner-loop guidance optimization)
SEQUENCE = "Ace-A-Nme"
OUT_DIR = "tests/guidance/out"
PREFIX = f"euler_guided_{EULER_STEPS}_{OBJECTIVE}"

# Maps OBJECTIVE to the matching unguided rejection-sampling baseline's saved
# accepted samples+energies (see rejection_baseline_positive_phi.py /
# rejection_baseline_smiley.py). May not exist yet if that script hasn't run.
_REJECTION_PREFIX = {
    "pos_phi": "rejection_positive_phi",
    "phi_target": "rejection_phi_target",
    "smiley": "rejection_smiley",
}[OBJECTIVE]
REJECTION_SAMPLES_PATH = f"{OUT_DIR}/{_REJECTION_PREFIX}_samples.pt"

# Box target (one blob at phi in [-2, -1], psi around center) that the smiley
# geometry below is scaled + recentered to fit inside.
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

# Offsets from FACE_CENTER, uniformly scaled by _INNER_SCALE before applying
# the overall smiley scale/recenter -- 1.0 is the original (unscaled) layout;
# smaller values pull eyes/mouth toward the middle of the face (and shrink
# the mouth arc's spread/curvature along with them). Radii are left alone --
# only positions (and the mouth arc's size) move.
_INNER_SCALE = 0.75  # new (less centered eyes/mouth, bigger mouth arc)
#_INNER_SCALE = 0.5

EYE_CENTERS = [
    (_BOX_CENTER[0] + dx * _INNER_SCALE * _SMILEY_SCALE, _BOX_CENTER[1] + dy * _INNER_SCALE * _SMILEY_SCALE)
    for dx, dy in [(-1.0, 1.0), (1.0, 1.0)]
]
EYE_RADIUS = 0.4 * _SMILEY_SCALE  # new (bigger eyes)
#EYE_RADIUS = 0.3 * _SMILEY_SCALE
# A shallow upward-curving arc (a "U") for the mouth, sampled as several
# small exclusion circles along the curve psi = -1.3 + 0.2 * phi**2
# (in coordinates relative to FACE_CENTER, before scaling).
MOUTH_PHIS = [-1.5, -1.125, -0.75, -0.375, 0.0, 0.375, 0.75, 1.125, 1.5]  # new (broader smile)
#MOUTH_PHIS = [-1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2]
MOUTH_CENTERS = [
    (
        _BOX_CENTER[0] + p * _INNER_SCALE * _SMILEY_SCALE,
        _BOX_CENTER[1] + (-1.5 + 0.35 * p**2) * _INNER_SCALE * _SMILEY_SCALE,  # new (lower, thicker, more curved)
        #_BOX_CENTER[1] + (-1.3 + 0.2 * p**2) * _INNER_SCALE * _SMILEY_SCALE,
    )
    for p in MOUTH_PHIS
]
MOUTH_RADIUS = 0.35 * _SMILEY_SCALE  # new (thicker smile)
#MOUTH_RADIUS = 0.2 * _SMILEY_SCALE
EYE_MOUTH_WEIGHT = 5.0


def guidance_hyperparams() -> dict:
    """The actual guidance-algorithm knobs (model.guidance_* settings) -- what's worth
    seeing at a glance on stdout. Excludes run bookkeeping (sample/batch count, seed)
    and shape-specification constants (smiley geometry, EYE_MOUTH_WEIGHT isn't a
    guidance hyperparameter either -- it's part of the cost function's own shape).
    """
    try:
        gamma_repr = inspect.getsource(GUIDANCE_GAMMA).strip()
    except (OSError, TypeError):
        gamma_repr = repr(GUIDANCE_GAMMA)

    return {
        "EULER_STEPS": EULER_STEPS,
        "GUIDANCE_INNER_STEPS": GUIDANCE_INNER_STEPS,
        "GUIDANCE_GAMMA": gamma_repr,
        "GUIDANCE_LR": GUIDANCE_LR,
        "GUIDANCE_W_TERMINAL": GUIDANCE_W_TERMINAL,
        "GUIDANCE_W_VF": GUIDANCE_W_VF,
        "GUIDANCE_W_CONTROL": GUIDANCE_W_CONTROL,
        "GUIDANCE_INIT_CONTROL": GUIDANCE_INIT_CONTROL,
    }


def describe_hyperparams() -> dict:
    """Full run record for the saved CSV: guidance hyperparameters plus bookkeeping
    and (for "smiley") the shape-specification constants."""
    hparams = {"OBJECTIVE": OBJECTIVE, "NUM_SAMPLES": NUM_SAMPLES, "BATCH_SIZE": BATCH_SIZE, "SEED": SEED}
    hparams.update(guidance_hyperparams())
    if OBJECTIVE == "smiley":
        hparams.update(
            {
                "FACE_CENTER": FACE_CENTER,
                "FACE_RADIUS": FACE_RADIUS,
                "EYE_RADIUS": EYE_RADIUS,
                "MOUTH_RADIUS": MOUTH_RADIUS,
                "EYE_MOUTH_WEIGHT": EYE_MOUTH_WEIGHT,
            }
        )
    return hparams


def load_model_and_data():
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../../configs"):
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
    hparams = describe_hyperparams()
    print("=== Guidance hyperparameters ===")
    for k, v in guidance_hyperparams().items():
        print(f"{k}: {v}")
    print("=================================\n")

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

    def pos_phi_cost_fn(x1: torch.Tensor) -> torch.Tensor:
        # EGNN is reflection-equivariant: some samples mid-generation are the
        # wrong (mirror-image) enantiomer, for which raw phi has the opposite
        # sign of the true, chirality-corrected phi. Without this correction,
        # guidance would push those samples' phi the wrong way.
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        phi = dihedrals(x1 * sign, phi_idx)  # (batch, num_phi)
        return one_sided_quadratic_penalty(phi, threshold=0.0, penalize_below=True).sum(dim=-1)

    def phi_target_cost_fn(x1: torch.Tensor) -> torch.Tensor:
        # Same chirality correction as pos_phi_cost_fn. Zero only at phi=1,
        # quadratic penalty growing on both sides -- box_quadratic_penalty with
        # low=high=1.0 collapses to exactly (phi-1)**2, since only one of its
        # two relu terms is ever nonzero for a given phi.
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        phi = dihedrals(x1 * sign, phi_idx)  # (batch, num_phi)
        return box_quadratic_penalty(phi, low=1.0, high=1.0).sum(dim=-1)

    def smiley_cost_fn(x1: torch.Tensor) -> torch.Tensor:
        # Same chirality correction as pos_phi_cost_fn, applied to both phi and psi.
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

    guidance_cost_fn = {
        "pos_phi": pos_phi_cost_fn,
        "phi_target": phi_target_cost_fn,
        "smiley": smiley_cost_fn,
    }[OBJECTIVE]

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

    # Distance to the matching unguided rejection-sampling baseline (samples
    # that already satisfy the same constraint, drawn without guidance) --
    # the more direct "is guidance good" comparison than either side's
    # distance to the unconstrained true trajectory above. Works fine with a
    # different sample count on each side (see module docstring).
    if os.path.exists(REJECTION_SAMPLES_PATH):
        rejection_data = torch.load(REJECTION_SAMPLES_PATH, weights_only=False)
        rejection_samples_physical = rejection_data["samples_physical"]
        rejection_energy = rejection_data["energy"]
        metrics.update(
            energy_wasserstein(
                pred_energy=e_generated, true_energy=rejection_energy, prefix=f"{PREFIX}-vs-rejection"
            )
        )
        metrics.update(
            torus_wasserstein(
                rejection_samples_physical, samples_physical, eval_ctx.topology, prefix=f"{PREFIX}-vs-rejection"
            )
        )
    else:
        print(f"\nNo rejection baseline found at {REJECTION_SAMPLES_PATH} -- run the matching baseline script first.")

    print("\n".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = f"{OUT_DIR}/{PREFIX}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows((f"hparam/{k}", v) for k, v in hparams.items())
        writer.writerows(metrics.items())
    print(f"saved {csv_path}")

    samples_path = f"{OUT_DIR}/{PREFIX}_samples.pt"
    torch.save(
        {
            "samples": samples,  # normalized space, pre-chirality-fix
            "samples_physical": samples_physical,
            "energy": e_generated,
            "phi": phi_generated,
            "psi": psi_generated,
            "flip_mask": flip_mask,
            "hparams": hparams,
        },
        samples_path,
    )
    print(f"saved {samples_path}")


if __name__ == "__main__":
    main()
