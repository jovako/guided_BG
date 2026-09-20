"""Guidance sanity check: steer alanine dipeptide samples toward a chosen objective.

Same setup as ``plot_unguided_euler_ramachandran.py``, but guided. Set
OBJECTIVE below to switch the terminal cost:
    - "pos_phi": one-sided penalty, zero cost once phi > 0.
    - "phi_target": quadratic penalty, zero only at phi=PHI_TARGET_ANGLE (1-D).
    - "phi_psi_target": periodic quadratic penalty, zero only at a single
      (phi, psi) point (torus_distance, wraps at +-pi).
    - "smiley": attracts within FACE_RADIUS of FACE_CENTER, repels from small
      eye/mouth circles.
    - "smiley_reference": periodic quadratic bowl centered on a moving target
      (a fixed reference sample's own endpoint prediction at the same t,
      see linear_coupling_trajectory_smiley_center.py) until
      SMILEY_REFERENCE_T_SWITCH, then reverts to "smiley".

Also reports the Wasserstein distance to the matching unguided
rejection-sampling baseline (rejection_baseline_*.py's saved samples), which
is the more direct "is guidance good" comparison than energy-w2/torus-w2 to
the true trajectory (guidance deliberately biases away from that).

Run with:
    uv run python tests/guidance/visualization/plot_guided_euler_ramachandran.py
"""

from __future__ import annotations

import csv
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from transferable_samplers.evaluation.metrics.wasserstein_distances import energy_wasserstein, torus_wasserstein
from transferable_samplers.evaluation.plots.plot_ramachandran import plot_ramachandran
from transferable_samplers.guidance.costs import (  # noqa: F401
    box_quadratic_penalty,
    one_sided_quadratic_penalty,
    quadratic_target_penalty,
    repel_within_radius_penalty,
    torus_distance,
    within_radius_penalty,
)
from transferable_samplers.guidance.euler_density_integrator import generate_proposal_guided_euler
from transferable_samplers.guidance.observables import dihedrals, get_dihedral_atom_indices
from transferable_samplers.utils.chirality import ChiralitySignChecker
from transferable_samplers.utils.init_resume_utils import resolve_init
from transferable_samplers.utils.standardization import destandardize_coords

# Not auto-applied -- set the constants below by hand to match OBJECTIVE.
OBJECTIVE = "smiley"  # "pos_phi", "phi_target", "phi_psi_target", "smiley", or "smiley_reference"

NUM_SAMPLES = 64
BATCH_SIZE = 64
SEED = 42
EULER_STEPS = 250
GUIDANCE_INNER_STEPS = 1

# Gamma schedule: rise (0 -> GAMMA_HI, slope GAMMA_LO_SLOPE) up to
# GAMMA_THRESHOLD, then a plateau at GAMMA_HI, then (if GAMMA_USE_DECAY) a
# decline back toward 0 starting at GAMMA_DECAY_THRESHOLD.
GAMMA_HI = 2.0
GAMMA_LO_SLOPE = 12.
GAMMA_THRESHOLD = 0.
GAMMA_USE_DECAY = False
GAMMA_DECAY_THRESHOLD = 0.7
GAMMA_DECAY_SLOPE_MAG = 10.0


def _gamma_fn(t: torch.Tensor) -> float:
    if t <= GAMMA_THRESHOLD:
        return GAMMA_LO_SLOPE * t
    if not GAMMA_USE_DECAY or t <= GAMMA_DECAY_THRESHOLD:
        return GAMMA_HI
    return max(0.0, GAMMA_HI - GAMMA_DECAY_SLOPE_MAG * (t - GAMMA_DECAY_THRESHOLD))


GUIDANCE_GAMMA = _gamma_fn
GUIDANCE_LR = 2.e-3
GUIDANCE_W_TERMINAL = 1.0
GUIDANCE_W_VF = 0.
GUIDANCE_W_CONTROL = 0.0
SEQUENCE = "Ace-A-Nme"
OUT_DIR = "tests/guidance/out"
PREFIX = f"euler_guided_{EULER_STEPS}_{OBJECTIVE}"

# Matching unguided rejection-sampling baseline's saved samples per OBJECTIVE
# (see rejection_baseline_*.py). May not exist yet if that script hasn't run.
_REJECTION_PREFIX = {
    "pos_phi": "rejection_positive_phi",
    "phi_target": "rejection_phi_target",
    "phi_psi_target": "rejection_phi_psi_target",
    "smiley": "rejection_smiley",
    "smiley_reference": "rejection_smiley",
}[OBJECTIVE]
REJECTION_SAMPLES_PATH = f"{OUT_DIR}/{_REJECTION_PREFIX}_samples.pt"

# For OBJECTIVE == "smiley_reference": fixed (x0, x1) reference pair from
# linear_coupling_trajectory_smiley_center.py, and the t at which guidance
# switches from tracking its endpoint prediction to the plain smiley cost.
REFERENCE_TRAJECTORY_PATH = f"{OUT_DIR}/linear_coupling_trajectory_smiley_center.pt"
SMILEY_REFERENCE_T_SWITCH = 0.5

# Single-point target for OBJECTIVE == "phi_psi_target" (radians, before
# alignment to the Ramachandran histogram grid).
PHI_PSI_TARGET = (-2.0, 1.5)

# plot_ramachandran's histogram bins (100 over [-pi, pi]) -- nudges each
# target onto its nearest bin center so plotted lines/metrics land cleanly.
_RAMA_BIN_EDGES = np.linspace(-np.pi, np.pi, 101)
_RAMA_BIN_CENTERS = (_RAMA_BIN_EDGES[:-1] + _RAMA_BIN_EDGES[1:]) / 2


def _nearest_rama_bin_center(value: float) -> float:
    return float(_RAMA_BIN_CENTERS[np.argmin(np.abs(_RAMA_BIN_CENTERS - value))])


def _rama_bin_range(value: float) -> tuple[float, float]:
    """The [lo, hi) edges of the histogram bin containing ``value``."""
    idx = int(np.searchsorted(_RAMA_BIN_EDGES, value)) - 1
    return float(_RAMA_BIN_EDGES[idx]), float(_RAMA_BIN_EDGES[idx + 1])


# 1-D target for OBJECTIVE == "phi_target": nudged off exactly 1.0 to the
# center of the nearest histogram bin.
PHI_TARGET_ANGLE = _nearest_rama_bin_center(1.0)

# 2-D target for OBJECTIVE == "phi_psi_target": PHI_PSI_TARGET, with each
# coordinate independently nudged to the center of its nearest histogram bin.
PHI_PSI_TARGET_ANGLE = (_nearest_rama_bin_center(PHI_PSI_TARGET[0]), _nearest_rama_bin_center(PHI_PSI_TARGET[1]))

# Box target (one blob at phi in [-2, -1], psi around center) that the smiley
# geometry below is scaled + recentered to fit inside.
PHI_TARGET = (-2.0, -1.0)
PSI_TARGET = (-0.5, 0.5)

# Smiley face: samples are pulled to stay inside FACE_RADIUS of FACE_CENTER,
# and repelled from small circles marking the eyes and mouth. Scaled +
# recentered to fit inside the box target above.
_BOX_CENTER = ((PHI_TARGET[0] + PHI_TARGET[1]) / 2, (PSI_TARGET[0] + PSI_TARGET[1]) / 2)
_BOX_HALF_EXTENT = min(PHI_TARGET[1] - PHI_TARGET[0], PSI_TARGET[1] - PSI_TARGET[0]) / 2
_SMILEY_SCALE = 0.9 * _BOX_HALF_EXTENT / 2.2

FACE_CENTER = _BOX_CENTER
FACE_RADIUS = 2.2 * _SMILEY_SCALE

# Offsets from FACE_CENTER, scaled by _INNER_SCALE before the overall
# scale/recenter -- smaller values pull eyes/mouth toward the face center.
_INNER_SCALE = 0.75

EYE_CENTERS = [
    (_BOX_CENTER[0] + dx * _INNER_SCALE * _SMILEY_SCALE, _BOX_CENTER[1] + dy * _INNER_SCALE * _SMILEY_SCALE)
    for dx, dy in [(-1.0, 1.0), (1.0, 1.0)]
]
EYE_RADIUS = 0.4 * _SMILEY_SCALE
# Mouth: a shallow upward "U" arc, sampled as small exclusion circles along
# psi = -1.5 + 0.35*phi**2 (relative to FACE_CENTER, before scaling).
MOUTH_PHIS = [-1.5, -1.125, -0.75, -0.375, 0.0, 0.375, 0.75, 1.125, 1.5]
MOUTH_CENTERS = [
    (
        _BOX_CENTER[0] + p * _INNER_SCALE * _SMILEY_SCALE,
        _BOX_CENTER[1] + (-1.5 + 0.35 * p**2) * _INNER_SCALE * _SMILEY_SCALE,
    )
    for p in MOUTH_PHIS
]
MOUTH_RADIUS = 0.35 * _SMILEY_SCALE
EYE_MOUTH_WEIGHT = 5.0


def guidance_hyperparams() -> dict:
    """Guidance-algorithm knobs worth printing at a glance; excludes run
    bookkeeping and shape-specification constants (smiley geometry)."""
    return {
        "EULER_STEPS": EULER_STEPS,
        "GUIDANCE_INNER_STEPS": GUIDANCE_INNER_STEPS,
        "GAMMA_HI": GAMMA_HI,
        "GAMMA_LO_SLOPE": GAMMA_LO_SLOPE,
        "GAMMA_THRESHOLD": GAMMA_THRESHOLD,
        "GAMMA_USE_DECAY": GAMMA_USE_DECAY,
        "GAMMA_DECAY_THRESHOLD": GAMMA_DECAY_THRESHOLD if GAMMA_USE_DECAY else None,
        "GAMMA_DECAY_SLOPE_MAG": GAMMA_DECAY_SLOPE_MAG if GAMMA_USE_DECAY else None,
        "GUIDANCE_LR": GUIDANCE_LR,
        "GUIDANCE_W_TERMINAL": GUIDANCE_W_TERMINAL,
        "GUIDANCE_W_VF": GUIDANCE_W_VF,
        "GUIDANCE_W_CONTROL": GUIDANCE_W_CONTROL,
    }


def describe_hyperparams() -> dict:
    """Full run record for the saved CSV: guidance hyperparameters plus bookkeeping
    and shape-specification constants."""
    hparams = {"OBJECTIVE": OBJECTIVE, "NUM_SAMPLES": NUM_SAMPLES, "BATCH_SIZE": BATCH_SIZE, "SEED": SEED}
    hparams.update(guidance_hyperparams())
    if OBJECTIVE in ("smiley", "smiley_reference"):
        hparams.update(
            {
                "FACE_CENTER": FACE_CENTER,
                "FACE_RADIUS": FACE_RADIUS,
                "EYE_RADIUS": EYE_RADIUS,
                "MOUTH_RADIUS": MOUTH_RADIUS,
                "EYE_MOUTH_WEIGHT": EYE_MOUTH_WEIGHT,
            }
        )
    if OBJECTIVE == "smiley_reference":
        hparams.update(
            {
                "SMILEY_REFERENCE_T_SWITCH": SMILEY_REFERENCE_T_SWITCH,
                "REFERENCE_TRAJECTORY_PATH": REFERENCE_TRAJECTORY_PATH,
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
    # Precomputed once -- rebuilding the bond adjacency list from the mdtraj
    # topology on every guidance step would be too slow.
    chirality_checker = ChiralitySignChecker(eval_ctx.topology, eval_ctx.true_data.samples[:1])

    def pos_phi_cost_fn(x1: torch.Tensor) -> torch.Tensor:
        # chirality-correct phi before penalizing (EGNN is reflection-equivariant)
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        phi = dihedrals(x1 * sign, phi_idx)  # (batch, num_phi)
        return one_sided_quadratic_penalty(phi, threshold=0.0, penalize_below=True).sum(dim=-1)

    def phi_target_cost_fn(x1: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        phi = dihedrals(x1 * sign, phi_idx)  # (batch, num_phi)
        return box_quadratic_penalty(phi, low=PHI_TARGET_ANGLE, high=PHI_TARGET_ANGLE).sum(dim=-1)

    def phi_psi_target_cost_fn(x1: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        x1_fixed = x1 * sign
        phi = dihedrals(x1_fixed, phi_idx).squeeze(-1)
        psi = dihedrals(x1_fixed, psi_idx).squeeze(-1)
        dist = torus_distance(phi, psi, PHI_PSI_TARGET_ANGLE[0], PHI_PSI_TARGET_ANGLE[1])
        return quadratic_target_penalty(dist)

    def smiley_cost_fn(x1: torch.Tensor) -> torch.Tensor:
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

        return cost

    if OBJECTIVE == "smiley_reference":
        _ref_data = torch.load(REFERENCE_TRAJECTORY_PATH, weights_only=False)
        _x0_ref = _ref_data["x0"].to(device)  # (1, atoms, dims), normalized space
        _x1_ref = _ref_data["x1"].to(device)

    def smiley_reference_cost_fn(x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t > SMILEY_REFERENCE_T_SWITCH:
            return smiley_cost_fn(x1)

        # Periodic quadratic bowl centered on the reference point's own
        # endpoint prediction at this same t (a moving target).
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        x1_fixed = x1 * sign
        phi_cur = dihedrals(x1_fixed, phi_idx).squeeze(-1)
        psi_cur = dihedrals(x1_fixed, psi_idx).squeeze(-1)

        with torch.no_grad():
            x_t_ref = (1.0 - t) * _x0_ref + t * _x1_ref  # (1, atoms, dims)
            v_ref = model.net(t.reshape(1), x_t_ref.reshape(1, -1), encodings=None).reshape_as(x_t_ref)
            x1_hat_ref = x_t_ref + (1.0 - t) * v_ref
            flip_ref = chirality_checker.flip_mask(x1_hat_ref)
            sign_ref = torch.where(flip_ref, -1.0, 1.0).to(x1_hat_ref)[:, None, None]
            x1_hat_ref_fixed = x1_hat_ref * sign_ref
            phi_ref = dihedrals(x1_hat_ref_fixed, phi_idx).squeeze(-1)
            psi_ref = dihedrals(x1_hat_ref_fixed, psi_idx).squeeze(-1)

        dist = torus_distance(phi_cur, psi_cur, phi_ref, psi_ref)
        return quadratic_target_penalty(dist)

    guidance_cost_fn = {
        "pos_phi": pos_phi_cost_fn,
        "phi_target": phi_target_cost_fn,
        "phi_psi_target": phi_psi_target_cost_fn,
        "smiley_reference": smiley_reference_cost_fn,
        "smiley": smiley_cost_fn,
    }[OBJECTIVE]

    torch.manual_seed(SEED)
    samples = []
    num_batches = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE

    # Wrap the batched guidance_cost_fn into the single-sample convention
    # generate_proposal_guided_euler expects. Only smiley_reference uses t.
    if OBJECTIVE == "smiley_reference":

        def single_sample_cost_fn(x1_flat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return guidance_cost_fn(x1_flat.view(1, -1, 3), t).squeeze()
    else:

        def single_sample_cost_fn(x1_flat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return guidance_cost_fn(x1_flat.view(1, -1, 3)).squeeze()

    for i in range(num_batches):
        n = min(BATCH_SIZE, NUM_SAMPLES - i * BATCH_SIZE)
        x, _, _ = generate_proposal_guided_euler(
            model, n, num_atoms, lambda x1, t: GUIDANCE_W_TERMINAL * single_sample_cost_fn(x1, t),
            gamma=GUIDANCE_GAMMA, alpha=GUIDANCE_LR, lam=GUIDANCE_W_CONTROL,
            beta=GUIDANCE_W_VF, use_score_deviation=GUIDANCE_W_VF != 0.0,
            n_inner=GUIDANCE_INNER_STEPS, n_steps=EULER_STEPS,
            device=device, track_density=False,
        )
        samples.append(x.detach().cpu())
        print(f"batch {i + 1}/{num_batches} done ({sum(s.shape[0] for s in samples)}/{NUM_SAMPLES} samples)")

    samples = torch.cat(samples, dim=0)  # normalized space

    print("Computing target (OpenMM) energy of generated samples...")
    samples_physical = destandardize_coords(samples, eval_ctx.normalization_std)

    # Flip mirror-image (wrong-chirality) samples to match the true reference.
    flip_mask = chirality_checker.flip_mask(samples_physical)
    print(f"chirality: flipped {flip_mask.float().mean():.1%} of samples to match the true reference")
    samples_physical = samples_physical.clone()
    samples_physical[flip_mask] *= -1
    
    def log_image_fn(fig, name: str) -> None:
        path = f"{OUT_DIR}/{name.replace('/', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"saved {path}")
        plt.close(fig)

    plot_ramachandran(
        log_image_fn,
        samples_physical,
        eval_ctx.topology,
        prefix=PREFIX,
        phi_target=PHI_TARGET_ANGLE if OBJECTIVE == "phi_target" else None,
    )

    with torch.no_grad():
        e_generated = eval_ctx.target_energy.energy(samples)

    phi_generated = dihedrals(samples_physical, phi_idx).squeeze(-1)
    psi_generated = dihedrals(samples_physical, psi_idx).squeeze(-1)

    dist_to_face = torus_distance(phi_generated, psi_generated, FACE_CENTER[0], FACE_CENTER[1])
    in_face_circle = dist_to_face <= FACE_RADIUS

    in_eye_or_mouth = torch.zeros_like(in_face_circle)
    for cx, cy in EYE_CENTERS:
        dist = torus_distance(phi_generated, psi_generated, cx, cy)
        in_eye_or_mouth = in_eye_or_mouth | (dist <= EYE_RADIUS)
    for cx, cy in MOUTH_CENTERS:
        dist = torus_distance(phi_generated, psi_generated, cx, cy)
        in_eye_or_mouth = in_eye_or_mouth | (dist <= MOUTH_RADIUS)


    metrics = {
        f"{PREFIX}/mean-energy": e_generated.mean().item(),
        f"{PREFIX}/correct-chirality-rate": 1 - flip_mask.float().mean().item(),
        f"{PREFIX}/frac-in-face-circle": in_face_circle.float().mean().item(),
        f"{PREFIX}/frac-in-eye-or-mouth": in_eye_or_mouth.float().mean().item(),
        f"{PREFIX}/mean-phi": phi_generated.mean().item(),
        f"{PREFIX}/mean-psi": psi_generated.mean().item(),
    }

    if OBJECTIVE in ("phi_target", "phi_psi_target"):
        # Fraction outside the target's own Ramachandran histogram bin, plus
        # mean energy of the in-bin samples.
        phi_lo, phi_hi = _rama_bin_range(PHI_TARGET_ANGLE if OBJECTIVE == "phi_target" else PHI_PSI_TARGET_ANGLE[0])
        in_target_bin = (phi_generated >= phi_lo) & (phi_generated < phi_hi)
        if OBJECTIVE == "phi_psi_target":
            psi_lo, psi_hi = _rama_bin_range(PHI_PSI_TARGET_ANGLE[1])
            in_target_bin = in_target_bin & (psi_generated >= psi_lo) & (psi_generated < psi_hi)
        metrics[f"{PREFIX}/frac-outside-target-bin"] = 1 - in_target_bin.float().mean().item()
        metrics[f"{PREFIX}/mean-energy-in-target-bin"] = (
            e_generated[in_target_bin].mean().item() if in_target_bin.any() else float("nan")
        )
    metrics.update(energy_wasserstein(pred_energy=e_generated, true_energy=eval_ctx.true_data.E_target, prefix=PREFIX))
    metrics.update(torus_wasserstein(eval_ctx.true_data.samples, samples_physical, eval_ctx.topology, prefix=PREFIX))

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
