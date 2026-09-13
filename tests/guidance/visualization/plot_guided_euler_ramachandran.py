"""Guidance sanity check: steer alanine dipeptide samples toward a chosen objective.

Same setup as ``plot_unguided_euler_ramachandran.py`` (fixed-step Euler
integration, ECNF++ on Ace-A-Nme, no SNIS/importance-weighting), but with
``use_guidance=True``. Set OBJECTIVE below to switch the terminal cost:
    - "pos_phi": one-sided penalty, zero cost once phi > 0.
    - "phi_target": quadratic penalty, zero cost only exactly at
      phi=PHI_TARGET_ANGLE (~1, nudged to the center of the nearest
      plot_ramachandran histogram bin so the plotted target line lands
      cleanly in one pixel column; 1-D: ignores psi entirely).
    - "phi_psi_target": periodic quadratic penalty, zero cost only exactly at
      (phi, psi) = (PHI_PSI_TARGET[0], PHI_PSI_TARGET[1]) -- a single point in
      the full 2-D Ramachandran plane, using torus_distance so the bowl wraps
      correctly if the target sits near the +-pi seam.
    - "smiley": pulls samples to stay within FACE_RADIUS of FACE_CENTER (the
      face outline) and strongly repels them from small circles marking the
      eyes and mouth -- so density fills the disk everywhere except
      smiley-shaped holes.
    - "smiley_reference": time-dependent. Until t=SMILEY_REFERENCE_T_SWITCH,
      same phi_psi_target-style periodic quadratic bowl as "phi_psi_target",
      but centered on a MOVING target: the phi/psi of the endpoint prediction
      the network itself makes, at the SAME t, for a fixed reference point --
      the noised (linear-coupling) version at that t of a real accepted
      (rejection-sampled) sample from the smiley center (see
      linear_coupling_trajectory_smiley_center.py). After the switch, reverts
      to the ordinary "smiley" cost. Only supported with
      USE_EULER_DENSITY_MODULE=True -- ``_integrate_guided`` has no way to
      pass ``t`` into a guidance cost function.

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

# Diagnostic toggle: model._integrate_guided (the reference) vs
# generate_proposal_guided_euler (euler_density_integrator.py, what the
# hparam searches actually use) -- confirmed functionally equivalent
# (same seed/params -> frac_in_face=0.984 both ways, samples match to
# ~1e-3 mean abs diff), added here to isolate whether a "much worse" result
# comes from the integration algorithm itself or from something else in
# this script's own surrounding code (metrics, chirality handling, etc.).
USE_EULER_DENSITY_MODULE = True

# Best-known hyperparameters found so far per OBJECTIVE (not auto-applied --
# set the constants below by hand to match whichever you're running):
#   - "pos_phi": EULER_STEPS=100, constant GUIDANCE_GAMMA=2.0, GUIDANCE_LR=0.01,
#     GUIDANCE_W_TERMINAL=20.0, GUIDANCE_INNER_STEPS=1, GUIDANCE_OPTIMIZER="gd"
#     (from tests/guidance/out/gif_full_space_pos_phi.log).
#   - "phi_target": EULER_STEPS=200, constant GUIDANCE_GAMMA=2.0, GUIDANCE_LR=0.01,
#     GUIDANCE_W_TERMINAL=20.0, GUIDANCE_INNER_STEPS=1, GUIDANCE_OPTIMIZER="gd"
#     (from tests/guidance/out/gif_full_space_phi_target.log).
#   - "smiley": EULER_STEPS=250, GUIDANCE_INNER_STEPS=1, gamma schedule
#     gamma_hi=1.91, gamma_lo_slope=4.7 (linear rise: lo_slope*t up to
#     gamma_threshold), gamma_threshold=0.408, gamma_use_decay=True,
#     gamma_decay_threshold=0.805, gamma_decay_slope_mag=7.18, GUIDANCE_LR
#     (alpha)=0.08863, GUIDANCE_W_TERMINAL=22.4, GUIDANCE_W_VF=0,
#     GUIDANCE_W_CONTROL=0 -- the "66%-valid" config the current 20h
#     sample_smiley_guided_snis_pool.py run is using (see that script /
#     hparam_search_smiley_euler_density.py's SEED_PARAMS).
OBJECTIVE = "smiley"  # "pos_phi", "phi_target", "phi_psi_target", "smiley", or "smiley_reference"

NUM_SAMPLES = 64
BATCH_SIZE = 64
SEED = 42
EULER_STEPS = 250
GUIDANCE_INNER_STEPS = 1

# Gamma schedule: rise (0 -> GAMMA_HI, slope GAMMA_LO_SLOPE) up to
# GAMMA_THRESHOLD, then a plateau at GAMMA_HI -- optionally followed by a
# decline (negative slope) back toward 0 starting at GAMMA_DECAY_THRESHOLD,
# if GAMMA_USE_DECAY. Same schedule shape as
# hparam_search_smiley_euler_density.py's make_gamma.
GAMMA_HI = 2.0
GAMMA_LO_SLOPE = 12.
GAMMA_THRESHOLD = 0.
GAMMA_USE_DECAY = False
GAMMA_DECAY_THRESHOLD = 0.7  # >= 0.7, per this session's convention
GAMMA_DECAY_SLOPE_MAG = 10.0  # magnitude; applied as negative below


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
    "phi_psi_target": "rejection_phi_psi_target",
    "smiley": "rejection_smiley",
    "smiley_reference": "rejection_smiley",  # still targets the smiley shape in the end
}[OBJECTIVE]
REJECTION_SAMPLES_PATH = f"{OUT_DIR}/{_REJECTION_PREFIX}_samples.pt"

# For OBJECTIVE == "smiley_reference": the fixed (x0, x1) reference pair built by
# linear_coupling_trajectory_smiley_center.py (x1 = the rejection-sampled point
# closest to the smiley center, x0 = a Gaussian prior draw), and the t at which
# guidance switches from tracking that reference's own endpoint prediction to
# the plain smiley shape cost.
REFERENCE_TRAJECTORY_PATH = f"{OUT_DIR}/linear_coupling_trajectory_smiley_center.pt"
SMILEY_REFERENCE_T_SWITCH = 0.5

# Single-point target for OBJECTIVE == "phi_psi_target": (phi, psi) in radians,
# before alignment to the Ramachandran histogram grid (see PHI_PSI_TARGET_ANGLE
# below).
PHI_PSI_TARGET = (-2.0, 1.5)

# plot_ramachandran's Ramachandran histogram bins (100 bins over [-pi, pi],
# bin width 2*pi/100) -- used below to nudge each 1-D/2-D target onto the
# center of its nearest bin, so the plotted target line/point and the
# "in target bin" metrics land cleanly in one pixel (column, or cell) instead
# of straddling an edge between two.
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
        "GUIDANCE_INIT_CONTROL": GUIDANCE_INIT_CONTROL,
    }


def describe_hyperparams() -> dict:
    """Full run record for the saved CSV: guidance hyperparameters plus bookkeeping
    and (for "smiley") the shape-specification constants."""
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
        # Same chirality correction as pos_phi_cost_fn. Zero only at
        # phi=PHI_TARGET_ANGLE, quadratic penalty growing on both sides --
        # box_quadratic_penalty with low=high=PHI_TARGET_ANGLE collapses to
        # exactly (phi-PHI_TARGET_ANGLE)**2, since only one of its two relu
        # terms is ever nonzero for a given phi.
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        phi = dihedrals(x1 * sign, phi_idx)  # (batch, num_phi)
        return box_quadratic_penalty(phi, low=PHI_TARGET_ANGLE, high=PHI_TARGET_ANGLE).sum(dim=-1)

    def phi_psi_target_cost_fn(x1: torch.Tensor) -> torch.Tensor:
        # Same chirality correction as pos_phi_cost_fn, applied to both phi and
        # psi. Periodic quadratic bowl (torus_distance) centered on
        # PHI_PSI_TARGET_ANGLE -- zero cost only exactly at that single point.
        with torch.no_grad():
            flip_mask = chirality_checker.flip_mask(x1)
        sign = torch.where(flip_mask, -1.0, 1.0).to(x1)[:, None, None]
        x1_fixed = x1 * sign
        phi = dihedrals(x1_fixed, phi_idx).squeeze(-1)
        psi = dihedrals(x1_fixed, psi_idx).squeeze(-1)
        dist = torus_distance(phi, psi, PHI_PSI_TARGET_ANGLE[0], PHI_PSI_TARGET_ANGLE[1])
        return quadratic_target_penalty(dist)

    def smiley_cost_fn(x1: torch.Tensor) -> torch.Tensor:
        # Same chirality correction as pos_phi_cost_fn, applied to both phi and psi.
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
        # After the switch, this is exactly smiley_cost_fn.
        if t > SMILEY_REFERENCE_T_SWITCH:
            return smiley_cost_fn(x1)

        # Before the switch: same periodic quadratic bowl as phi_psi_target_cost_fn,
        # but centered on a MOVING target -- the reference (noised smiley-center)
        # point's own endpoint prediction at this SAME t, not a fixed point. The
        # reference computation carries no gradient (it doesn't depend on the
        # current sample's control u at all -- torch.no_grad() just avoids
        # wasting a graph on it).
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

    if USE_EULER_DENSITY_MODULE:
        # generate_proposal_guided_euler wants a SINGLE-sample terminal_cost
        # ((d,), t in, scalar out) -- wrap the batched guidance_cost_fn above
        # rather than duplicating each OBJECTIVE's cost logic. Only
        # smiley_reference actually uses t; the others just ignore it.
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
    else:
        if OBJECTIVE == "smiley_reference":
            raise NotImplementedError(
                "smiley_reference needs t inside the guidance cost function, but "
                "_integrate_guided's guidance_cost_fn callback never receives t -- "
                "only supported with USE_EULER_DENSITY_MODULE=True."
            )
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
        # Fraction of samples that landed outside the target's own bin/cell in
        # the plot_ramachandran histogram (same 100x100 bins over [-pi, pi]
        # used for PHI_TARGET_ANGLE/PHI_PSI_TARGET_ANGLE above) -- i.e. not in
        # the pixel (column, for phi_target; cell, for phi_psi_target) the
        # target sits in -- plus the mean energy of just those in-bin samples.
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
