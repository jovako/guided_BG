"""Optuna (TPE) hyperparameter search: smiley guidance via the exact-density
guided Euler integrator (euler_density_integrator.py), searched cheaply with
``track_density=False`` (skips the per-step Jacobian -- see that module's
docstring) since no density is needed for this frac_in_face/energy-w2
comparison.

Narrow search around the live config in plot_guided_euler_ramachandran.py:
gamma_hi=2.1, gamma_lo_slope=12.0 (quadratic rise: lo_slope*t**2),
gamma_threshold=0.4, gamma_use_decay=True, gamma_decay_threshold=0.4,
gamma_decay_slope_mag=2.0, alpha=0.089, w_terminal=22.4. gamma_use_decay and
gamma_threshold are FIXED here (not searched) -- the point of a narrow search
is to refine the decay-schedule shape around this specific regime, not
re-explore whether decay helps at all or when the rise phase should end.

Same objective as the other smiley searches (minimize the Wasserstein
distance to the unguided rejection-sampling baseline subject to
frac_in_face >= FRAC_IN_FACE_TARGET), but using
``generate_proposal_guided_euler``:
    - n_inner=1 (per "keep inner steps to 1 if possible")
    - the inner loop is a *normalized* functional gradient step, and the
      step function is ``x_next = cxt + dt*v_control`` where
      ``cxt = x + gamma*u`` -- the control's shift is teleported directly
      into the trajectory, not merely used to compute a velocity. See
      make_guided_euler_step's docstring in euler_density_integrator.py.
    - n_steps=250 (the minimum step count found orientation-safe for the
      UNGUIDED case; the guided case turned out to fail orientation much
      more often -- guidance's position teleport is itself a source of
      curvature, hence the gamma schedule below rather than a constant)
    - gamma as a schedule: quadratic rise (0 -> hi, lo_slope*t**2) up to
      gamma_threshold (fixed at 0.4), then a plateau at hi, then a decline
      (negative slope) back toward 0 starting at gamma_decay_threshold.
Searched: gamma_hi, gamma_lo_slope, gamma_decay_threshold,
gamma_decay_slope_mag, alpha (inner step length), w_terminal.
gamma_threshold is fixed (not searched, see SEED_PARAMS/GAMMA_THRESHOLD_FIXED).

Once a good trial is found, re-run it with track_density=True (the default)
to get the exact log-density for that specific configuration.

Run with:
    uv run python tests/guidance/hparam_search/hparam_search_smiley_euler_density.py
"""

from __future__ import annotations

import csv
import math
import shutil
import time
from pathlib import Path

import hydra
import optuna
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from transferable_samplers.evaluation.metrics.wasserstein_distances import energy_wasserstein, torus_wasserstein
from transferable_samplers.guidance.costs import repel_within_radius_penalty, torus_distance, within_radius_penalty
from transferable_samplers.guidance.euler_density_integrator import generate_proposal_guided_euler
from transferable_samplers.guidance.observables import dihedrals, get_dihedral_atom_indices
from transferable_samplers.utils.chirality import ChiralitySignChecker
from transferable_samplers.utils.init_resume_utils import resolve_init
from transferable_samplers.utils.standardization import destandardize_coords

BUDGET_SECONDS = 2 * 3600
EVAL_BATCH = 64
SEED = 42
FRAC_IN_FACE_TARGET = 0.98
MIN_FREE_DISK_GB = 2.0
SEQUENCE = "Ace-A-Nme"
OUT_DIR = "tests/guidance/out"
STUDY_NAME = "smiley_euler_density_n250_gammaquad_decay_v2"
STUDY_PATH = f"sqlite:///{OUT_DIR}/hparam_search_{STUDY_NAME}.db"
CSV_PATH = f"{OUT_DIR}/hparam_search_{STUDY_NAME}_results.csv"

# Current live config from plot_guided_euler_ramachandran.py (quadratic rise:
# gamma = gamma_lo_slope * t**2 up to gamma_threshold).
SEED_PARAMS = {
    "gamma_hi": 2.1, "gamma_lo_slope": 12.0,
    "gamma_decay_threshold": 0.4, "gamma_decay_slope_mag": 2.0,
    "alpha": 8.9e-2, "w_terminal": 22.4,
}

# Fixed, not searched.
N_STEPS = 250
N_INNER = 1
EYE_MOUTH_WEIGHT = 5.0
REJECTION_SAMPLES_PATH = f"{OUT_DIR}/rejection_smiley_samples.pt"
GAMMA_USE_DECAY = True  # fixed on for this search, see module docstring
GAMMA_THRESHOLD_FIXED = 0.4  # rise threshold, fixed not searched (per user instruction)

# Narrow ranges around SEED_PARAMS.
GAMMA_DECAY_THRESHOLD_BOUNDS = (0.4, 0.7)
BOUNDS = {
    "gamma_hi": (1.2, 3.2),
    "gamma_lo_slope": (8.0, 25.0),
    "gamma_decay_slope_mag": (1.0, 5.0),
    "alpha": (0.03, 0.15),
    "w_terminal": (10.0, 45.0),
}

# Same box target + derived smiley geometry as plot_guided_euler_ramachandran.py
# (kept in sync by hand -- must match that file's active values).
PHI_TARGET = (-2.0, -1.0)
PSI_TARGET = (-0.5, 0.5)
_BOX_CENTER = ((PHI_TARGET[0] + PHI_TARGET[1]) / 2, (PSI_TARGET[0] + PSI_TARGET[1]) / 2)
_BOX_HALF_EXTENT = min(PHI_TARGET[1] - PHI_TARGET[0], PSI_TARGET[1] - PSI_TARGET[0]) / 2
_SMILEY_SCALE = 0.9 * _BOX_HALF_EXTENT / 2.2
_INNER_SCALE = 0.75

FACE_CENTER = _BOX_CENTER
FACE_RADIUS = 2.2 * _SMILEY_SCALE
EYE_CENTERS = [
    (_BOX_CENTER[0] + dx * _INNER_SCALE * _SMILEY_SCALE, _BOX_CENTER[1] + dy * _INNER_SCALE * _SMILEY_SCALE)
    for dx, dy in [(-1.0, 1.0), (1.0, 1.0)]
]
EYE_RADIUS = 0.4 * _SMILEY_SCALE
MOUTH_PHIS = [-1.5, -1.125, -0.75, -0.375, 0.0, 0.375, 0.75, 1.125, 1.5]
MOUTH_CENTERS = [
    (
        _BOX_CENTER[0] + p * _INNER_SCALE * _SMILEY_SCALE,
        _BOX_CENTER[1] + (-1.5 + 0.35 * p**2) * _INNER_SCALE * _SMILEY_SCALE,
    )
    for p in MOUTH_PHIS
]
MOUTH_RADIUS = 0.35 * _SMILEY_SCALE

CSV_FIELDNAMES = [
    "optuna_trial_number",
    "gamma_hi",
    "gamma_lo_slope",
    "gamma_threshold",
    "gamma_decay_threshold",
    "gamma_decay_slope_mag",
    "alpha",
    "w_terminal",
    "frac_in_face",
    "frac_in_eye_or_mouth",
    "mean_energy",
    "vs_rejection_energy_w2",
    "vs_rejection_torus_w2",
    "elapsed_s",
    "failed",
    "error",
    "objective_value",
]


def load_model_and_data():
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(
            config_name="eval",
            overrides=["experiment=single_system/eval/ecnf++_Ace-A-Nme_snis"],
        )
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    model = hydra.utils.instantiate(cfg.model)
    state_dict = resolve_init(
        init_ckpt_path=cfg.get("ckpt_path"),
        init_hf_state_dict_path=cfg.get("hf_state_dict_path"),
        scratch_dir=cfg.paths.scratch_dir,
    )
    model.load_state_dict(state_dict)
    return model, datamodule, cfg.data.num_atoms


def make_terminal_cost(phi_idx, psi_idx, chirality_checker):
    """Single-sample convention (see make_guided_field): x1_flat is (d,), not batched."""

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

    return terminal_cost


def make_gamma(params: dict):
    """Quadratic rise (0 -> hi, ``lo_slope * t**2``) up to gamma_threshold,
    then a plateau at hi -- optionally followed by a decline (negative
    slope) back toward 0 starting at gamma_decay_threshold, if
    params["gamma_use_decay"]. Matches plot_guided_euler_ramachandran.py's
    current ``_gamma_fn``.
    """
    hi, lo_slope, rise_threshold = params["gamma_hi"], params["gamma_lo_slope"], params["gamma_threshold"]
    use_decay = params.get("gamma_use_decay", False)
    decay_threshold = params.get("gamma_decay_threshold")
    decay_slope = -params["gamma_decay_slope_mag"] if use_decay else None

    def gamma_fn(t: torch.Tensor) -> float:
        if t <= rise_threshold:
            return lo_slope * t**2
        if not use_decay or t <= decay_threshold:
            return hi
        return max(0.0, hi + decay_slope * (t - decay_threshold))

    return gamma_fn


def objective_value(frac_in_face: float, energy_w2: float, failed: bool) -> float:
    """Penalized, log1p-compressed scalar objective (minimize)."""
    if failed or frac_in_face != frac_in_face:  # NaN check
        raw = 1e7
    elif frac_in_face < FRAC_IN_FACE_TARGET:
        raw = 1e5 + (FRAC_IN_FACE_TARGET - frac_in_face) * 1e6
    else:
        raw = energy_w2
    return math.log1p(raw)


def run_trial(model, eval_ctx, phi_idx, psi_idx, chirality_checker, num_atoms, device, rejection_data, params: dict) -> dict:
    terminal_cost = make_terminal_cost(phi_idx, psi_idx, chirality_checker)

    torch.manual_seed(SEED)

    t0 = time.time()
    result = dict(params)
    try:
        x, _, _ = generate_proposal_guided_euler(
            model, EVAL_BATCH, num_atoms, lambda x1, t: params["w_terminal"] * terminal_cost(x1),
            gamma=make_gamma(params), alpha=params["alpha"], n_inner=N_INNER, n_steps=N_STEPS,
            use_score_deviation=False, beta=0.0, device=device, track_density=False,
        )
        x = x.detach()

        x_phys = destandardize_coords(x.cpu(), eval_ctx.normalization_std)
        flip_mask = chirality_checker.flip_mask(x_phys)
        x_phys = x_phys.clone()
        x_phys[flip_mask] *= -1

        phi = dihedrals(x_phys, phi_idx).squeeze(-1)
        psi = dihedrals(x_phys, psi_idx).squeeze(-1)
        dist_to_face = torus_distance(phi, psi, FACE_CENTER[0], FACE_CENTER[1])
        frac_in_face = (dist_to_face <= FACE_RADIUS).float().mean().item()

        in_eye_or_mouth = torch.zeros(EVAL_BATCH, dtype=torch.bool)
        for cx, cy in EYE_CENTERS:
            d = torus_distance(phi, psi, cx, cy)
            in_eye_or_mouth = in_eye_or_mouth | (d <= EYE_RADIUS)
        for cx, cy in MOUTH_CENTERS:
            d = torus_distance(phi, psi, cx, cy)
            in_eye_or_mouth = in_eye_or_mouth | (d <= MOUTH_RADIUS)

        with torch.no_grad():
            e_generated = eval_ctx.target_energy.energy(x.to(device))

        vs_rejection_energy_w2 = energy_wasserstein(
            pred_energy=e_generated.cpu(), true_energy=rejection_data["energy"], prefix="x"
        )["x/energy-w2"]
        vs_rejection_torus_w2 = torus_wasserstein(
            rejection_data["samples_physical"], x_phys, eval_ctx.topology, prefix="x"
        )["x/torus-w2"]

        result.update(
            frac_in_face=frac_in_face,
            frac_in_eye_or_mouth=in_eye_or_mouth.float().mean().item(),
            mean_energy=e_generated.mean().item(),
            vs_rejection_energy_w2=vs_rejection_energy_w2,
            vs_rejection_torus_w2=vs_rejection_torus_w2,
            elapsed_s=time.time() - t0,
            failed=False,
            error="",
        )
    except Exception as exc:  # noqa: BLE001 -- must never crash the search
        result.update(
            frac_in_face=float("nan"),
            frac_in_eye_or_mouth=float("nan"),
            mean_energy=float("nan"),
            vs_rejection_energy_w2=float("inf"),
            vs_rejection_torus_w2=float("inf"),
            elapsed_s=time.time() - t0,
            failed=True,
            error=repr(exc)[:200],
        )
    return result


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, datamodule, num_atoms = load_model_and_data()
    model = model.to(device).eval()

    eval_ctx = datamodule.prepare_eval(sequence=SEQUENCE, stage="test")
    phi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="phi")
    psi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="psi")
    chirality_checker = ChiralitySignChecker(eval_ctx.topology, eval_ctx.true_data.samples[:1])

    rejection_data = torch.load(REJECTION_SAMPLES_PATH, weights_only=False)
    print(f"loaded rejection baseline: {rejection_data['energy'].shape[0]} accepted samples")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STUDY_PATH,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    if len(study.trials) == 0:
        study.enqueue_trial(SEED_PARAMS)
        print(f"Starting cold, seeded with: {SEED_PARAMS}")
    else:
        print(f"Study has {len(study.trials)} trials already recorded.")

    csv_exists = Path(CSV_PATH).exists()
    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if not csv_exists:
        writer.writeheader()
        csv_file.flush()

    start_time = time.time()

    def optuna_objective(trial: optuna.Trial) -> float:
        free_gb = shutil.disk_usage("/").free / 1e9
        if free_gb < MIN_FREE_DISK_GB:
            print(f"Free disk on / dropped to {free_gb:.2f} GB, stopping search.")
            study.stop()
            raise optuna.exceptions.TrialPruned()

        params = {
            "gamma_hi": trial.suggest_float("gamma_hi", *BOUNDS["gamma_hi"], log=True),
            "gamma_lo_slope": trial.suggest_float("gamma_lo_slope", *BOUNDS["gamma_lo_slope"], log=True),
            "gamma_threshold": GAMMA_THRESHOLD_FIXED,
            "gamma_decay_threshold": trial.suggest_float("gamma_decay_threshold", *GAMMA_DECAY_THRESHOLD_BOUNDS),
            "gamma_decay_slope_mag": trial.suggest_float("gamma_decay_slope_mag", *BOUNDS["gamma_decay_slope_mag"], log=True),
            "alpha": trial.suggest_float("alpha", *BOUNDS["alpha"], log=True),
            "w_terminal": trial.suggest_float("w_terminal", *BOUNDS["w_terminal"], log=True),
        }
        params["gamma_use_decay"] = GAMMA_USE_DECAY

        result = run_trial(model, eval_ctx, phi_idx, psi_idx, chirality_checker, num_atoms, device, rejection_data, params)
        value = objective_value(result["frac_in_face"], result["vs_rejection_energy_w2"], result["failed"])

        row = {k: result.get(k, float("nan")) for k in CSV_FIELDNAMES if k != "optuna_trial_number" and k != "objective_value"}
        row["optuna_trial_number"] = trial.number
        row["objective_value"] = value
        writer.writerow(row)
        csv_file.flush()

        status = "FAILED" if result["failed"] else f"frac_in_face={result['frac_in_face']:.3f} energy_w2={result['vs_rejection_energy_w2']:.3f}"
        elapsed_min = (time.time() - start_time) / 60
        best_marker = ""
        if study.trials and any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            try:
                if study.best_value is not None and value <= study.best_value:
                    best_marker = " *new best*"
            except ValueError:
                pass
        gamma_desc = (
            f"gamma_hi={params['gamma_hi']:.3g} lo_slope={params['gamma_lo_slope']:.3g} t*={params['gamma_threshold']:.3g} "
            f"decay_t*={params['gamma_decay_threshold']:.3g} decay_slope=-{params['gamma_decay_slope_mag']:.3g}"
        )
        print(
            f"[{elapsed_min:6.1f} min] optuna trial {trial.number}: {gamma_desc} "
            f"alpha={params['alpha']:.4g} w_term={params['w_terminal']:.3g} -> {status}{best_marker}",
            flush=True,
        )
        if device == "cuda" and trial.number % 20 == 0:
            torch.cuda.empty_cache()

        return value

    study.optimize(optuna_objective, timeout=BUDGET_SECONDS)

    csv_file.close()

    best = study.best_trial
    print(f"\nSearch done: {len(study.trials)} total trials in study.")
    print(f"Best objective (log1p-compressed): {best.value:.4f}")
    print(f"Best params: {best.params}")


if __name__ == "__main__":
    main()
