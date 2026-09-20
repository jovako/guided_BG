"""Optuna (TPE) hyperparameter search for the smiley guidance objective.

Same fixed settings and objective as hparam_search_smiley_refined.py
(euler_steps=200, inner_steps=1, EYE_MOUTH_WEIGHT=15.0; minimize the
Wasserstein distance to the unguided rejection-sampling baseline subject to
frac_in_face >= FRAC_IN_FACE_TARGET), searched with Optuna's TPE sampler
(the landscape is sharply non-smooth, which a GP-style sampler assumes away).

The frac_in_face constraint is handled by penalizing a single-objective
value rather than Optuna's constraint machinery:
    objective = log1p(energy_w2)                if qualifying
    objective = log1p(1e5 + violation * 1e6)    if frac_in_face < target
    objective = log1p(1e7)                       if the trial crashed

Warm-starts from hparam_search_smiley_refined_results.csv (imported as
completed Optuna trials via optuna.trial.create_trial), excluding rows from
a since-corrected gamma_threshold range.

State persists in an Optuna sqlite study (STUDY_PATH), so re-running this
script resumes automatically.

Run with:
    uv run python tests/guidance/hparam_search/hparam_search_smiley_optuna.py
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
from transferable_samplers.guidance.costs import repel_within_radius_penalty, within_radius_penalty
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
STUDY_NAME = "smiley_refined"
STUDY_PATH = f"sqlite:///{OUT_DIR}/hparam_search_smiley_optuna.db"
CSV_PATH = f"{OUT_DIR}/hparam_search_smiley_optuna_results.csv"
WARM_START_CSV = f"{OUT_DIR}/hparam_search_smiley_refined_results.csv"

# Fixed, not searched (same as hparam_search_smiley_refined.py).
EULER_STEPS = 200
INNER_STEPS = 1
EYE_MOUTH_WEIGHT = 15.0
GAMMA_THRESHOLD_MAX = 0.32  # ceiling
GAMMA_THRESHOLD_FLOOR = 0.05
REJECTION_SAMPLES_PATH = f"{OUT_DIR}/rejection_smiley_samples.pt"

# Search space bounds, with margin beyond what hparam_search_smiley_refined.py explored.
BOUNDS = {
    "gamma_value": (0.3, 6.0),
    "gamma_hi": (0.8, 6.0),
    "gamma_lo_slope": (0.15, 3.0),
    "lr": (1e-3, 0.05),
    "w_terminal": (5.0, 200.0),
    "w_control": (1e-5, 0.02),
}
W_VF_CHOICES = [0.0, 0.01, 0.05]

# Same box target + derived smiley geometry as the other guidance scripts.
PHI_TARGET = (-2.0, -1.0)
PSI_TARGET = (-0.5, 0.5)
_BOX_CENTER = ((PHI_TARGET[0] + PHI_TARGET[1]) / 2, (PSI_TARGET[0] + PSI_TARGET[1]) / 2)
_BOX_HALF_EXTENT = min(PHI_TARGET[1] - PHI_TARGET[0], PSI_TARGET[1] - PSI_TARGET[0]) / 2
_SMILEY_SCALE = 0.9 * _BOX_HALF_EXTENT / 2.2
_INNER_SCALE = 0.5

FACE_CENTER = _BOX_CENTER
FACE_RADIUS = 2.2 * _SMILEY_SCALE
EYE_CENTERS = [
    (_BOX_CENTER[0] + dx * _INNER_SCALE * _SMILEY_SCALE, _BOX_CENTER[1] + dy * _INNER_SCALE * _SMILEY_SCALE)
    for dx, dy in [(-1.0, 1.0), (1.0, 1.0)]
]
EYE_RADIUS = 0.3 * _SMILEY_SCALE
MOUTH_PHIS = [-1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2]
MOUTH_CENTERS = [
    (
        _BOX_CENTER[0] + p * _INNER_SCALE * _SMILEY_SCALE,
        _BOX_CENTER[1] + (-1.3 + 0.2 * p**2) * _INNER_SCALE * _SMILEY_SCALE,
    )
    for p in MOUTH_PHIS
]
MOUTH_RADIUS = 0.2 * _SMILEY_SCALE

CSV_FIELDNAMES = [
    "optuna_trial_number",
    "gamma_mode",
    "gamma_value",
    "gamma_hi",
    "gamma_lo_slope",
    "gamma_threshold",
    "lr",
    "w_terminal",
    "w_vf",
    "w_control",
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


def make_gamma(params: dict):
    if params["gamma_mode"] == "constant":
        return params["gamma_value"]

    hi, lo_slope, threshold = params["gamma_hi"], params["gamma_lo_slope"], params["gamma_threshold"]

    def gamma_fn(t: torch.Tensor) -> float:
        return hi if t > threshold else lo_slope * t

    return gamma_fn


def make_cost_fn(phi_idx, psi_idx, chirality_checker):
    def terminal_cost(x1_flat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x1 = x1_flat.view(1, -1, 3)
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

        return cost.squeeze()

    return terminal_cost


def float_distribution(low: float, high: float, log: bool = False):
    """Optuna 2.x/3.x compatibility: 2.x has no unified FloatDistribution(log=...)."""
    if hasattr(optuna.distributions, "FloatDistribution"):
        return optuna.distributions.FloatDistribution(low, high, log=log)
    if log:
        return optuna.distributions.LogUniformDistribution(low, high)
    return optuna.distributions.UniformDistribution(low, high)


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
    terminal_cost = make_cost_fn(phi_idx, psi_idx, chirality_checker)
    gamma = make_gamma(params)

    torch.manual_seed(SEED)

    t0 = time.time()
    result = dict(params)
    try:
        x, _, _ = generate_proposal_guided_euler(
            model, EVAL_BATCH, num_atoms,
            lambda x1, t: params["w_terminal"] * terminal_cost(x1, t),
            gamma=gamma, alpha=params["lr"], lam=params["w_control"], beta=params["w_vf"],
            n_inner=INNER_STEPS, n_steps=EULER_STEPS,
            device=device, track_density=False,
        )
        x = x.detach()

        x_phys = destandardize_coords(x.cpu(), eval_ctx.normalization_std)
        flip_mask = chirality_checker.flip_mask(x_phys)
        x_phys = x_phys.clone()
        x_phys[flip_mask] *= -1

        phi = dihedrals(x_phys, phi_idx).squeeze(-1)
        psi = dihedrals(x_phys, psi_idx).squeeze(-1)
        dist_to_face = torch.sqrt((phi - FACE_CENTER[0]) ** 2 + (psi - FACE_CENTER[1]) ** 2)
        frac_in_face = (dist_to_face <= FACE_RADIUS).float().mean().item()

        in_eye_or_mouth = torch.zeros(EVAL_BATCH, dtype=torch.bool)
        for cx, cy in EYE_CENTERS:
            d = torch.sqrt((phi - cx) ** 2 + (psi - cy) ** 2)
            in_eye_or_mouth = in_eye_or_mouth | (d <= EYE_RADIUS)
        for cx, cy in MOUTH_CENTERS:
            d = torch.sqrt((phi - cx) ** 2 + (psi - cy) ** 2)
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


def load_warm_start_trials() -> list[tuple[dict, dict, float]]:
    """Load hparam_search_smiley_refined_results.csv as (params, distributions, value) tuples.

    Excludes schedule rows with gamma_threshold > GAMMA_THRESHOLD_MAX + eps
    (a since-abandoned search region).
    """
    if not Path(WARM_START_CSV).exists():
        print(f"No warm-start CSV found at {WARM_START_CSV}, starting cold.")
        return []

    entries = []
    skipped = 0
    with open(WARM_START_CSV) as f:
        for row in csv.DictReader(f):
            try:
                mode = row["gamma_mode"]
                failed = row["failed"] == "True"
                frac_in_face = float(row["frac_in_face"])
                energy_w2 = float(row["vs_rejection_energy_w2"])

                if mode == "schedule":
                    threshold = float(row["gamma_threshold"])
                    if threshold > GAMMA_THRESHOLD_MAX + 1e-6:
                        skipped += 1
                        continue
                    threshold = min(threshold, GAMMA_THRESHOLD_MAX)  # clamp tiny float overshoot
                    params = {
                        "gamma_hi": float(row["gamma_hi"]),
                        "gamma_lo_slope": float(row["gamma_lo_slope"]),
                        "gamma_threshold": threshold,
                    }
                    distributions = {
                        "gamma_hi": float_distribution(*BOUNDS["gamma_hi"], log=True),
                        "gamma_lo_slope": float_distribution(*BOUNDS["gamma_lo_slope"], log=True),
                        "gamma_threshold": float_distribution(GAMMA_THRESHOLD_FLOOR, GAMMA_THRESHOLD_MAX),
                    }
                else:
                    params = {"gamma_value": float(row["gamma_value"])}
                    distributions = {"gamma_value": float_distribution(*BOUNDS["gamma_value"], log=True)}

                params["gamma_mode"] = mode
                distributions["gamma_mode"] = optuna.distributions.CategoricalDistribution(["schedule", "constant"])

                for key in ["lr", "w_terminal", "w_control"]:
                    params[key] = float(row[key])
                    distributions[key] = float_distribution(*BOUNDS[key], log=True)
                params["w_vf"] = float(row["w_vf"])
                distributions["w_vf"] = optuna.distributions.CategoricalDistribution(W_VF_CHOICES)

                value = objective_value(frac_in_face, energy_w2, failed)
                entries.append((params, distributions, value))
            except (KeyError, ValueError) as exc:
                print(f"Skipping unparseable warm-start row: {exc}")
                skipped += 1

    print(f"Loaded {len(entries)} warm-start trials from {WARM_START_CSV} ({skipped} skipped).")
    return entries


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
        for params, distributions, value in load_warm_start_trials():
            study.add_trial(optuna.trial.create_trial(params=params, distributions=distributions, value=value))
        print(f"Warm-started study with {len(study.trials)} historical trials.")
    else:
        print(f"Resuming existing study with {len(study.trials)} trials already recorded.")

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

        gamma_mode = trial.suggest_categorical("gamma_mode", ["schedule", "constant"])
        if gamma_mode == "constant":
            params = {"gamma_value": trial.suggest_float("gamma_value", *BOUNDS["gamma_value"], log=True)}
        else:
            params = {
                "gamma_hi": trial.suggest_float("gamma_hi", *BOUNDS["gamma_hi"], log=True),
                "gamma_lo_slope": trial.suggest_float("gamma_lo_slope", *BOUNDS["gamma_lo_slope"], log=True),
                "gamma_threshold": trial.suggest_float("gamma_threshold", GAMMA_THRESHOLD_FLOOR, GAMMA_THRESHOLD_MAX),
            }
        params["gamma_mode"] = gamma_mode
        params["lr"] = trial.suggest_float("lr", *BOUNDS["lr"], log=True)
        params["w_terminal"] = trial.suggest_float("w_terminal", *BOUNDS["w_terminal"], log=True)
        params["w_vf"] = trial.suggest_categorical("w_vf", W_VF_CHOICES)
        params["w_control"] = trial.suggest_float("w_control", *BOUNDS["w_control"], log=True)

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
        print(
            f"[{elapsed_min:6.1f} min] optuna trial {trial.number}: mode={gamma_mode} lr={params['lr']:.4g} "
            f"w_term={params['w_terminal']:.3g} w_control={params['w_control']:.4g} -> {status}{best_marker}",
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
