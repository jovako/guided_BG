"""Optuna (TPE) hyperparameter search: 150 Euler steps, 1 inner step, constant gamma.

Same objective as hparam_search_smiley_optuna.py (minimize the Wasserstein
distance to the unguided rejection-sampling baseline subject to
frac_in_face >= FRAC_IN_FACE_TARGET), but a smaller, cheaper regime than the
200/600-step searches:
    - euler_steps=150
    - inner_steps=1
    - gamma is constant in t (no two-piece schedule)
    - guidance_w_vf=0, guidance_w_control=0 (both terms disabled)
    - guidance_optimizer="gd" (plain gradient descent, w/ L2-normalized
      gradient: u_t -= lr * grad / (grad.pow(2).sum() + 1e-8), not Adam)
Searched: lr, gamma_value, w_terminal.

(The 100-step version of this search -- run before the L2 normalization was
restored to the gd branch -- found trials qualifying on frac_in_face>=0.98
only at astronomically bad energy_w2: the phi/psi landed in the smiley
region but the 3D geometry was otherwise badly distorted, plausibly because
raw/unnormalized gradient steps aren't capped and can blow up in whichever
coordinate happens to have a large gradient. lr bounds here are widened
accordingly, since dividing by the (batch-global) squared-gradient sum
changes the effective step scale a lot relative to the raw-gradient version.)

No warm start: this regime doesn't match any prior search's fixed settings,
so this study starts cold. State persists in its own Optuna sqlite study,
so re-running this script resumes automatically.

Run with:
    uv run python tests/guidance/hparam_search/hparam_search_smiley_optuna_150steps_constant.py
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
STUDY_NAME = "smiley_150steps_constant_gd_l2norm"
STUDY_PATH = f"sqlite:///{OUT_DIR}/hparam_search_smiley_optuna_150steps_constant_gd_l2norm.db"
CSV_PATH = f"{OUT_DIR}/hparam_search_smiley_optuna_150steps_constant_gd_l2norm_results.csv"

# Fixed, not searched.
EULER_STEPS = 150
INNER_STEPS = 1
INIT_CONTROL = "zero"
EYE_MOUTH_WEIGHT = 15.0
W_VF = 0.0
W_CONTROL = 0.0
GUIDANCE_OPTIMIZER = "gd"
REJECTION_SAMPLES_PATH = f"{OUT_DIR}/rejection_smiley_samples.pt"

BOUNDS = {
    "gamma_value": (0.3, 6.0),
    "lr": (1e-3, 3.0),
    "w_terminal": (5.0, 200.0),
}

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
    "gamma_value",
    "lr",
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


def make_cost_fn(phi_idx, psi_idx, chirality_checker):
    def guidance_cost_fn(x1: torch.Tensor) -> torch.Tensor:
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

    return guidance_cost_fn


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
    model.use_guidance = True
    model.guidance_cost_fn = make_cost_fn(phi_idx, psi_idx, chirality_checker)
    model.guidance_num_steps = EULER_STEPS
    model.guidance_inner_steps = INNER_STEPS
    model.guidance_gamma = params["gamma_value"]
    model.guidance_lr = params["lr"]
    model.guidance_optimizer = GUIDANCE_OPTIMIZER
    model.guidance_w_terminal = params["w_terminal"]
    model.guidance_w_vf = W_VF
    model.guidance_w_control = W_CONTROL
    model.guidance_init_control = INIT_CONTROL

    torch.manual_seed(SEED)
    z = model.prior.sample(EVAL_BATCH, num_atoms, device=device)

    t0 = time.time()
    result = dict(params)
    try:
        with torch.no_grad():
            x = model._integrate_guided(model.net, z, encodings=None)

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
    print(f"Study has {len(study.trials)} trials already recorded." if study.trials else "Starting cold.")

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
            "gamma_value": trial.suggest_float("gamma_value", *BOUNDS["gamma_value"], log=True),
            "lr": trial.suggest_float("lr", *BOUNDS["lr"], log=True),
            "w_terminal": trial.suggest_float("w_terminal", *BOUNDS["w_terminal"], log=True),
        }

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
            f"[{elapsed_min:6.1f} min] optuna trial {trial.number}: gamma={params['gamma_value']:.4g} "
            f"lr={params['lr']:.4g} w_term={params['w_terminal']:.3g} -> {status}{best_marker}",
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
