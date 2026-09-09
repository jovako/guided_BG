"""Optuna (TPE) hyperparameter search: smiley guidance via the exact-density
guided Euler integrator (euler_density_integrator.py), searched cheaply with
``track_density=False`` (skips the per-step jacrev Jacobian -- ~60x cheaper,
see that module's docstring) and no density needed for this frac_in_face/
energy-w2 comparison.

Same objective as the other smiley searches (minimize the Wasserstein
distance to the unguided rejection-sampling baseline subject to
frac_in_face >= FRAC_IN_FACE_TARGET), but using
``generate_proposal_guided_euler``:
    - n_inner=1 (per "keep inner steps to 1 if possible")
    - the inner loop is a *normalized* functional gradient step, and the
      step function is ``x_next = cxt + dt*v_control`` where
      ``cxt = x + gamma*u`` -- matching ``_integrate_guided``'s exact
      recursion (the control's shift is teleported directly into the
      trajectory, not merely used to compute a velocity). Two earlier,
      weaker versions were tried and fixed this session: the reference
      guided_euler.py's original ``F = f0 + u`` never re-evaluated the
      network at all; a later fix re-evaluated the network at ``cxt`` but
      still routed through a plain ``x + dt*F(t,x)`` step, silently
      discarding the teleport. See make_guided_euler_step's docstring in
      euler_density_integrator.py for the full account.
    - n_steps=250 (the minimum step count found orientation-safe this
      session: n=200/225 had a nontrivial fraction of samples go
      non-orientation-preserving near t=1 (28%/9% at batch=64), n=250/300
      had zero failures)
Searched: gamma, alpha (inner step length), w_terminal.

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
STUDY_NAME = "smiley_euler_density_n250"
STUDY_PATH = f"sqlite:///{OUT_DIR}/hparam_search_smiley_euler_density_n250.db"
CSV_PATH = f"{OUT_DIR}/hparam_search_smiley_euler_density_n250_results.csv"

# Carried over from the n=200 search (smiley_euler_density_v2, best trial:
# gamma=0.669, alpha=0.158, w_term=13.6 -> frac_in_face=1.0, energy_w2=7.8)
# as a starting point -- not re-validated at n=250.
SEED_PARAMS = {"gamma": 0.669, "alpha": 0.158, "w_terminal": 13.6}

# Fixed, not searched.
N_STEPS = 250
N_INNER = 1
EYE_MOUTH_WEIGHT = 5.0
REJECTION_SAMPLES_PATH = f"{OUT_DIR}/rejection_smiley_samples.pt"

BOUNDS = {
    "gamma": (0.3, 6.0),
    "alpha": (0.001, 0.5),  # _integrate_guided's gd branch needed lr~0.001-0.05 at similar step counts
    "w_terminal": (5.0, 100.0),
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
    "gamma",
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
            model, EVAL_BATCH, num_atoms, lambda x1: params["w_terminal"] * terminal_cost(x1),
            gamma=params["gamma"], alpha=params["alpha"], n_inner=N_INNER, n_steps=N_STEPS,
            use_score_deviation=False, beta=0.0, device=device, track_density=False,
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
            "gamma": trial.suggest_float("gamma", *BOUNDS["gamma"], log=True),
            "alpha": trial.suggest_float("alpha", *BOUNDS["alpha"], log=True),
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
            f"[{elapsed_min:6.1f} min] optuna trial {trial.number}: gamma={params['gamma']:.4g} "
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
