"""Coarse random hyperparameter search for the smiley guidance objective.

Searches ``FlowMatchingModule``'s guidance hyperparameters (euler steps,
inner steps, control init scheme, gamma schedule, lr, terminal/vf/control
weights). EYE_MOUTH_WEIGHT is fixed, not searched -- it's part of the shape
specification (how firmly the eyes/mouth holes are enforced), not a
guidance-algorithm hyperparameter. Looking for settings that:
    - hit >= FRAC_IN_FACE_TARGET (default 0.99) fraction of samples inside
      the face circle (the containment constraint), while
    - minimizing energy-w2 to the true (unguided) trajectory -- i.e. staying
      as close as possible to the unguided baseline while still satisfying
      the face constraint.

This is a random search (not a grid -- the joint space is too large), run
under a wall-clock time budget (``BUDGET_SECONDS``). A handful of trials at
the front specifically test inner_steps=1 with causal_zero init. Every
trial's result is appended to a CSV immediately (so progress survives an
early stop), and a small best-so-far summary is rewritten after every trial.
Non-finite OpenMM energies (unphysical geometry from aggressive guidance)
are caught per-trial and recorded as failed rather than crashing the search.

Only ever writes small CSV rows -- no per-trial samples or plots are saved,
to avoid using meaningful disk space during a long unattended run. Free disk
space on ``/`` is checked periodically as a safety net.

Run with:
    uv run python tests/guidance/hparam_search/hparam_search.py
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import time
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from transferable_samplers.evaluation.metrics.wasserstein_distances import energy_wasserstein, torus_wasserstein
from transferable_samplers.guidance.costs import repel_within_radius_penalty, within_radius_penalty
from transferable_samplers.guidance.observables import dihedrals, get_dihedral_atom_indices
from transferable_samplers.utils.chirality import ChiralitySignChecker
from transferable_samplers.utils.init_resume_utils import resolve_init
from transferable_samplers.utils.standardization import destandardize_coords

BUDGET_SECONDS = 8 * 3600
EVAL_BATCH = 64
SEED = 42
MAX_EULER_STEPS = 300
FRAC_IN_FACE_TARGET = 0.99
MIN_FREE_DISK_GB = 2.0
SEQUENCE = "Ace-A-Nme"
OUT_DIR = "tests/guidance/out"
CSV_PATH = f"{OUT_DIR}/hparam_search_results.csv"
BEST_PATH = f"{OUT_DIR}/hparam_search_best.json"

# Same box target as plot_guided_euler_ramachandran.py, and the same
# derived (scaled + recentered) smiley geometry.
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
# Fixed, not searched: this is part of the shape specification (how firmly
# the eyes/mouth holes are enforced), not a guidance-algorithm hyperparameter.
EYE_MOUTH_WEIGHT = 15.0

FIELDNAMES = [
    "trial_idx",
    "euler_steps",
    "inner_steps",
    "init_control",
    "gamma_hi",
    "gamma_lo_slope",
    "gamma_threshold",
    "lr",
    "w_terminal",
    "w_vf",
    "w_control",
    "eye_mouth_weight",
    "frac_in_face",
    "frac_in_eye_or_mouth",
    "mean_energy",
    "energy_w2",
    "torus_w2",
    "elapsed_s",
    "failed",
    "error",
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


def make_gamma_fn(hi: float, lo_slope: float, threshold: float):
    def gamma_fn(t: torch.Tensor) -> float:
        return hi if t > threshold else lo_slope * t

    return gamma_fn


def make_cost_fn(phi_idx, psi_idx, chirality_checker, eye_mouth_weight: float):
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
            cost = cost + eye_mouth_weight * repel_within_radius_penalty(dist, EYE_RADIUS)
        for cx, cy in MOUTH_CENTERS:
            dist = torch.sqrt((phi - cx) ** 2 + (psi - cy) ** 2)
            cost = cost + eye_mouth_weight * repel_within_radius_penalty(dist, MOUTH_RADIUS)

        return cost

    return guidance_cost_fn


def sample_params(rng: random.Random) -> dict:
    return {
        "euler_steps": rng.choice([100, 150, 200, 250, MAX_EULER_STEPS]),
        "inner_steps": rng.choice([1, 1, 2, 3, 4]),
        "init_control": rng.choice(["zero", "causal_zero"]),
        "gamma_hi": 10 ** rng.uniform(0.0, 1.0),
        "gamma_lo_slope": 10 ** rng.uniform(-0.3, 0.7),
        "gamma_threshold": rng.uniform(0.3, 0.8),
        "lr": 10 ** rng.uniform(-3.0, -1.3),
        "w_terminal": 10 ** rng.uniform(0.7, 2.3),
        "w_vf": rng.choice([0.0, 0.0, 0.0, 0.01, 0.1]),
        "w_control": 10 ** rng.uniform(-5.0, -2.0),
    }


def forced_trials() -> list[dict]:
    """A few trials targeting the inner_steps=1 + causal_zero hypothesis up front."""
    trials = []
    for gamma_hi in [3.0, 5.0, 8.0]:
        for lr in [5e-3, 1e-2, 2e-2]:
            trials.append(
                {
                    "euler_steps": 200,
                    "inner_steps": 1,
                    "init_control": "causal_zero",
                    "gamma_hi": gamma_hi,
                    "gamma_lo_slope": 2.0,
                    "gamma_threshold": 0.6,
                    "lr": lr,
                    "w_terminal": 50.0,
                    "w_vf": 0.0,
                    "w_control": 1e-3,
                }
            )
    return trials


def run_trial(model, eval_ctx, phi_idx, psi_idx, chirality_checker, num_atoms, device, params: dict) -> dict:
    model.use_guidance = True
    model.guidance_cost_fn = make_cost_fn(phi_idx, psi_idx, chirality_checker, EYE_MOUTH_WEIGHT)
    model.guidance_num_steps = params["euler_steps"]
    model.guidance_inner_steps = params["inner_steps"]
    model.guidance_gamma = make_gamma_fn(params["gamma_hi"], params["gamma_lo_slope"], params["gamma_threshold"])
    model.guidance_lr = params["lr"]
    model.guidance_w_terminal = params["w_terminal"]
    model.guidance_w_vf = params["w_vf"]
    model.guidance_w_control = params["w_control"]
    model.guidance_init_control = params["init_control"]

    torch.manual_seed(SEED)
    z = model.prior.sample(EVAL_BATCH, num_atoms, device=device)

    t0 = time.time()
    result = dict(params)
    result["eye_mouth_weight"] = EYE_MOUTH_WEIGHT
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

        energy_w2 = energy_wasserstein(pred_energy=e_generated.cpu(), true_energy=eval_ctx.true_data.E_target, prefix="x")[
            "x/energy-w2"
        ]
        torus_w2 = torus_wasserstein(eval_ctx.true_data.samples, x_phys, eval_ctx.topology, prefix="x")["x/torus-w2"]

        result.update(
            frac_in_face=frac_in_face,
            frac_in_eye_or_mouth=in_eye_or_mouth.float().mean().item(),
            mean_energy=e_generated.mean().item(),
            energy_w2=energy_w2,
            torus_w2=torus_w2,
            elapsed_s=time.time() - t0,
            failed=False,
            error="",
        )
    except Exception as exc:  # noqa: BLE001 -- must never crash an overnight search
        result.update(
            frac_in_face=float("nan"),
            frac_in_eye_or_mouth=float("nan"),
            mean_energy=float("nan"),
            energy_w2=float("inf"),
            torus_w2=float("inf"),
            elapsed_s=time.time() - t0,
            failed=True,
            error=repr(exc)[:200],
        )
    return result


def write_best(results: list[dict]) -> None:
    qualifying = [r for r in results if not r["failed"] and r["frac_in_face"] >= FRAC_IN_FACE_TARGET]
    qualifying.sort(key=lambda r: r["energy_w2"])
    with open(BEST_PATH, "w") as f:
        json.dump(
            {
                "num_trials": len(results),
                "num_qualifying": len(qualifying),
                "best": qualifying[0] if qualifying else None,
                "top_5": qualifying[:5],
            },
            f,
            indent=2,
        )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, datamodule, num_atoms = load_model_and_data()
    model = model.to(device).eval()

    eval_ctx = datamodule.prepare_eval(sequence=SEQUENCE, stage="test")
    phi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="phi")
    psi_idx = get_dihedral_atom_indices(eval_ctx.topology, kind="psi")
    chirality_checker = ChiralitySignChecker(eval_ctx.topology, eval_ctx.true_data.samples[:1])

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    csv_exists = Path(CSV_PATH).exists()
    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if not csv_exists:
        writer.writeheader()
        csv_file.flush()

    rng = random.Random(0)
    queue = forced_trials()
    results = []

    start_time = time.time()
    trial_idx = 0
    while time.time() - start_time < BUDGET_SECONDS:
        free_gb = shutil.disk_usage("/").free / 1e9
        if free_gb < MIN_FREE_DISK_GB:
            print(f"Free disk on / dropped to {free_gb:.2f} GB, stopping search early.")
            break

        params = queue.pop(0) if queue else sample_params(rng)
        result = run_trial(model, eval_ctx, phi_idx, psi_idx, chirality_checker, num_atoms, device, params)
        result["trial_idx"] = trial_idx
        results.append(result)

        writer.writerow({k: result[k] for k in FIELDNAMES})
        csv_file.flush()

        status = "FAILED" if result["failed"] else f"frac_in_face={result['frac_in_face']:.3f} energy_w2={result['energy_w2']:.2f}"
        elapsed_min = (time.time() - start_time) / 60
        print(
            f"[{elapsed_min:6.1f} min] trial {trial_idx}: steps={params['euler_steps']} inner={params['inner_steps']} "
            f"init={params['init_control']} lr={params['lr']:.4g} w_term={params['w_terminal']:.3g} -> {status}",
            flush=True,
        )

        if trial_idx % 20 == 0:
            write_best(results)
            if device == "cuda":
                torch.cuda.empty_cache()

        trial_idx += 1

    write_best(results)
    csv_file.close()
    print(f"Search done: {trial_idx} trials in {(time.time() - start_time) / 3600:.2f} hours. See {BEST_PATH}")


if __name__ == "__main__":
    main()
