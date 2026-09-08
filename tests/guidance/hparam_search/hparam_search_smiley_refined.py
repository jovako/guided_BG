"""Refined smiley guidance search around trial 35 of hparam_search.py.

Fixes euler_steps=200, inner_steps=1, init_control="zero" (causal_zero was
conclusively bad across the whole first search -- 0/236 qualifying trials,
regardless of steps/inner_steps -- so it's excluded here rather than
re-explored) and carefully varies the remaining guidance knobs around trial
35's known-good config:
    euler_steps=200 inner=1 init=zero gamma_hi=1.90 gamma_lo_slope=0.52
    gamma_threshold=0.32 lr=0.0144 w_terminal=39.2 w_control=0.0065 w_vf=0.0
    -> frac_in_face=1.0, energy-w2 (vs true trajectory)=19.67

Two gamma parametrizations are both tried per instruction ("also try constant
gamma_ts"): a plain constant, or the two-piece schedule
(hi if t > threshold else lo_slope * t) -- with gamma_threshold constrained
to <= 0.32 (explicit ceiling from instruction; an earlier version of this
script had it backwards as a floor, which is why some of the first ~75
trials in the results CSV sit right at the 0.32 boundary from below).

Objective (different from the original search!): minimize the Wasserstein
distance between the guided samples and the ALREADY-GENERATED unguided
rejection-sampling baseline for the smiley constraint
(rejection_baseline_smiley.py's saved samples+energies, 500 accepted, 10.1%
acceptance rate) -- the direct "is guidance good" comparison against samples
that already satisfy the same constraint, rather than each side's separate
distance to the unconstrained true trajectory -- subject to
frac_in_face >= 0.98.

EYE_MOUTH_WEIGHT is fixed at 15.0, matching hparam_search.py and trial 35
(plot_guided_euler_ramachandran.py currently uses 5.0 -- kept at 15.0 here
for direct comparability with the original search's results).

Run with:
    uv run python tests/guidance/hparam_search/hparam_search_smiley_refined.py
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

BUDGET_SECONDS = 2 * 3600
EVAL_BATCH = 64
SEED = 42
FRAC_IN_FACE_TARGET = 0.98
MIN_FREE_DISK_GB = 2.0
SEQUENCE = "Ace-A-Nme"
OUT_DIR = "tests/guidance/out"
CSV_PATH = f"{OUT_DIR}/hparam_search_smiley_refined_results.csv"
BEST_PATH = f"{OUT_DIR}/hparam_search_smiley_refined_best.json"
REJECTION_SAMPLES_PATH = f"{OUT_DIR}/rejection_smiley_samples.pt"

# Fixed, not searched.
EULER_STEPS = 200
INNER_STEPS = 1
INIT_CONTROL = "zero"
EYE_MOUTH_WEIGHT = 15.0
GAMMA_THRESHOLD_MAX = 0.32  # ceiling, not floor -- corrected after an earlier misstatement
GAMMA_THRESHOLD_FLOOR = 0.05  # avoid the near-degenerate threshold~0 regime

# Same box target + derived smiley geometry as plot_guided_euler_ramachandran.py / hparam_search.py.
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

FIELDNAMES = [
    "trial_idx",
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


def sample_params(rng: random.Random) -> dict:
    # Roughly 1/3 constant gamma, 2/3 schedule (trial 35 was a schedule; still
    # want a real chance at the qualitatively different constant regime).
    mode = rng.choice(["schedule", "schedule", "constant"])
    if mode == "constant":
        gamma_value = 10 ** rng.uniform(-0.3, 0.7)  # ~0.5 to 5.0
        gamma_hi = gamma_lo_slope = gamma_threshold = float("nan")
    else:
        gamma_value = float("nan")
        gamma_hi = 10 ** rng.uniform(0.0, 0.7)  # ~1.0 to 5.0
        gamma_lo_slope = 10 ** rng.uniform(-0.7, 0.4)  # ~0.2 to 2.5
        gamma_threshold = rng.uniform(GAMMA_THRESHOLD_FLOOR, GAMMA_THRESHOLD_MAX)

    return {
        "gamma_mode": mode,
        "gamma_value": gamma_value,
        "gamma_hi": gamma_hi,
        "gamma_lo_slope": gamma_lo_slope,
        "gamma_threshold": gamma_threshold,
        "lr": 10 ** rng.uniform(-2.5, -1.5),  # ~0.003 to 0.03
        "w_terminal": 10 ** rng.uniform(1.0, 2.0),  # ~10 to 100
        "w_vf": rng.choice([0.0, 0.0, 0.0, 0.01, 0.05]),
        "w_control": 10 ** rng.uniform(-4.0, -1.7),  # ~1e-4 to 0.02
    }


def _jitter(rng: random.Random, value: float, spread: float = 0.15) -> float:
    """Multiplicative log-uniform jitter: value * 10**U(-spread, spread).

    spread=0.15 -> roughly a 0.7x-1.4x swing (a "small step"), vs. sample_params'
    multi-decade wide draws.
    """
    return value * 10 ** rng.uniform(-spread, spread)


def sample_params_local(rng: random.Random, best: dict | None, explore_prob: float = 0.15) -> dict:
    """Small perturbation around the current best qualifying trial.

    Keeps best's gamma_mode and nudges every numeric parameter by a small
    multiplicative factor, rather than sample_params' wide independent
    redraws -- once we have a strong anchor, wide resampling mostly just
    lands far away in a landscape we've seen is very sensitive to these
    parameters. Occasionally (explore_prob) still takes a fresh wide draw so
    the search doesn't get stuck if the local neighborhood is a dead end.
    """
    if best is None or rng.random() < explore_prob:
        return sample_params(rng)

    if best["gamma_mode"] == "constant":
        gamma_value = _jitter(rng, best["gamma_value"])
        gamma_hi = gamma_lo_slope = gamma_threshold = float("nan")
    else:
        gamma_value = float("nan")
        gamma_hi = _jitter(rng, best["gamma_hi"])
        gamma_lo_slope = _jitter(rng, best["gamma_lo_slope"])
        gamma_threshold = max(GAMMA_THRESHOLD_FLOOR, min(GAMMA_THRESHOLD_MAX, _jitter(rng, best["gamma_threshold"])))

    return {
        "gamma_mode": best["gamma_mode"],
        "gamma_value": gamma_value,
        "gamma_hi": gamma_hi,
        "gamma_lo_slope": gamma_lo_slope,
        "gamma_threshold": gamma_threshold,
        "lr": _jitter(rng, best["lr"]),
        "w_terminal": _jitter(rng, best["w_terminal"]),
        "w_vf": best["w_vf"],
        "w_control": _jitter(rng, best["w_control"]),
    }


def seeded_trials() -> list[dict]:
    """Trial 35 itself, plus its gamma_hi as a constant, as anchors/sanity checks."""
    return [
        {
            "gamma_mode": "schedule",
            "gamma_value": float("nan"),
            "gamma_hi": 1.8984965629030828,
            "gamma_lo_slope": 0.5230874825897318,
            "gamma_threshold": 0.32033163683763044,
            "lr": 0.014377868564663446,
            "w_terminal": 39.207502403773994,
            "w_vf": 0.0,
            "w_control": 0.006536065551995293,
        },
        {
            "gamma_mode": "constant",
            "gamma_value": 1.8984965629030828,
            "gamma_hi": float("nan"),
            "gamma_lo_slope": float("nan"),
            "gamma_threshold": float("nan"),
            "lr": 0.014377868564663446,
            "w_terminal": 39.207502403773994,
            "w_vf": 0.0,
            "w_control": 0.006536065551995293,
        },
    ]


def run_trial(model, eval_ctx, phi_idx, psi_idx, chirality_checker, num_atoms, device, rejection_data, params: dict) -> dict:
    model.use_guidance = True
    model.guidance_cost_fn = make_cost_fn(phi_idx, psi_idx, chirality_checker)
    model.guidance_num_steps = EULER_STEPS
    model.guidance_inner_steps = INNER_STEPS
    model.guidance_gamma = make_gamma(params)
    model.guidance_lr = params["lr"]
    model.guidance_w_terminal = params["w_terminal"]
    model.guidance_w_vf = params["w_vf"]
    model.guidance_w_control = params["w_control"]
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


def load_existing_results() -> list[dict]:
    """Resume support: re-parse a previous run's CSV back into result dicts."""
    if not Path(CSV_PATH).exists():
        return []
    numeric_fields = [
        "trial_idx",
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
    ]
    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            r = dict(row)
            for k in numeric_fields:
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    r[k] = float("nan")
            r["failed"] = r["failed"] == "True"
            rows.append(r)
    return rows


def write_best(results: list[dict]) -> None:
    qualifying = [r for r in results if not r["failed"] and r["frac_in_face"] >= FRAC_IN_FACE_TARGET]
    qualifying.sort(key=lambda r: r["vs_rejection_energy_w2"])
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

    rejection_data = torch.load(REJECTION_SAMPLES_PATH, weights_only=False)
    print(f"loaded rejection baseline: {rejection_data['energy'].shape[0]} accepted samples")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    csv_exists = Path(CSV_PATH).exists()
    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if not csv_exists:
        writer.writeheader()
        csv_file.flush()

    rng = random.Random(0)
    results = load_existing_results()
    queue = [] if results else seeded_trials()  # skip re-running the anchors if already recorded
    current_best = None  # best qualifying (frac_in_face >= target, not failed) result so far
    for r in results:
        if not r["failed"] and r["frac_in_face"] >= FRAC_IN_FACE_TARGET:
            if current_best is None or r["vs_rejection_energy_w2"] < current_best["vs_rejection_energy_w2"]:
                current_best = r
    if results:
        print(f"resuming: {len(results)} existing trials loaded, current best energy_w2={current_best['vs_rejection_energy_w2'] if current_best else 'none'}")

    start_time = time.time()
    trial_idx = len(results)
    while time.time() - start_time < BUDGET_SECONDS:
        free_gb = shutil.disk_usage("/").free / 1e9
        if free_gb < MIN_FREE_DISK_GB:
            print(f"Free disk on / dropped to {free_gb:.2f} GB, stopping search early.")
            break

        params = queue.pop(0) if queue else sample_params_local(rng, current_best)
        result = run_trial(model, eval_ctx, phi_idx, psi_idx, chirality_checker, num_atoms, device, rejection_data, params)
        result["trial_idx"] = trial_idx
        results.append(result)

        if (
            not result["failed"]
            and result["frac_in_face"] >= FRAC_IN_FACE_TARGET
            and (current_best is None or result["vs_rejection_energy_w2"] < current_best["vs_rejection_energy_w2"])
        ):
            current_best = result

        writer.writerow({k: result[k] for k in FIELDNAMES})
        csv_file.flush()

        status = (
            "FAILED"
            if result["failed"]
            else f"frac_in_face={result['frac_in_face']:.3f} vs_rejection_energy_w2={result['vs_rejection_energy_w2']:.3f}"
        )
        elapsed_min = (time.time() - start_time) / 60
        gamma_desc = (
            f"const={params['gamma_value']:.3g}"
            if params["gamma_mode"] == "constant"
            else f"hi={params['gamma_hi']:.3g} lo={params['gamma_lo_slope']:.3g} t*={params['gamma_threshold']:.3g}"
        )
        best_marker = " *new best*" if current_best is result else ""
        print(
            f"[{elapsed_min:6.1f} min] trial {trial_idx}: gamma[{gamma_desc}] lr={params['lr']:.4g} "
            f"w_term={params['w_terminal']:.3g} w_control={params['w_control']:.3g} -> {status}{best_marker}",
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
