"""Run SNIS (self-normalized importance sampling) on the current-hparams
pool, restricted to samples that are both `valid` (orientation-preserving)
and `in_smiley_mask` (inside the constraint region).

Sibling of run_snis_on_pool.py, pointed at the pool prepared by
prepare_snis_target_energy_current_hparams.py. Additionally reports summary
statistics (energy_md, phi, psi) two ways to make the SNIS correction's
effect visible: "without SNIS" (plain unweighted mean/std over the subset)
vs. "with SNIS" (importance-weighted using softmax(logw)).

`logw = E_source - E_target` (snis_sampler.py's convention), with
`E_source = neg_logq` and `E_target = energy_combined`; multinomial
resampling via `resampling_idx`.

Run with:
    uv run python tests/guidance/sampling/run_snis_on_pool_current_hparams.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from transferable_samplers.samplers.resampling import resampling_idx

POOL_PATH = Path("tests/guidance/out/snis_pool_smiley_current_hparams/combined_pool_with_target_energy.pt")
SAVE_PATH = POOL_PATH.parent / "snis_resampled.pt"
SEED = 42


def weighted_mean_std(x: torch.Tensor, w: torch.Tensor) -> tuple[float, float]:
    mean = (w * x).sum()
    var = (w * (x - mean) ** 2).sum()
    return mean.item(), var.sqrt().item()


def main() -> None:
    d = torch.load(POOL_PATH, weights_only=False)
    mask = d["valid"] & d["in_smiley_mask"]
    n_subset = mask.sum().item()
    print(f"Subset (valid & in_smiley, i.e. the zero-cost area of the face): {n_subset}/{len(mask)}")

    energy_md = d["energy_md"][mask]
    phi = d["phi"][mask]
    psi = d["psi"][mask]
    neg_logq = d["neg_logq"][mask]
    energy_combined = d["energy_combined"][mask]
    logw = neg_logq - energy_combined  # E_source - E_target, snis_sampler.py's convention

    print(f"logw stats: mean={logw.mean().item():.3f} std={logw.std().item():.3f} "
          f"min={logw.min().item():.3f} max={logw.max().item():.3f}")

    w = torch.softmax(logw, dim=-1)
    ess = 1.0 / (w**2).sum().item()
    print(f"Effective sample size: {ess:.1f} / {n_subset} ({100 * ess / n_subset:.1f}%)")

    top_k = min(10, n_subset)
    top_w, top_idx = torch.topk(w, top_k)
    print(f"Top-{top_k} weight mass: {top_w.sum().item():.4f} (of total 1.0) -- max single weight: {w.max().item():.4f}")

    # --- metrics without SNIS (raw, unweighted) vs with SNIS (importance-weighted) ---
    print("\nMetric                | without SNIS (raw)     | with SNIS (weighted)")
    for name, x in [("energy_md", energy_md), ("phi", phi), ("psi", psi)]:
        raw_mean, raw_std = x.mean().item(), x.std().item()
        snis_mean, snis_std = weighted_mean_std(x, w)
        print(f"{name:22s} | mean={raw_mean:8.3f} std={raw_std:6.3f} | mean={snis_mean:8.3f} std={snis_std:6.3f}")

    torch.manual_seed(SEED)
    resample_index = resampling_idx(logw, "multinomial")
    n_unique = resample_index.unique().numel()
    print(f"\nResampled {len(resample_index)} draws (with replacement), {n_unique} unique source samples "
          f"({100 * n_unique / n_subset:.1f}% of the subset used at least once)")

    # Cross-check: resampled (uniform-weight) stats should match the weighted stats above.
    print("\nCross-check (resampled, uniform-weight, should match 'with SNIS' above):")
    for name, x in [("energy_md", energy_md), ("phi", phi), ("psi", psi)]:
        resampled_x = x[resample_index]
        print(f"{name:22s} | mean={resampled_x.mean().item():8.3f} std={resampled_x.std().item():6.3f}")

    resampled = {
        "x": d["x"][mask][resample_index],
        "x_phys": d["x_phys"][mask][resample_index],
        "phi": phi[resample_index],
        "psi": psi[resample_index],
        "energy_md": energy_md[resample_index],
        "neg_logq": neg_logq[resample_index],
        "logw": logw,  # full pre-resampling log-weights for the subset, for diagnostics
        "resample_index": resample_index,
        "ess": ess,
        "n_subset": n_subset,
    }
    torch.save(resampled, SAVE_PATH)
    print(f"\nsaved resampled set to {SAVE_PATH}")


if __name__ == "__main__":
    main()
