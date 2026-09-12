"""Run SNIS (self-normalized importance sampling) on the density-sampling
pool, restricted to samples that are both `valid` (orientation-preserving,
so their exact density is trustworthy) and `in_smiley_mask` (actually inside
the constraint region -- outside it the combined target energy is a soft
penalty, not the true energy, so those points shouldn't be resampled as if
they were valid draws from the constrained equilibrium).

Uses the same convention as snis_sampler.py: `logw = E_source - E_target`,
here `E_source = neg_logq` (the guided proposal's own -log q, already exact
since we restricted to valid samples) and `E_target = energy_combined`
(energy_md + smiley_cost, from prepare_snis_target_energy.py) -- multinomial
resampling via the same `resampling_idx` helper the real SNISSampler uses.

Reports effective sample size (ESS) and weight concentration as a check that
the guided proposal isn't so different in shape from the true constrained
distribution that a handful of samples dominate the resampled set.

Run with:
    uv run python tests/guidance/sampling/run_snis_on_pool.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from transferable_samplers.samplers.resampling import resampling_idx

POOL_PATH = Path("tests/guidance/out/snis_pool_smiley/combined_pool_with_target_energy.pt")
SAVE_PATH = POOL_PATH.parent / "snis_resampled.pt"
SEED = 42


def main() -> None:
    d = torch.load(POOL_PATH, weights_only=False)
    mask = d["valid"] & d["in_smiley_mask"]
    n_subset = mask.sum().item()
    print(f"Subset (valid & in_smiley): {n_subset}/{len(mask)}")

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

    torch.manual_seed(SEED)
    resample_index = resampling_idx(logw, "multinomial")
    n_unique = resample_index.unique().numel()
    print(f"Resampled {len(resample_index)} draws (with replacement), {n_unique} unique source samples "
          f"({100 * n_unique / n_subset:.1f}% of the subset used at least once)")

    resampled = {
        "x": d["x"][mask][resample_index],
        "x_phys": d["x_phys"][mask][resample_index],
        "phi": d["phi"][mask][resample_index],
        "psi": d["psi"][mask][resample_index],
        "energy_md": d["energy_md"][mask][resample_index],
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
