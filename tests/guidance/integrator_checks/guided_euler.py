"""
Guided fixed-step Euler integrator with EXACT log-density, for ECNF++ / flow matching.

Handles the case where the inner guidance loop RE-EVALUATES f_theta (score-deviation
regulariser), so the correction u*(x) is no longer a network-free map of the endpoint
estimate and the divergence must be taken through the whole unrolled inner loop.

Key design decisions:

1.  The whole guided field is written as a PURE function F(t, x) of a single sample,
    built only from torch.func-composable primitives.  The Jacobian is then just
    jacrev(F), which unrolls the inner loop automatically.  This is the only way to
    get the *correct* divergence: any hand-written "div f_theta + div u" split is
    wrong once the inner loop calls the network.

2.  The inner optimiser uses torch.func.grad, NOT torch.autograd.grad.  autograd.grad
    does not compose inside a jacrev transform.

3.  NO .detach() and NO torch.no_grad() anywhere inside the inner loop.  A detached
    iterate makes du*/dx == 0 and yields a silently wrong (but plausible-looking)
    log-density.  This is the single most likely bug in this file.

4.  The density is the EXACT log-det of the discrete Euler map,
        log|det(I + dt * J_F)|
    not the first-order surrogate dt * tr(J_F).  With that choice the integrator is a
    discrete normalising flow whose log q is exact BY CONSTRUCTION: discretisation
    error moves the proposal, but never biases the importance weights.  That is what
    SNIS needs.  d = 66, so the slogdet is free.

Rough cost estimate, RTX 3060, batch 8, alanine dipeptide (d = 66):
    unguided dopri5 baseline ............................. ~0.4  samples/s
    guided Euler, n_steps=100, N_inner=3 ................. ~0.04 samples/s
    guided Euler, n_steps=100, N_inner=5 ................. ~0.03 samples/s
Budget a pilot of 500-1000 samples, not 10_000.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.func import grad, jacrev, vmap

Tensor = torch.Tensor


# --------------------------------------------------------------------------------------
# 1. The guided vector field
# --------------------------------------------------------------------------------------


def make_guided_field(
    net: Callable,
    encodings,
    terminal_cost: Callable[[Tensor], Tensor],
    *,
    alpha: float = 0.1,
    lam: float = 0.0,
    beta: float = 0.0,
    n_inner: int = 3,
    use_score_deviation: bool = True,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Build F(t, x) -> dx for a SINGLE sample (x has shape (d,), t has shape ()).

    Args:
        net: the flow-matching network, called as net(t: (1,), x: (1, d)) -> (1, d).
        encodings: system conditioning, closed over (constant w.r.t. x).
        terminal_cost: C(x1_hat) -> scalar.  For a (phi, psi) fiber target this is
            e.g.  0.5 * ||wrap(dihedrals(x1_hat) - R_star)||^2.  Must be built from
            torch.func-composable ops (no .item(), no in-place on inputs).
        alpha: inner step size.
        lam: weight of the control cost ||u||^2.
        beta: weight of the score-deviation regulariser.
        n_inner: number of inner gradient steps.
        use_score_deviation: if False, the inner objective is network-free and the
            whole thing collapses to the cheap case (kept here so you can A/B it with
            one flag and reuse the identical density code).
    """

    def f_theta(t: Tensor, x: Tensor) -> Tensor:
        return net(t.reshape(1), x.view(1, -1), encodings=encodings).flatten()

    def inner_objective(u: Tensor, t: Tensor, x: Tensor, f0: Tensor) -> Tensor:
        # Endpoint estimate under the correction (linear path, sigma = 0).
        x1_hat = x + (1.0 - t) * (f0 + u)
        obj = terminal_cost(x1_hat)

        if lam != 0.0:
            obj = obj + lam * (u * u).sum()

        if use_score_deviation and beta != 0.0:
            # THE re-evaluation.  This is what makes u* a non-trivial function of the
            # network and forces the divergence through the inner loop.
            f_pert = f_theta(t, x + (1.0 - t) * u)
            obj = obj + beta * ((f_pert - f0) ** 2).sum()

        return obj

    grad_u = grad(inner_objective, argnums=0)  # torch.func.grad -- composable

    def F(t: Tensor, x: Tensor) -> Tensor:
        f0 = f_theta(t, x)
        u = torch.zeros_like(x)
        for _ in range(n_inner):
            g = grad_u(u, t, x, f0)
            u = u - alpha * g  # no detach, no no_grad
        return f0 + u

    return F


# --------------------------------------------------------------------------------------
# 2. The integrator
# --------------------------------------------------------------------------------------


def guided_euler(
    F: Callable[[Tensor, Tensor], Tensor],
    z: Tensor,
    *,
    n_steps: int = 100,
    exact_logdet: bool = True,
    check_orientation: bool = True,
) -> tuple[Tensor, Tensor]:
    """Integrate x' = F(t, x) from t=0 to t=1 and accumulate the exact log-Jacobian.

    Args:
        F: guided field for a single sample, from make_guided_field.
        z: prior samples, shape (B, d).
        n_steps: number of uniform Euler steps.
        exact_logdet: True  -> log|det(I + dt J)|   (exact for the discrete map)
                      False -> dt * tr(J)           (continuous-limit surrogate)
        check_orientation: assert every Euler step is orientation-preserving.

    Returns:
        x:      (B, d) samples
        logdet: (B,)   total log|det dT/dz|, so that
                       log q(x) = prior.logp(z) - logdet
    """
    B, d = z.shape
    dt = 1.0 / n_steps
    x = z.clone()
    logdet = torch.zeros(B, device=z.device, dtype=z.dtype)
    eye = torch.eye(d, device=z.device, dtype=z.dtype)

    # jacrev with has_aux gives Jacobian AND value from one primal pass.
    def F_aux(t: Tensor, x_: Tensor):
        v = F(t, x_)
        return v, v

    jac_and_val = vmap(jacrev(F_aux, argnums=1, has_aux=True), in_dims=(None, 0))

    for k in range(n_steps):
        t = torch.as_tensor(k * dt, device=z.device, dtype=z.dtype)
        J, Fx = jac_and_val(t, x)  # (B, d, d), (B, d)

        if exact_logdet:
            sign, ld = torch.linalg.slogdet(eye + dt * J)
            if check_orientation and not bool((sign > 0).all()):
                raise RuntimeError(
                    f"Euler step {k} is not orientation-preserving "
                    f"(det <= 0 for {(sign <= 0).sum().item()} samples). "
                    f"The discrete map is not injective -- reduce dt or alpha."
                )
            logdet = logdet + ld
        else:
            logdet = logdet + dt * torch.diagonal(J, dim1=-2, dim2=-1).sum(-1)

        x = x + dt * Fx

    return x, logdet


# --------------------------------------------------------------------------------------
# 3. Plugging into the SNIS path
# --------------------------------------------------------------------------------------


def generate_proposal_guided(module, num_samples: int, F, *, n_steps=100, **kw):
    """Drop-in replacement for FlowMatchingModule.generate_proposal.

    Mirrors the stock contract: returns (x, -log q).  E_source for the SNIS sampler is
    exactly this second return value, so the weights logw = E_source - E_target stay
    consistent with the map you actually applied.
    """
    z = module.prior.sample(num_samples)
    logp_z = module.prior.logp(z)
    x, logdet = guided_euler(F, z, n_steps=n_steps, **kw)
    logq = logp_z - logdet
    return x, -logq


# --------------------------------------------------------------------------------------
# 4. Verification -- run all of these before trusting a single number
# --------------------------------------------------------------------------------------


def check_unguided_consistency(module, net, encodings, z, n_steps_list=(100, 200, 400)):
    """alpha=0 must reproduce the stock dopri5 log q as n_steps -> inf.

    This is the master correctness test: it validates the integrator, the log-det
    convention, the sign, and the prior handling all at once, with guidance switched
    off so any discrepancy is purely integration.
    """
    F0 = make_guided_field(
        net, encodings, terminal_cost=lambda x1: x1.sum() * 0.0,
        alpha=0.0, n_inner=0, use_score_deviation=False,
    )
    ref_x, ref_neg_logq = module.generate_proposal(len(z))  # stock dopri5
    out = {}
    for n in n_steps_list:
        x, logdet = guided_euler(F0, z, n_steps=n)
        logq = module.prior.logp(z) - logdet
        out[n] = {"logq_mean": logq.mean().item(), "logq_std": logq.std().item()}
    out["dopri5"] = {
        "logq_mean": (-ref_neg_logq).mean().item(),
        "logq_std": (-ref_neg_logq).std().item(),
    }
    return out


def check_divergence_finite_difference(F, t, x, eps=1e-4):
    """Central-difference the divergence against jacrev. Should agree to ~1e-4 rel.

    THE test that catches a detach in the inner loop: finite differences see the true
    field, jacrev sees the detached one, and they will disagree by an O(1) amount.
    """
    d = x.shape[0]
    J = jacrev(F, argnums=1)(t, x)
    div_ad = torch.diagonal(J).sum()

    div_fd = torch.zeros((), device=x.device, dtype=x.dtype)
    for i in range(d):
        e = torch.zeros_like(x)
        e[i] = eps
        div_fd = div_fd + (F(t, x + e)[i] - F(t, x - e)[i]) / (2 * eps)

    rel = (div_ad - div_fd).abs() / div_fd.abs().clamp_min(1e-12)
    return {"ad": div_ad.item(), "fd": div_fd.item(), "rel_err": rel.item()}


def check_translation_modes(F, t, x, tol=1e-5):
    """The mean-free prior means 3 global-translation directions should be exact null
    modes of J.  If they are, slogdet over the full 66-dim space equals slogdet over
    the 63-dim mean-free subspace (those eigenvalues are 1 in I + dt J) and the
    density is correct without any explicit projection.  If they are NOT, the network
    is not centring its input and you must restrict the log-det to the subspace.
    """
    J = jacrev(F, argnums=1)(t, x)
    n_atoms = x.shape[0] // 3
    res = {}
    ok = True
    for axis in range(3):
        v = torch.zeros_like(x).view(n_atoms, 3)
        v[:, axis] = 1.0
        v = (v / v.norm()).flatten()
        nrm = (J @ v).norm().item()
        res[f"axis_{axis}_Jv_norm"] = nrm
        ok = ok and nrm < tol
    res["pass"] = ok
    return res
