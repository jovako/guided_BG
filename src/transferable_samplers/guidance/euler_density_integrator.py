"""Guided fixed-step Euler integrator with EXACT log-density, for ECNF++ / flow matching.

Builds the discrete step function directly rather than a generic vector field
``dx/dt = F(t,x)``, to match ``_integrate_guided``'s actual recursion:

    cxt    = x + gamma * u         (u from n_inner steps of normalized GD)
    x_next = cxt + dt * v_control  (v_control = net(t, cxt))

``_integrate_guided`` steps FROM the perturbed point ``cxt``, not from ``x``:
the control shift ``gamma*u`` is teleported into the trajectory, not merely
used to compute a velocity, so it can't be expressed as a plain ``F(t,x)``
fed through a generic integrator.

At ``alpha=0``/``n_inner=0`` this reduces to plain unguided Euler
(``x + dt*net(t,x)``), so ``check_unguided_consistency`` validates the
integrator/log-det machinery independent of guidance.

Design notes:

- Inner loop uses ``torch.func.grad`` (plain ``torch.autograd.grad`` can't
  call ``requires_grad_()`` inside a functorch transform). But nesting
  ``torch.func.grad``/``vjp`` inside ANOTHER functorch transform's primal
  pass (e.g. ``jacrev``) crashes on this torch build -- so the exact-density
  Jacobian below uses plain ``torch.autograd.grad`` instead of ``jacrev``.
- No ``.detach()``/``no_grad()`` anywhere in the inner loop -- a detached
  iterate makes ``du*/dx == 0`` and silently gives a wrong log-density.
- Inner-loop step is L2-normalized per sample (``u -= alpha*g/(||g||+eps)``),
  not raw gradient descent -- raw gradient magnitude varies by orders of
  magnitude along the trajectory, making an unnormalized step unstable.
- Both the inner-loop gradient and the outer Jacobian exploit that the
  network has no cross-sample coupling: summing a per-sample scalar over the
  batch before one backward call recovers every sample's own gradient/row at
  once, making both costs ~independent of batch size instead of O(batch_size).

Cost: ~1-2 network calls per step (batched), plus, when ``track_density=True``,
``d`` backward passes total for the Jacobian (``d = num_atoms*dims``).
``track_density=False`` skips the Jacobian for cheap hyperparameter search;
turn it on only to verify a winning config, after running
``check_unguided_consistency``.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.func import functional_call, grad, vmap

Tensor = torch.Tensor

__all__ = [
    "make_guided_euler_step",
    "guided_euler_exact",
    "generate_proposal_guided_euler",
    "check_unguided_consistency",
]


# --------------------------------------------------------------------------------------
# The guided step (inner loop + teleport + velocity, all in one differentiable map)
# --------------------------------------------------------------------------------------


def make_guided_euler_step(
    net: Callable,
    encodings: dict[str, Tensor] | None,
    terminal_cost: Callable[[Tensor], Tensor],
    *,
    dt: float,
    gamma: float | Callable[[Tensor], float] = 1.0,
    alpha: float = 0.1,
    lam: float = 0.0,
    beta: float = 0.0,
    n_inner: int = 1,
    use_score_deviation: bool = True,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Build ``step(t, x) -> x_next`` for a batch (``x`` shape ``(B, d)``).

    Matches ``_integrate_guided``'s recursion -- see module docstring.

    Args:
        net: Flow-matching network, called as ``net(t, x, encodings=...)``
            with ``t`` shape ``(1,)`` (broadcasts to any batch size) and
            ``x`` shape ``(B, d)``. Pass a detached ``copy.deepcopy``.
        encodings: System conditioning, batched to match ``x``.
        terminal_cost: ``C(x1_hat) -> scalar``, single-sample convention
            (``(d,)`` in, scalar out) -- vmapped internally over the batch.
        dt: Euler step size, baked in at construction (the teleport term
            isn't ``dt``-rescalable by a generic integrator).
        gamma: Control weight in ``cxt = x + gamma*u``, constant or callable of ``t``.
        alpha: Inner step length (L2-normalized per sample).
        lam: Weight of ``||u||^2``.
        beta: Weight of the score-deviation regularizer.
        n_inner: Inner gradient-descent steps (0 or 1 in practice; ``>1``
            accumulates correctly across iterations but is untested here).
        use_score_deviation: If False, skip the extra network call for the beta term.
    """

    _net_params = {k: v.detach() for k, v in net.named_parameters()}
    _net_buffers = {k: v.detach() for k, v in net.named_buffers()}

    def f_theta(t: Tensor, x: Tensor) -> Tensor:
        return functional_call(
            net, (_net_params, _net_buffers),
            args=(t.reshape(1), x), kwargs={"encodings": encodings},
        )

    terminal_cost_batched = vmap(terminal_cost)  # (B, d) -> (B,)

    def inner_objective_sum(u: Tensor, t: Tensor, x: Tensor, f0: Tensor, gamma_t: Tensor) -> Tensor:
        # Sum over the batch: valid since samples don't interact, and lets one
        # backward call recover every sample's own gradient (see module docstring).
        cxt = x + gamma_t * u
        v_control = f_theta(t, cxt)
        x1_hat = cxt + (1.0 - t) * v_control  # linear extrapolation from cxt to t=1
        total = terminal_cost_batched(x1_hat).sum()

        if lam != 0.0:
            total = total + lam * (u * u).sum()

        if use_score_deviation and beta != 0.0:
            total = total + beta * ((v_control - f0) ** 2).sum()

        return total

    def step(t: Tensor, x: Tensor) -> Tensor:
        gamma_t = gamma(t) if callable(gamma) else torch.as_tensor(gamma, dtype=x.dtype, device=x.device)
        f0 = f_theta(t, x) if (use_score_deviation and beta != 0.0) else torch.zeros((), dtype=x.dtype, device=x.device)
        u = torch.zeros_like(x)
        for i in range(n_inner):
            if i == 0:
                u.requires_grad_(True)  # later iterations reuse u's own graph
            total_obj = inner_objective_sum(u, t, x, f0, gamma_t)
            (g,) = torch.autograd.grad(total_obj, u, create_graph=True)
            g_norm = g.norm(dim=-1, keepdim=True)  # per-sample, not a batch-global norm
            u = u - alpha * g / (g_norm + 1e-8)
        cxt = x + gamma_t * u
        v_control = f_theta(t, cxt)
        return cxt + dt * v_control

    return step


# --------------------------------------------------------------------------------------
# The Euler integrator
# --------------------------------------------------------------------------------------


def guided_euler_exact(
    make_step: Callable[[float], Callable[[Tensor, Tensor], Tensor]],
    z: Tensor,
    *,
    n_steps: int = 100,
    exact_logdet: bool = True,
    check_orientation: bool = True,
    track_density: bool = True,
    raise_on_orientation_failure: bool = True,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Integrate t=0 to t=1 with a guided Euler step, optionally accumulating
    the exact log-Jacobian.

    Differentiates the step with plain ``torch.autograd.grad``, one call per
    output coordinate ``j`` (``d`` calls total, not ``d`` per sample): each
    call sums ``x_next[:, j]`` over the batch before differentiating, which
    recovers every sample's own row ``j`` at once (see module docstring).
    ``M`` already equals ``dx_next/dx`` (identity included), so
    ``slogdet(M)`` is the exact per-step log-density correction directly.

    Args:
        make_step: ``dt -> step(t, x) -> x_next``, batched (``x`` shape
            ``(B, d)``), e.g.
            ``functools.partial(make_guided_euler_step, net, encodings, terminal_cost, ...)``.
        z: Prior samples, flattened, shape ``(B, d)``.
        n_steps: Number of uniform Euler steps.
        exact_logdet: True -> ``log|det(dx_next/dx)|`` (exact). False ->
            ``tr(M - I)`` (continuous-limit surrogate, comparison only).
            Ignored if ``track_density`` is False.
        check_orientation: Track, per sample, whether every step stayed
            orientation-preserving (non-positive determinant => that
            sample's density is degenerate there). Ignored if
            ``track_density`` is False.
        track_density: If False, skip the Jacobian and just run the primal
            step -- as cheap as plain guided sampling, for hyperparameter
            search; turn on only to verify a winning config's density.
        raise_on_orientation_failure: If True (default), raise as soon as any
            sample fails orientation. If False, keep going and only flag
            failures via the returned ``valid`` mask -- use this with large
            batches where a minority failing (typically near ``t=1``)
            shouldn't discard the rest; filter with ``x[valid]``/``logdet[valid]``.

    Returns:
        x: ``(B, d)`` samples.
        logdet: ``(B,)`` total ``log|det dT/dz|`` (``log q(x) = prior.logp(z) - logdet``),
            or ``None`` if ``track_density`` is False.
        valid: ``(B,)`` bool mask, True where orientation held at every step,
            or ``None`` if ``track_density`` or ``check_orientation`` is False.
    """
    batch_size, d = z.shape
    dt = 1.0 / n_steps
    x = z.clone()
    step = make_step(dt=dt)

    if not track_density:
        for k in range(n_steps):
            t = torch.as_tensor(k * dt, device=z.device, dtype=z.dtype)
            x = step(t, x).detach()  # break the graph -- no backward needed here, and
            # the inner loop's create_graph=True would otherwise retain every step's
            # graph, growing memory with n_steps until it OOMs.
        return x, None, None

    logdet = torch.zeros(batch_size, device=z.device, dtype=z.dtype)
    eye = torch.eye(d, device=z.device, dtype=z.dtype)
    valid = torch.ones(batch_size, dtype=torch.bool, device=z.device) if check_orientation else None

    for k in range(n_steps):
        t = torch.as_tensor(k * dt, device=z.device, dtype=z.dtype)
        x_ = x.detach().requires_grad_(True)  # one leaf for the whole batch
        x_next = step(t, x_)
        rows = [
            torch.autograd.grad(x_next[:, j].sum(), x_, retain_graph=(j < d - 1))[0]
            for j in range(d)
        ]
        M = torch.stack(rows, dim=1)  # (B, d, d): M[b, j, k] = d x_next[b, j] / d x_[b, k]
        x_next = x_next.detach()

        if exact_logdet:
            sign, ld = torch.linalg.slogdet(M)
            if check_orientation:
                step_valid = sign > 0
                if raise_on_orientation_failure and not bool(step_valid.all()):
                    raise RuntimeError(
                        f"Euler step {k} is not orientation-preserving "
                        f"(det <= 0 for {(~step_valid).sum().item()} samples). "
                        f"Reduce dt/alpha, or pass raise_on_orientation_failure=False "
                        f"to filter those samples out instead of aborting."
                    )
                valid = valid & step_valid
            logdet = logdet + ld
        else:
            logdet = logdet + torch.diagonal(M - eye, dim1=-2, dim2=-1).sum(-1)

        x = x_next

    return x, logdet, valid


# --------------------------------------------------------------------------------------
# Plugging into the SNIS path
# --------------------------------------------------------------------------------------


def generate_proposal_guided_euler(
    model,
    num_samples: int,
    num_atoms: int,
    terminal_cost: Callable[[Tensor], Tensor],
    *,
    system_cond=None,
    gamma: float | Callable[[Tensor], float] = 1.0,
    alpha: float = 0.1,
    lam: float = 0.0,
    beta: float = 0.0,
    n_inner: int = 1,
    n_steps: int = 100,
    use_score_deviation: bool = True,
    device: str | torch.device = "cpu",
    **kw,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Drop-in replacement for ``FlowMatchingModule.generate_proposal`` using
    guided fixed-step Euler with an exact log-density.

    Mirrors the stock contract: returns ``(x, -log q)`` shaped
    ``(num_samples, num_atoms, dims)`` / ``(num_samples,)``, matching
    ``generate_proposal``'s sign convention, plus a third ``valid`` return
    (see ``guided_euler_exact``). All ``num_samples`` are always returned;
    pass ``raise_on_orientation_failure=False`` and filter with
    ``x[valid]``/``neg_logq[valid]`` yourself if wanted.

    Args:
        model: A ``FlowMatchingModule`` (only ``.net``/``.prior`` used --
            unlike ``generate_proposal``, doesn't need a Lightning
            ``Trainer``, so ``num_atoms``/``device`` are explicit).
        num_samples: Number of samples to draw.
        num_atoms: Atoms per conformation (ignored if ``system_cond`` given).
        terminal_cost: Single-sample cost, see ``make_guided_euler_step``.
        system_cond: Optional conditioning, same contract as ``generate_proposal``.
        gamma, alpha, lam, beta, n_inner, use_score_deviation: See ``make_guided_euler_step``.
        n_steps: Number of Euler steps.
        device: Device to sample on.
        **kw: Forwarded to ``guided_euler_exact`` (e.g. ``exact_logdet``,
            ``check_orientation``, ``track_density``, ``raise_on_orientation_failure``).
    """
    import copy
    from functools import partial

    if system_cond is not None:
        num_atoms = system_cond.encodings["atom_type"].size(0)

    z = model.prior.sample(num_samples, num_atoms, device=device)
    logp_z = model.prior.logp(z)

    batched_cond = system_cond.for_batch(num_samples, device) if system_cond else None
    encodings = batched_cond.encodings if batched_cond else None

    net = copy.deepcopy(model.net)
    net.requires_grad_(False)
    make_step = partial(
        make_guided_euler_step, net, encodings, terminal_cost,
        gamma=gamma, alpha=alpha, lam=lam, beta=beta, n_inner=n_inner,
        use_score_deviation=use_score_deviation,
    )

    z_flat = z.reshape(num_samples, -1)
    x_flat, logdet, valid = guided_euler_exact(make_step, z_flat, n_steps=n_steps, **kw)
    x = x_flat.reshape(num_samples, num_atoms, -1)

    if logdet is None:  # track_density=False -- no density to return
        return x, None, None

    dlogp = -logdet
    logq = logp_z + dlogp
    return x, -logq, valid


# --------------------------------------------------------------------------------------
# Verification -- run this before trusting a single guided number
# --------------------------------------------------------------------------------------


def check_unguided_consistency(model, z: Tensor, n_steps_list: tuple[int, ...] = (100, 200, 400, 800)) -> dict:
    """``alpha=0`` must reproduce the stock dopri5 log q as ``n_steps -> inf``.

    Master correctness test: validates the integrator, log-det convention,
    sign, and prior handling together, with guidance off (``alpha=0``,
    ``n_inner=0`` -- ``u`` stays exactly zero) so any discrepancy is purely
    about the integrator. Euler is 1st-order, so expect needing more steps
    than an RK4 equivalent to converge.

    Args:
        model: A ``FlowMatchingModule`` instance.
        z: Prior samples, shape ``(batch, num_atoms, dims)`` (unflattened).
        n_steps_list: Euler step counts to compare against dopri5.
    """
    import copy
    from functools import partial

    batch_size = z.shape[0]
    net = copy.deepcopy(model.net)
    net.requires_grad_(False)

    make_step0 = partial(
        make_guided_euler_step, net, None, terminal_cost=lambda x1: (x1 * 0.0).sum(),
        alpha=0.0, n_inner=0, use_score_deviation=False,
    )

    logp_z = model.prior.logp(z)
    ref_x, ref_dlogp = model._integrate(model.net, z, encodings=None, reverse=False, compute_dlogp=True)
    ref_logq = logp_z + ref_dlogp

    z_flat = z.reshape(batch_size, -1)
    out = {}
    for n in n_steps_list:
        try:
            x_flat, logdet, _valid = guided_euler_exact(make_step0, z_flat, n_steps=n)
            logq = logp_z - logdet
            out[n] = {"logq_mean": logq.mean().item(), "logq_std": logq.std().item()}
        except RuntimeError as exc:
            # Orientation failure -- a genuine finding (see module docstring),
            # not a bug. Report and keep going; finer steps is the fix.
            out[n] = {"error": repr(exc)[:200]}
    out["dopri5"] = {"logq_mean": ref_logq.mean().item(), "logq_std": ref_logq.std().item()}
    return out
