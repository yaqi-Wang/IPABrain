"""Numerically bounded diagonal-Gaussian teacher priors for imagery R+KL.

The teacher approximation matches the first two moments of an equally or
explicitly weighted mixture; it is not the exact mixture distribution.
All tensors are in the *same* frozen BrainVAE latent space, before w010.
"""
from __future__ import annotations

import torch


def teacher_moments(means, logvars, weights=None, *, min_logvar=-12.0, max_logvar=8.0):
    """Moment-match N teacher posteriors with tensors shaped [N, ...]."""
    if means.shape != logvars.shape or means.ndim < 2 or means.shape[0] == 0:
        raise ValueError('Expected equally shaped nonempty teacher means/logvars')
    means, logvars = means.float(), logvars.float()
    if not bool(torch.isfinite(means).all() and torch.isfinite(logvars).all()):
        raise ValueError('Teacher posterior parameters must be finite')
    if not min_logvar < max_logvar:
        raise ValueError('Invalid log-variance bounds')
    n = means.shape[0]
    if weights is None:
        weights = torch.ones(n, dtype=means.dtype, device=means.device)
    else:
        weights = torch.as_tensor(weights, dtype=means.dtype, device=means.device)
    if weights.shape != (n,) or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool(weights.sum() > 0):
        raise ValueError('Expected finite, nonnegative, nonzero teacher weights')
    weights = (weights / weights.sum()).view(n, *([1] * (means.ndim - 1)))
    mu = (weights * means).sum(dim=0)
    # Centered form avoids catastrophic cancellation for similar large means.
    var = (weights * (logvars.clamp(min_logvar, max_logvar).exp() + (means - mu).square())).sum(dim=0)
    logvar = var.clamp(min=torch.exp(means.new_tensor(min_logvar)),
                       max=torch.exp(means.new_tensor(max_logvar))).log()
    return mu, logvar


def diagonal_gaussian_kl(q_mean, q_logvar, p_mean, p_logvar, *, min_logvar=-12.0, max_logvar=8.0):
    """KL(q || p), averaged over batch and latent dimensions, in FP32."""
    if not min_logvar < max_logvar:
        raise ValueError('Invalid log-variance bounds')
    if q_mean.shape != q_logvar.shape or p_mean.shape != p_logvar.shape or q_mean.shape != p_mean.shape:
        raise ValueError('q and p posterior parameters must have identical shapes')
    q_mean, q_logvar, p_mean, p_logvar = (x.float() for x in (q_mean, q_logvar, p_mean, p_logvar))
    if not all(bool(torch.isfinite(x).all()) for x in (q_mean, q_logvar, p_mean, p_logvar)):
        raise ValueError('Posterior parameters must be finite')
    q_logvar = q_logvar.clamp(min_logvar, max_logvar)
    p_logvar = p_logvar.clamp(min_logvar, max_logvar)
    kl = 0.5 * (p_logvar - q_logvar + (q_logvar - p_logvar).exp()
                + (q_mean - p_mean).square() * (-p_logvar).exp() - 1.0)
    return kl.mean()


def replay_r_kl_backward(adapter, mapping, inputs, teacher, labels, prior_mean, prior_logvar,
                         micro, scaler, beta, *, weight=0.1, temperature=0.05, check_replay=False,
                         base_replay=None):
    """Exact R logical-batch gradient plus per-example KL, with dropout replay.

    mapping.brain must be in eval mode and expose encode() with mean/logvar.
    priors are per-label arrays [classes, ...], learned from training teachers.
    """
    if base_replay is None:
        from src.utils.imagery_w010_pilot import replay_backward as base_replay

    beta = float(beta)
    if not 0.0 <= beta < float('inf'):
        raise ValueError('beta must be finite and nonnegative')
    if len(inputs) == 0 or labels.shape != (len(inputs),) or micro <= 0:
        raise ValueError('Invalid logical batch')
    if mapping.brain.training:
        raise ValueError('The perception BrainVAE must remain frozen in eval mode')
    if prior_mean.shape != prior_logvar.shape or prior_mean.ndim < 2:
        raise ValueError('Prior tensors must be matched and include a class dimension')
    if bool((labels < 0).any()) or bool((labels >= len(prior_mean)).any()):
        raise ValueError('Prior labels out of range')
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state(inputs.device) if inputs.is_cuda else None
    result = base_replay(adapter, mapping, inputs, teacher, labels, micro, scaler,
                         weight, temperature, check_replay=check_replay)
    cpu_after = torch.get_rng_state()
    cuda_after = torch.cuda.get_rng_state(inputs.device) if inputs.is_cuda else None
    if beta == 0:
        result['kl'] = 0.0
        return result
    torch.set_rng_state(cpu_before)
    if inputs.is_cuda:
        torch.cuda.set_rng_state(cuda_before, inputs.device)
    kl_total = 0.0
    try:
        for start in range(0, len(inputs), micro):
            end = min(start + micro, len(inputs))
            from torch.utils.checkpoint import checkpoint
            def posterior_parameters(value):
                posterior = mapping.brain.encode(value)
                return posterior.mean, posterior.logvar
            mean, logvar = checkpoint(posterior_parameters, adapter(inputs[start:end]), use_reentrant=False)
            lab = labels[start:end]
            kl = diagonal_gaussian_kl(mean, logvar,
                                      prior_mean[lab], prior_logvar[lab])
            ratio = (end - start) / len(inputs)
            scaler.scale(beta * ratio * kl).backward()
            kl_total += ratio * float(kl.detach())
    finally:
        torch.set_rng_state(cpu_after)
        if inputs.is_cuda:
            torch.cuda.set_rng_state(cuda_after, inputs.device)
    result['kl'] = kl_total
    result['total'] += beta * kl_total
    return result
