from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F


def soft_clip_loss(preds: torch.Tensor, targets: torch.Tensor, temp: float = 0.125) -> torch.Tensor:
    preds = F.normalize(preds.flatten(1), dim=-1)
    targets = F.normalize(targets.flatten(1), dim=-1)
    target_sim = (targets @ targets.T) / temp
    pred_sim = (preds @ targets.T) / temp
    loss1 = -(pred_sim.log_softmax(-1) * target_sim.softmax(-1)).sum(-1).mean()
    loss2 = -(pred_sim.T.log_softmax(-1) * target_sim.softmax(-1)).sum(-1).mean()
    return (loss1 + loss2) * 0.5


def mse_latent_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(preds, targets)


def gaussian_kl_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    prior_mean: torch.Tensor | None = None,
    prior_logvar: torch.Tensor | None = None,
) -> torch.Tensor:
    if prior_mean is None:
        prior_mean = torch.zeros_like(mean)
    if prior_logvar is None:
        prior_logvar = torch.zeros_like(logvar)
    logvar = logvar.clamp(min=-30.0, max=20.0)
    prior_logvar = prior_logvar.clamp(min=-30.0, max=20.0)
    var_ratio = torch.exp(logvar - prior_logvar)
    mean_delta = (mean - prior_mean).pow(2) / torch.exp(prior_logvar).clamp_min(1e-8)
    kl = 0.5 * (prior_logvar - logvar + var_ratio + mean_delta - 1.0)
    return kl.mean()


def repeated_trial_consistency_loss(latents: torch.Tensor, stimulus_ids: list[str]) -> torch.Tensor:
    flat = F.normalize(latents.flatten(1), dim=-1)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, stimulus_id in enumerate(stimulus_ids):
        groups[str(stimulus_id)].append(idx)

    losses = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        group = flat[indices]
        sims = group @ group.T
        mask = ~torch.eye(len(indices), dtype=torch.bool, device=latents.device)
        losses.append((1.0 - sims[mask]).mean())
    if not losses:
        return latents.new_tensor(0.0)
    return torch.stack(losses).mean()
