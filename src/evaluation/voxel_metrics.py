from __future__ import annotations

import torch
import torch.nn.functional as F


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.mse_loss(pred.float(), target.float()).item()


def cosine(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    pred = F.normalize(pred, dim=-1, eps=eps)
    target = F.normalize(target, dim=-1, eps=eps)
    return F.cosine_similarity(pred, target, dim=-1).mean().item()


def pearson(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = (pred * target).sum(dim=1)
    denominator = pred.norm(dim=1) * target.norm(dim=1)
    return (numerator / denominator.clamp_min(eps)).mean().item()


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.float().flatten(1)
    y = y.float().flatten(1)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xy = torch.linalg.norm(x.T @ y).pow(2)
    xx = torch.linalg.norm(x.T @ x)
    yy = torch.linalg.norm(y.T @ y)
    return (xy / (xx * yy).clamp_min(eps)).item()


def voxel_metric_dict(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "mse": mse(pred, target),
        "pearson": pearson(pred, target),
        "cosine": cosine(pred, target),
        "cka": linear_cka(pred, target),
    }

