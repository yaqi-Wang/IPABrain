from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    flat = x.reshape(len(x), -1).astype(np.float32)
    return flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), eps)


def cosine_topk(query: np.ndarray, bank: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    query_norm = l2_normalize(query)
    bank_norm = l2_normalize(bank)
    sims = np.einsum("id,jd->ij", query_norm, bank_norm, optimize=False)
    k = min(k, bank_norm.shape[0])
    indices = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    sorted_order = np.take_along_axis(sims, indices, axis=1).argsort(axis=1)[:, ::-1]
    top_indices = np.take_along_axis(indices, sorted_order, axis=1)
    top_sims = np.take_along_axis(sims, top_indices, axis=1)
    return top_indices, top_sims


def aggregate_retrieval_priors(
    indices: np.ndarray,
    similarities: np.ndarray,
    latent_mean,
    latent_logvar,
    eps: float = 1e-8,
) -> tuple:
    import torch

    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=latent_mean.device)
    sim_tensor = torch.as_tensor(similarities, dtype=latent_mean.dtype, device=latent_mean.device)
    weights = sim_tensor.clamp_min(0.0)
    weight_sum = weights.sum(dim=1).clamp_min(eps)
    weights = weights / weight_sum.unsqueeze(1)

    selected_mean = latent_mean[index_tensor]
    selected_logvar = latent_logvar[index_tensor]
    prior_mean = (selected_mean * weights[:, :, None, None]).sum(dim=1)
    prior_logvar = (selected_logvar * weights[:, :, None, None]).sum(dim=1)
    return prior_mean, prior_logvar, weight_sum.cpu()
