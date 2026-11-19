"""Whole-candidate-set diagnostics, independent of validation batch boundaries."""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def representation_diagnostics(predictions, targets, device, block_size=32) -> dict:
    pred = torch.cat(predictions).float().flatten(1)
    true = torch.cat(targets).float().flatten(1)
    if pred.shape != true.shape or len(pred) < 2:
        raise ValueError("Diagnostics require at least 2 aligned representations")
    p, t = F.normalize(pred, dim=1), F.normalize(true, dim=1)
    matched = (p*t).sum(1)
    off_total, retrieved = 0.0, 0
    device = torch.device(device)
    candidate_pool = None
    if device.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        required = t.numel()*t.element_size()
        if free > required*2 + 512*1024**2:
            # Keep the fixed candidate set on device once instead of transferring
            # its 1.5 GB repeatedly for every query block at n=900.
            candidate_pool = t.to(device)
    for start in range(0, len(p), block_size):
        stop = min(start+block_size, len(p))
        query = p[start:stop].to(device)
        if candidate_pool is not None:
            similarities = (query @ candidate_pool.T).cpu()
        else:
            similarities = torch.cat([query @ t[j:j+block_size].to(device).T
                                      for j in range(0, len(t), block_size)], dim=1).cpu()
        retrieved += int((similarities.argmax(1) == torch.arange(start, stop)).sum())
        off_total += float(similarities.double().sum()) - float(matched[start:stop].double().sum())
    # Coordinate-wise centering is diagnostic only and never changes conditions.
    def energy(values):
        total = torch.zeros(values.shape[1], dtype=torch.float64)
        squared = 0.0
        for block in values.split(block_size):
            block = block.double()
            total += block.sum(0)
            squared += float(block.flatten().dot(block.flatten()))
        return squared/len(values)-float((total/len(values)).square().sum())
    wrong = off_total/(len(pred)*(len(pred)-1))
    return {"candidate_count": len(pred), "matched_cosine": float(matched.mean()),
            "off_diagonal_cosine": wrong, "match_minus_mismatch": float(matched.mean())-wrong,
            "norm_ratio": float((pred.norm(dim=1)/true.norm(dim=1).clamp_min(1e-8)).mean()),
            "centered_energy_ratio": energy(pred)/max(energy(true), 1e-12),
            "retrieval_top1": retrieved/len(pred), "centering_usage": "diagnostic_only"}
