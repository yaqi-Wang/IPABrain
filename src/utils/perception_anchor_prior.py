"""Legacy-style image-level, similarity-weighted Gaussian parameter prior."""
from __future__ import annotations
import math
import torch


def aggregate_anchor_parameters(means, logvars, image_ids, similarities):
    """Average repeated trials per image, then weight means and logvars by cosine.

    This reproduces the first-version parameter-aggregation idea, not the
    moments of a Gaussian mixture. Candidate identities must have been split
    into training/validation *before* calling this function.
    """
    if means.shape != logvars.shape or means.ndim < 2 or len(means)==0:
        raise ValueError('Expected aligned nonempty posterior tensors')
    if len(image_ids)!=len(means) or len(similarities)!=len(means):
        raise ValueError('Image IDs, similarities and posterior tensors must align')
    means=means.float();logvars=logvars.float()
    if not bool(torch.isfinite(means).all() and torch.isfinite(logvars).all()):
        raise ValueError('Posterior parameters must be finite')
    groups={}
    for index,(image,similarity) in enumerate(zip(image_ids,similarities)):
        image=int(image);similarity=float(similarity)
        if not math.isfinite(similarity):raise ValueError('Similarity must be finite')
        group=groups.setdefault(image,{'indices':[],'similarity':similarity})
        if abs(group['similarity']-similarity)>1e-6:
            raise ValueError('Repeated trials of an image must share its retrieval similarity')
        group['indices'].append(index)
    ordered=sorted(groups)
    scores=torch.tensor([max(groups[i]['similarity'],0.0) for i in ordered],dtype=torch.float32,device=means.device)
    if not bool(scores.sum()>0):raise ValueError('All retrieved similarities are nonpositive')
    weights=scores/scores.sum()
    aggregate_mean=torch.zeros_like(means[0]);aggregate_logvar=torch.zeros_like(logvars[0])
    for index,image in enumerate(ordered):
        indices=groups[image]['indices']
        aggregate_mean.add_(means[indices].mean(dim=0)*weights[index])
        aggregate_logvar.add_(logvars[indices].mean(dim=0)*weights[index])
    return aggregate_mean,aggregate_logvar,{'image_ids':ordered,'weights':weights.cpu().tolist(),
        'trials_per_image':[len(groups[i]['indices']) for i in ordered]}
