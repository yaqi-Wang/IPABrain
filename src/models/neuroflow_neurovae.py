"""Strict NeuroFlow NeuroVAE_P checkpoint adapter for frozen-mean heads."""
from __future__ import annotations

import torch

from src.ldm.modules.distributions.distributions import DiagonalGaussianDistribution_2D
from src.vendor.neuroflow.neurovae import NeuroVAE_P


class NeuroFlowNeuroVAE(NeuroVAE_P):
    """Expose the original generation posterior through the project's encode API.

    NeuroFlow's ``h = pre_proj1(encoder(x)).chunk(2, dim=1)[0]`` conditions
    generation. ``clip_proj(encoder(x))`` is its separate retrieval branch.
    Every original module is retained so loading all author weights is strict.
    """

    def __init__(self, ddconfig: dict, latent_tokens: int = 256, embed_dim: int = 1664):
        channels = int(ddconfig['ch']) * int(ddconfig['ch_mult'][-1])
        length = int(ddconfig['latent_dim'])
        for _ in range(int(ddconfig['num_down_blocks'])):
            length = (length + 1) // 2 if ddconfig.get('resamp_with_conv', True) else length // 2
        if latent_tokens != 256 or channels != 256 or ddconfig['z_channels'] != 256:
            raise ValueError('The published NeuroVAE_P has fixed 256-channel projection layers')
        if length != embed_dim:
            raise ValueError(f'NeuroVAE encoder length {length} differs from embed_dim {embed_dim}')
        super().__init__(ddconfig=ddconfig, clip_size=embed_dim)
        self.ddconfig = dict(ddconfig)
        self.latent_tokens, self.embed_dim = latent_tokens, embed_dim

    def encode(self, fmri: torch.Tensor) -> DiagonalGaussianDistribution_2D:
        if fmri.ndim == 2:
            fmri = fmri.unsqueeze(1)
        if fmri.ndim != 3 or tuple(fmri.shape[1:]) != (1, self.ddconfig['voxel_dim']):
            raise ValueError('NeuroVAE expects matching-subject fMRI shaped (B,V) or (B,1,V)')
        # No retrieval projector, sampling, CLIP normalization or image-VAE scale.
        return DiagonalGaussianDistribution_2D(self.pre_proj1(self.encoder(fmri)))
