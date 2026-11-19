"""Original SynBrain architecture/loss with the project's training interface.

Checkpoint keys match upstream src/vae/brainvae.py. In particular this uses
the original *unclamped* 2-D Gaussian, nin_shortcut, trans_conv, vanilla
attention, and the original decoder interpolation order.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.ldm.modules.distributions.distributions import DiagonalGaussianDistribution_2D
from src.models.brainvae import _token_mlp, soft_clip_loss
from src.models.synbrain_blocks import NeuroDecoder, NeuroEncoder


class SynBrainVAE(nn.Module):
    def __init__(self, ddconfig: dict, hidden_dim: int = 4096, linear_dim: int = 2048,
                 embed_dim: int = 1664, latent_tokens: int = 256,
                 hidden_channels: int = 128, clip_weight: float = 1000.0,
                 kl_weight: float = 0.001, semantic_source: str = "sample",
                 free_bits: float = 0.0, clip_reg_weight: float = 0.0,
                 recon_weight: float = 1.0, activation_checkpointing: bool = False):
        super().__init__()
        if semantic_source != "sample" or free_bits != 0 or clip_reg_weight != 0 or recon_weight != 1:
            raise ValueError("Original SynBrain requires sample SoftCLIP, free_bits=0, "
                             "clip_reg_weight=0 and recon_weight=1")
        if min(hidden_dim, linear_dim, embed_dim, latent_tokens, hidden_channels) <= 0:
            raise ValueError("SynBrain dimensions must be positive")
        if any(not math.isfinite(v) or v < 0 for v in (clip_weight, kl_weight)):
            raise ValueError("Loss weights must be finite and nonnegative")
        self.ddconfig = dict(ddconfig)
        if self.ddconfig.get("double_z", True):
            raise ValueError("SynBrain projects mean/logvar separately: ddconfig.double_z must be false")
        if latent_tokens != self.ddconfig["z_channels"] or hidden_channels != self.ddconfig["ch"]:
            raise ValueError("SynBrain latent_tokens / hidden_channels differ from ddconfig")
        depth = len(self.ddconfig["ch_mult"])
        down = int(self.ddconfig["num_down_blocks"])
        up = int(self.ddconfig["num_up_blocks"])
        if not 0 <= down <= depth or not 0 <= up <= depth:
            raise ValueError("Invalid SynBrain down/up block count")
        length = int(self.ddconfig["ada_length"])
        for _ in range(down):
            length = ((length + 1) // 2 if self.ddconfig.get("resamp_with_conv", True) else length // 2)
        if length != hidden_dim:
            raise ValueError(f"SynBrain encoder output length {length} != hidden_dim {hidden_dim}; "
                             "use the configuration accompanying the author checkpoint")
        self.hidden_dim, self.embed_dim, self.latent_tokens = hidden_dim, embed_dim, latent_tokens
        self.clip_weight, self.kl_weight = float(clip_weight), float(kl_weight)
        self.semantic_source = semantic_source
        self.free_bits, self.clip_reg_weight, self.recon_weight = 0.0, 0.0, 1.0
        self.activation_checkpointing = activation_checkpointing
        self.encoder = NeuroEncoder(**self.ddconfig)
        self.decoder = NeuroDecoder(**self.ddconfig)
        self.pre_projector_mean = _token_mlp(hidden_dim, linear_dim, embed_dim)
        self.pre_projector_logvar = _token_mlp(hidden_dim, linear_dim, embed_dim)
        self.post_projector = _token_mlp(embed_dim, linear_dim, hidden_dim)

    def _run(self, module, *args):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def encode(self, fmri: torch.Tensor) -> DiagonalGaussianDistribution_2D:
        if fmri.ndim == 2:
            fmri = fmri.unsqueeze(1)
        if fmri.ndim != 3 or fmri.shape[1] != self.ddconfig["in_channels"]:
            raise ValueError("SynBrain expects fMRI shaped (N,V) or (N,1,V)")
        hidden = self._run(self.encoder, fmri)
        # Keep the original Gaussian/KL arithmetic FP32 when an optional AMP
        # training loop autocasts the convolution/MLP. FP32 runs are unchanged.
        mean = self.pre_projector_mean(hidden).float()
        logvar = self.pre_projector_logvar(hidden).float()
        return DiagonalGaussianDistribution_2D(torch.cat((mean, logvar), dim=1))

    def decode(self, latent: torch.Tensor, target_length: int) -> torch.Tensor:
        return self._run(self.decoder, self.post_projector(latent), target_length)

    def forward(self, fmri: torch.Tensor, clip_target: torch.Tensor,
                sample_posterior: bool = True, kl_weight_override: float | None = None):
        if fmri.ndim == 2:
            fmri = fmri.unsqueeze(1)
        posterior = self.encode(fmri)
        latent = posterior.sample() if sample_posterior else posterior.mode()
        if clip_target.shape != latent.shape:
            raise ValueError("SynBrain posterior and target CLIP token shapes differ")
        recon = self.decode(latent, fmri.shape[-1])
        recon_loss = F.mse_loss(recon, fmri, reduction="sum") / len(fmri)
        kl_loss = posterior.kl().sum() / len(fmri)
        clip_loss = soft_clip_loss(latent, clip_target) if self.clip_weight else latent.new_tensor(0.0)
        active_kl = self.kl_weight if kl_weight_override is None else kl_weight_override
        if not math.isfinite(active_kl) or active_kl < 0:
            raise ValueError("Invalid active KL weight")
        loss = recon_loss + active_kl * kl_loss + self.clip_weight * clip_loss
        return latent, recon, {"loss": loss, "recon_loss": recon_loss,
                               "kl_loss": kl_loss, "clip_loss": clip_loss}
