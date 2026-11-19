"""Original SynBrain encoder + mean projector, trained only against CLIP tokens.

No fMRI decoder, log-variance branch, posterior sampling, KL, or prediction head.
SDPA preserves the original single-head attention equation and parameter keys.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.models.brainvae import _token_mlp, soft_clip_loss
from src.models.synbrain_blocks import AttnBlock, NeuroEncoder, nonlinearity


class SDPAAttnBlock(AttnBlock):
    def forward(self, x):
        hidden = self.norm(x)
        q, k, v = [layer(hidden).transpose(1, 2).unsqueeze(1)
                   for layer in (self.q, self.k, self.v)]
        # One head of width C, matching the original q @ k / sqrt(C).
        # Under BF16 the fused kernel handles scaling and softmax internally.
        hidden = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        return x + self.proj_out(hidden.squeeze(1).transpose(1, 2))


class CLIPNeuroEncoder(NeuroEncoder):
    def __init__(self, ddconfig, activation_checkpointing=True, sdpa_attention=True):
        super().__init__(**ddconfig)
        self.activation_checkpointing = activation_checkpointing
        if sdpa_attention:
            for name, module in list(self.named_modules()):
                if isinstance(module, AttnBlock):
                    replacement = SDPAAttnBlock(module.in_channels)
                    replacement.load_state_dict(module.state_dict(), strict=True)
                    parent_name, child = name.rsplit('.', 1)
                    setattr(self.get_submodule(parent_name), child, replacement)

    def _run(self, module, *args):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, x):
        h = self._run(lambda value: self.ada_maxpool(self.conv_in(value)), x)
        for level in self.down:
            for index, block in enumerate(level.block):
                h = self._run(block, h)
                if len(level.attn):
                    h = self._run(level.attn[index], h)
            if hasattr(level, 'downsample'):
                h = self._run(level.downsample, h)
        h = self._run(self.mid.block_1, h)
        h = self._run(self.mid.attn_1, h)
        h = self._run(self.mid.block_2, h)
        return self._run(lambda value: self.conv_out(nonlinearity(self.norm_out(value))), h)


class BrainVAECLIPEncoder(nn.Module):
    def __init__(self, voxels, ddconfig, hidden_dim=4096, linear_dim=2048,
                 tokens=256, channels=1664, activation_checkpointing=True,
                 sdpa_attention=True):
        super().__init__()
        if ddconfig.get('double_z', True) or ddconfig['z_channels'] != tokens:
            raise ValueError('Expected original separate mean projection and matching token count')
        if ddconfig.get('dropout', 0.) != 0.:
            raise ValueError('Full-batch gradient replay requires deterministic dropout=0')
        if ddconfig.get('attn_type', 'vanilla') != 'vanilla' or ddconfig.get('use_linear_attn', False):
            raise ValueError('This comparison preserves original vanilla single-head attention')
        length = ddconfig['ada_length']
        for _ in range(ddconfig['num_down_blocks']):
            length = (length+1)//2 if ddconfig.get('resamp_with_conv', True) else length//2
        if length != hidden_dim:
            raise ValueError('Encoder length and mean projector input differ')
        self.voxels, self.tokens, self.channels = voxels, tokens, channels
        self.encoder = CLIPNeuroEncoder(ddconfig, activation_checkpointing, sdpa_attention)
        self.pre_projector_mean = _token_mlp(hidden_dim, linear_dim, channels)
        self.activation_checkpointing = activation_checkpointing

    def forward(self, fmri):
        if fmri.ndim != 2 or fmri.shape[1] != self.voxels:
            raise ValueError(f'Expected real fMRI shaped [B,{self.voxels}]')
        hidden = self.encoder(fmri[:, None])
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            mu = checkpoint(self.pre_projector_mean, hidden, use_reentrant=False)
        else:
            mu = self.pre_projector_mean(hidden)
        if mu.shape[1:] != (self.tokens, self.channels):
            raise ValueError('Predicted CLIP token shape changed')
        return mu


def build_brainvae_clip_encoder(voxels, config):
    options = dict(config)
    if options.pop('type') != 'brainvae_clip_encoder':
        raise ValueError('Expected the encoder-only BrainVAE architecture')
    return BrainVAECLIPEncoder(voxels, **options)


def clip_supervision_losses(prediction, target, objective):
    """Raw-coordinate MSE + optional soft-target CLIP or hard-label InfoNCE.

    'contrastive_loss' is the shared trainer's generic name for the unweighted
    pairwise term. objective.type records whether it is SoftCLIP or InfoNCE.
    """
    if prediction.shape != target.shape or len(prediction) < 2:
        raise ValueError('Paired CLIP supervision requires matching tokens and >=2 samples')
    kind = objective['type']
    weight, temperature = float(objective['weight']), float(objective['temperature'])
    if kind not in ('mse_softclip', 'mse_infonce', 'mse'):
        raise ValueError(f'Unknown supervised encoder loss: {kind}')
    if not math.isfinite(weight) or weight < 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid loss coefficient or temperature')
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction, target = prediction.float(), target.detach().float()
        mse = F.mse_loss(prediction, target)
        if kind == 'mse_softclip':
            pairwise = soft_clip_loss(prediction, target, temp=temperature)
        elif kind == 'mse_infonce':
            logits = F.normalize(prediction.flatten(1), dim=1) @ F.normalize(target.flatten(1), dim=1).T / temperature
            labels = torch.arange(len(prediction), device=prediction.device)
            pairwise = (F.cross_entropy(logits, labels)+F.cross_entropy(logits.T, labels))/2
        else:
            pairwise = prediction.new_zeros(())
        total = mse + weight*pairwise
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite supervised encoder loss')
    return dict(total=total, gen_mse=mse, contrastive_loss=pairwise)
