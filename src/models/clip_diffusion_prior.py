"""Independent noisy-CLIP denoiser conditioned on frozen BrainVAE means."""
from __future__ import annotations

import math

import torch
from torch import nn

from src.models.clip_mapping import _check_shape


def sinusoidal_embedding(values: torch.Tensor, width: int) -> torch.Tensor:
    frequency = torch.exp(torch.arange(0, width, 2, device=values.device, dtype=torch.float32)
                          * (-math.log(10000.0)/width))
    angles = values.float().unsqueeze(-1)*frequency
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)


class ConditionalDenoisingBlock(nn.Module):
    def __init__(self, width, heads, mlp_ratio, dropout):
        super().__init__()
        self.time_projection = nn.Linear(width, width)
        self.self_norm, self.cross_norm, self.ffn_norm = (nn.LayerNorm(width) for _ in range(3))
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(width, int(width*mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(int(width*mlp_ratio), width), nn.Dropout(dropout))

    def forward(self, value, condition, time):
        value = value+self.time_projection(time).unsqueeze(1)
        normalized = self.self_norm(value)
        value = value+self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        value = value+self.cross_attention(self.cross_norm(value), condition, condition, need_weights=False)[0]
        return value+self.ffn(self.ffn_norm(value))


class ClipConditionalPrior(nn.Module):
    def __init__(self, latent_tokens=256, embed_dim=1664, hidden_dim=512, condition_depth=2,
                 depth=4, num_heads=8, mlp_ratio=4.0, dropout=0.1,
                 gradient_checkpointing=False):
        super().__init__()
        if min(latent_tokens, embed_dim, hidden_dim, condition_depth, depth, num_heads) <= 0:
            raise ValueError("Prior dimensions/depth/heads must be positive")
        if hidden_dim % 2 or hidden_dim % num_heads:
            raise ValueError("Prior width must be even and divisible by heads")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid prior FFN ratio/dropout")
        self.latent_tokens, self.embed_dim = latent_tokens, embed_dim
        self.hidden_dim, self.gradient_checkpointing = hidden_dim, gradient_checkpointing
        self.register_buffer('position_embedding', sinusoidal_embedding(torch.arange(latent_tokens), hidden_dim)[None])
        self.condition_projection = nn.Linear(embed_dim, hidden_dim)
        self.condition_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, int(hidden_dim*mlp_ratio), dropout,
                                       activation='gelu', batch_first=True, norm_first=True)
            for _ in range(condition_depth)])
        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.noisy_projection = nn.Linear(embed_dim, hidden_dim)
        self.time_embedding = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*4), nn.SiLU(),
                                            nn.Linear(hidden_dim*4, hidden_dim))
        self.denoising_blocks = nn.ModuleList([
            ConditionalDenoisingBlock(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.output_norm, self.output_projection = nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, embed_dim)

    def _run(self, block, *inputs):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(block, *inputs, use_reentrant=False)
        return block(*inputs)

    def encode_condition(self, mean):
        _check_shape(mean, self.latent_tokens, self.embed_dim)
        hidden = self.condition_projection(mean)+self.position_embedding
        for block in self.condition_blocks:
            hidden = self._run(block, hidden)
        return self.condition_norm(hidden)

    def predict_noise(self, noisy_clip, timesteps, condition):
        _check_shape(noisy_clip, self.latent_tokens, self.embed_dim)
        if (timesteps.shape != (len(noisy_clip),) or timesteps.dtype != torch.long
                or condition.shape != (len(noisy_clip), self.latent_tokens, self.hidden_dim)):
            raise ValueError("Invalid prior timesteps/condition shape")
        time = self.time_embedding(sinusoidal_embedding(timesteps, self.hidden_dim))
        hidden = self.noisy_projection(noisy_clip)+self.position_embedding
        for block in self.denoising_blocks:
            hidden = self._run(block, hidden, condition, time)
        return self.output_projection(self.output_norm(hidden))


def build_clip_prior(config):
    options = dict(config)
    if options.pop('type') != 'clip_conditional_prior':
        raise ValueError('Expected independent clip_conditional_prior')
    return ClipConditionalPrior(**options)
