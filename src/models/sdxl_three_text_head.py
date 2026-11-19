"""One shared frozen-brain-mean trunk with three SDXL text-condition outputs."""
from __future__ import annotations

import torch
from torch import nn

from src.models.sdxl_text_head import CrossAttentionBlock, sinusoidal_positions


class ThreeConditionTextHead(nn.Module):
    """μ [B,256,1664] -> CLIP-L, bigG token, and bigG pooled conditions."""

    def __init__(self, target_means: dict, width: int = 256, heads: int = 8,
                 depth: int = 2, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        if (width <= 0 or heads <= 0 or width % heads or width % 2 or depth <= 0
                or mlp_ratio <= 0 or not 0 <= dropout < 1):
            raise ValueError('Invalid three-condition head dimensions or dropout')
        expected = {'clip_l': (77, 768), 'bigg_tokens': (77, 1280), 'bigg_pooled': (1280,)}
        if set(target_means) != set(expected):
            raise ValueError('All three training-only target means are required')
        for name, shape in expected.items():
            value = target_means[name]
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError('Invalid train-only mean for {}'.format(name))
            self.register_buffer('mean_' + name, value.detach().float().clone(), persistent=True)
        self.input_norm = nn.LayerNorm(1664)
        self.input_projection = nn.Linear(1664, width)
        self.register_buffer('brain_position', sinusoidal_positions(256, width), persistent=True)
        self.text_queries = nn.Parameter(torch.empty(1, 77, width))
        nn.init.normal_(self.text_queries, std=0.02)
        self.blocks = nn.ModuleList([CrossAttentionBlock(width, heads, mlp_ratio, dropout)
                                     for _ in range(depth)])
        self.output_norm = nn.LayerNorm(width)
        self.clip_l = nn.Linear(width, 768)
        self.bigg_tokens = nn.Linear(width, 1280)
        self.bigg_pooled = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                         nn.GELU(), nn.Linear(width, 1280))
        for layer in (self.clip_l, self.bigg_tokens, self.bigg_pooled[-1]):
            nn.init.normal_(layer.weight, std=0.001)
            nn.init.zeros_(layer.bias)

    def forward(self, mean: torch.Tensor) -> dict:
        if mean.ndim != 3 or tuple(mean.shape[1:]) != (256, 1664):
            raise ValueError('BrainVAE mean must be shaped [B,256,1664]')
        memory = self.input_projection(self.input_norm(mean)) + self.brain_position
        queries = self.text_queries.expand(len(mean), -1, -1)
        for block in self.blocks:
            queries = block(queries, memory)
        hidden = self.output_norm(queries)
        return {
            'clip_l': self.clip_l(hidden).float() + self.mean_clip_l,
            'bigg_tokens': self.bigg_tokens(hidden).float() + self.mean_bigg_tokens,
            'bigg_pooled': self.bigg_pooled(hidden.mean(dim=1)).float() + self.mean_bigg_pooled,
        }
