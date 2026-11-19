"""A compact BrainVAE-mean to SDXL CLIP-L token baseline.

The input is the frozen BrainVAE posterior mean, not the output of w010.
The fixed output mean is estimated on the internal training images only.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_positions(length: int, width: int) -> torch.Tensor:
    if width <= 0 or width % 2:
        raise ValueError('width must be a positive even integer')
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
    result = torch.empty(1, length, width)
    result[0, :, 0::2] = torch.sin(position * frequency)
    result[0, :, 1::2] = torch.cos(position * frequency)
    return result


class CrossAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * mlp_ratio), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(width * mlp_ratio, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        values = self.memory_norm(memory)
        attended = self.attention(self.query_norm(queries), values, values, need_weights=False)[0]
        queries = queries + self.dropout(attended)
        return queries + self.dropout(self.ff(self.ff_norm(queries)))


class MeanToCLIPTextHead(nn.Module):
    """[B,256,1664] posterior mean -> [B,77,768] unprojected CLIP-L tokens."""

    def __init__(self, training_target_mean: torch.Tensor, width: int = 256,
                 heads: int = 8, depth: int = 2, mlp_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        if (width <= 0 or heads <= 0 or width % heads or width % 2 or depth <= 0
                or mlp_ratio <= 0 or not 0 <= dropout < 1):
            raise ValueError('Invalid text-head dimensions or dropout')
        if tuple(training_target_mean.shape) != (77, 768) or not torch.isfinite(training_target_mean).all():
            raise ValueError('Training target mean must be finite [77,768]')
        self.input_norm = nn.LayerNorm(1664)
        self.input_projection = nn.Linear(1664, width)
        self.register_buffer('brain_position', sinusoidal_positions(256, width), persistent=True)
        self.register_buffer('target_mean', training_target_mean.detach().float().clone(), persistent=True)
        self.text_queries = nn.Parameter(torch.empty(1, 77, width))
        nn.init.normal_(self.text_queries, std=0.02)
        self.blocks = nn.ModuleList([CrossAttentionBlock(width, heads, mlp_ratio, dropout)
                                     for _ in range(depth)])
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, 768)
        # Start close to the train-only constant baseline but allow gradients
        # into the cross-attention trunk on the very first update.
        nn.init.normal_(self.output_projection.weight, std=0.001)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, mean: torch.Tensor) -> torch.Tensor:
        if mean.ndim != 3 or tuple(mean.shape[1:]) != (256, 1664):
            raise ValueError('BrainVAE mean must be shaped [B,256,1664]')
        memory = self.input_projection(self.input_norm(mean)) + self.brain_position
        queries = self.text_queries.expand(len(mean), -1, -1)
        for block in self.blocks:
            queries = block(queries, memory)
        delta = self.output_projection(self.output_norm(queries)).float()
        # Keep the large, fixed target mean and raw regression loss in FP32.
        return delta + self.target_mean
