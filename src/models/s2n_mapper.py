from __future__ import annotations

import math

import torch
import torch.nn as nn


class S2NMapper(nn.Module):
    def __init__(
        self,
        latent_tokens: int = 256,
        embed_dim: int = 1664,
        depth: int = 8,
        num_heads: int = 13,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.latent_tokens = latent_tokens
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, latent_tokens, embed_dim), requires_grad=False)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.final = nn.Linear(embed_dim, embed_dim)
        self._init_pos_embed()

    def _init_pos_embed(self) -> None:
        position = torch.arange(self.latent_tokens, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim))
        pe = torch.zeros(self.latent_tokens, self.embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.pos_embed.data.copy_(pe.unsqueeze(0))

    def forward(self, clip_latent: torch.Tensor) -> torch.Tensor:
        if clip_latent.ndim == 2:
            clip_latent = clip_latent.unsqueeze(1).expand(-1, self.latent_tokens, -1)
        if clip_latent.shape[1] != self.latent_tokens:
            clip_latent = torch.nn.functional.interpolate(
                clip_latent.transpose(1, 2),
                size=self.latent_tokens,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return self.final(self.blocks(clip_latent + self.pos_embed.to(clip_latent.device)))

