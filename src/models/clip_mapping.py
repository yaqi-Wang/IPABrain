"""Brain posterior mean -> raw OpenCLIP image tokens (never CLIP -> brain)."""
from __future__ import annotations

import math

import torch
from torch import nn


class MeanBridge(nn.Module):
    """Shared per-token residual MLP, initialized to the identity mapping."""

    def __init__(self, latent_tokens: int = 256, embed_dim: int = 1664,
                 hidden_dim: int = 256) -> None:
        super().__init__()
        if min(latent_tokens, embed_dim, hidden_dim) <= 0:
            raise ValueError("MeanBridge dimensions must be positive")
        self.latent_tokens, self.embed_dim = latent_tokens, embed_dim
        self.residual = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(),
                                      nn.Linear(hidden_dim, embed_dim))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, mean: torch.Tensor) -> torch.Tensor:
        _check_shape(mean, self.latent_tokens, self.embed_dim)
        return mean + self.residual(mean)


class TokenFormer(nn.Module):
    """Standard Transformer with token mixing and an external residual path.

    Internal LayerNorm is allowed; the final raw-coordinate prediction is not
    L2-normalized or multiplied by a diffusion VAE scale factor.
    """

    def __init__(self, latent_tokens: int = 256, embed_dim: int = 1664,
                 hidden_dim: int = 512, depth: int = 8, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        if min(latent_tokens, embed_dim, hidden_dim, depth, num_heads) <= 0:
            raise ValueError("TokenFormer dimensions/depth/heads must be positive")
        if hidden_dim % num_heads or hidden_dim % 2:
            raise ValueError("TokenFormer hidden_dim must be even and divisible by num_heads")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid TokenFormer feedforward ratio or dropout")
        self.latent_tokens, self.embed_dim = latent_tokens, embed_dim
        self.input_projection = nn.Linear(embed_dim, hidden_dim)
        position = torch.arange(latent_tokens, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        positional = torch.zeros(1, latent_tokens, hidden_dim)
        positional[0, :, 0::2] = torch.sin(position * frequency)
        positional[0, :, 1::2] = torch.cos(position * frequency)
        self.register_buffer("position_embedding", positional)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio), dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        # TransformerEncoder clones a layer: initialize each clone independently.
        for block in self.transformer.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.zeros_(block.self_attn.in_proj_bias)
            for linear in (block.self_attn.out_proj, block.linear1, block.linear2):
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
        self.output_projection = nn.Linear(hidden_dim, embed_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, mean: torch.Tensor) -> torch.Tensor:
        _check_shape(mean, self.latent_tokens, self.embed_dim)
        hidden = self.input_projection(mean) + self.position_embedding
        return mean + self.output_projection(self.transformer(hidden))


class TokenFormerDualHead(TokenFormer):
    """Shared TokenFormer trunk with separate raw generation and semantic outputs."""

    def __init__(self, *args, semantic_hidden_dim: int = 512,
                 gradient_checkpointing: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if semantic_hidden_dim <= 0:
            raise ValueError("semantic_hidden_dim must be positive")
        self.gradient_checkpointing = gradient_checkpointing
        hidden_dim = self.input_projection.out_features
        self.semantic_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, semantic_hidden_dim),
            nn.GELU(), nn.Linear(semantic_hidden_dim, self.embed_dim),
        )

    def forward_features(self, mean: torch.Tensor) -> torch.Tensor:
        _check_shape(mean, self.latent_tokens, self.embed_dim)
        hidden = self.input_projection(mean) + self.position_embedding
        if not (self.gradient_checkpointing and self.training and torch.is_grad_enabled()):
            return self.transformer(hidden)
        from torch.utils.checkpoint import checkpoint
        for block in self.transformer.layers:
            hidden = checkpoint(block, hidden, use_reentrant=False)
        return hidden

    def forward(self, mean: torch.Tensor) -> torch.Tensor:
        return mean + self.output_projection(self.forward_features(mean))

    def forward_train(self, mean: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.forward_features(mean)
        return {"generation": mean + self.output_projection(hidden),
                "semantic": self.semantic_head(hidden)}


class AdaToken(nn.Module):
    """SiT-based mapping from perceptual posterior means to CLIP features.

    This follows its x-prediction launcher: t=0, adaLN-Zero, direct output,
    and no external mean residual. It starts at zero, rather than identity.
    Importing SiT lazily keeps timm optional for the other mapping heads.
    """

    def __init__(self, latent_tokens: int = 256, embed_dim: int = 1664,
                 depth: int = 8, num_heads: int = 13, mlp_ratio: float = 4.0,
                 fused_attn: bool = True, qk_norm: bool = False,
                 gradient_checkpointing: bool = True) -> None:
        super().__init__()
        if min(latent_tokens, embed_dim, depth, num_heads) <= 0:
            raise ValueError("AdaToken dimensions/depth/heads must be positive")
        if embed_dim % 2 or embed_dim % num_heads:
            raise ValueError("AdaToken embed_dim must be even and divisible by num_heads")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
            raise ValueError("Invalid AdaToken feedforward ratio")
        from src.s2n.sit import SiT

        self.latent_tokens, self.embed_dim = latent_tokens, embed_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.sit = SiT(num_patches=latent_tokens, hidden_size=embed_dim,
                       depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                       fused_attn=fused_attn, qk_norm=qk_norm)

    def forward(self, mean: torch.Tensor) -> torch.Tensor:
        _check_shape(mean, self.latent_tokens, self.embed_dim)
        timestep = mean.new_zeros(len(mean))
        if not (self.gradient_checkpointing and self.training and torch.is_grad_enabled()):
            return self.sit(mean, timestep)
        from torch.utils.checkpoint import checkpoint

        hidden = mean + self.sit.pos_embed
        conditioning = self.sit.t_embedder(timestep)
        for block in self.sit.blocks:
            hidden = checkpoint(block, hidden, conditioning, use_reentrant=False)
        return self.sit.final_layer(hidden, conditioning)


def _check_shape(value: torch.Tensor, tokens: int, channels: int) -> None:
    if value.ndim != 3 or tuple(value.shape[1:]) != (tokens, channels):
        raise ValueError(f"Expected posterior means shaped (N, {tokens}, {channels}); got {tuple(value.shape)}")


def build_clip_mapper(config: dict) -> nn.Module:
    options = dict(config)
    kind = options.pop("type")
    if kind == "meanbridge":
        return MeanBridge(**options)
    if kind == "tokenformer":
        return TokenFormer(**options)
    if kind == "tokenformer_dual":
        return TokenFormerDualHead(**options)
    if kind == "adatoken":
        return AdaToken(**options)
    if kind == "adatoken_refinement":
        from src.models.adatoken_refinement import AdaTokenRefinement
        return AdaTokenRefinement(**options)
    raise ValueError(f"Unknown CLIP mapping head: {kind}")
