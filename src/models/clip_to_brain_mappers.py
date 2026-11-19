"""Comparable token-preserving CLIP-Vision to BrainVAE-mean mappers."""
import torch
from torch import nn
from src.models.s2n_mapper import S2NMapper


class TokenMapper(nn.Module):
    """Shared across the token positions; no positional mixing."""
    def __init__(self, kind, embed_dim=1664, hidden_dim=2048):
        super().__init__()
        if kind == 'linear':
            self.net = nn.Linear(embed_dim, embed_dim)
        elif kind == 'mlp':
            self.net = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, embed_dim))
        else:
            raise ValueError(f'Unsupported mapper: {kind}')

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError('Expected (batch, tokens, channels)')
        return self.net(x)


def build_mapper(kind, tokens=256, dim=1664, mlp_hidden=2048,
                 s2n_depth=8, s2n_heads=13, s2n_ratio=4.):
    if kind == 's2n':
        return S2NMapper(latent_tokens=tokens, embed_dim=dim, depth=s2n_depth,
                         num_heads=s2n_heads, mlp_ratio=s2n_ratio)
    return TokenMapper(kind, embed_dim=dim, hidden_dim=mlp_hidden)
