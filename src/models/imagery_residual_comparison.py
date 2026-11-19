"""Voxel adapters initialized exactly as the identity mapping."""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                 nn.GELU(), nn.Linear(width, width))
    def forward(self, x):
        return x + self.net(x)


class ImageryResidualAdapter(nn.Module):
    """A linear or a two-block residual MLP voxel correction."""
    def __init__(self, voxels, kind, width=512):
        super().__init__()
        self.kind = kind
        if kind == 'linear':
            self.correction = nn.Linear(voxels, voxels, bias=True)
            nn.init.zeros_(self.correction.weight)
            nn.init.zeros_(self.correction.bias)
        elif kind == 'mlp':
            self.correction = nn.Sequential(nn.Linear(voxels,width), nn.GELU(),
                    ResidualBlock(width), ResidualBlock(width), nn.LayerNorm(width), nn.Linear(width,voxels))
            nn.init.zeros_(self.correction[-1].weight)
            nn.init.zeros_(self.correction[-1].bias)
        else: raise ValueError('Unsupported adapter kind')
    def forward(self,x):
        if x.ndim != 2: raise ValueError('Expected batch x voxels')
        return x + self.correction(x)
