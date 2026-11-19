"""Opt-in refinements of a pretrained AdaToken; legacy AdaToken stays unchanged."""
import math

import torch
from torch import nn

from src.models.clip_mapping import AdaToken, _check_shape


class AdaTokenRefinement(AdaToken):
    def __init__(self, *args, sample_condition=False, condition_width=128,
                 fusion="none", alpha_init=0.5, gate_width=64,
                 train_backbone=True, **kwargs):
        super().__init__(*args, **kwargs)
        if fusion not in {"none", "fixed", "learned"}:
            raise ValueError("fusion must be none, fixed or learned")
        if not math.isfinite(alpha_init) or not 0 <= alpha_init <= 1:
            raise ValueError("alpha_init must be in [0, 1]")
        if fusion == "learned" and not 0 < alpha_init < 1:
            raise ValueError("Learned fusion requires 0 < alpha_init < 1")
        if condition_width <= 0 or gate_width <= 0:
            raise ValueError("Adapter widths must be positive")
        self.fusion, self.alpha_init = fusion, alpha_init
        self.sample_condition = sample_condition
        self.sit.requires_grad_(train_backbone)
        # Position embeddings always remain fixed, including when unfreezing SiT.
        self.sit.pos_embed.requires_grad_(False)
        if sample_condition:
            self.condition_adapter = nn.Sequential(
                nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, condition_width),
                nn.SiLU(), nn.Linear(condition_width, self.embed_dim))
            nn.init.zeros_(self.condition_adapter[-1].weight)
            nn.init.zeros_(self.condition_adapter[-1].bias)
        if fusion == "learned":
            self.fusion_gate = nn.Sequential(
                nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, gate_width),
                nn.SiLU(), nn.Linear(gate_width, 1))
            nn.init.zeros_(self.fusion_gate[-1].weight)
            nn.init.constant_(self.fusion_gate[-1].bias, math.log(alpha_init / (1-alpha_init)))

    def mixing_weight(self, mean):
        if self.fusion == "learned":
            return self.fusion_gate(mean.mean(dim=1)).sigmoid().unsqueeze(-1)
        return mean.new_full((len(mean), 1, 1), self.alpha_init)

    def forward(self, mean):
        _check_shape(mean, self.latent_tokens, self.embed_dim)
        if not self.sample_condition:
            prediction = super().forward(mean)
        else:
            hidden = mean + self.sit.pos_embed
            condition = self.sit.t_embedder(mean.new_zeros(len(mean)))
            condition = condition + self.condition_adapter(mean.mean(dim=1))
            for block in self.sit.blocks:
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint
                    hidden = checkpoint(block, hidden, condition, use_reentrant=False)
                else:
                    hidden = block(hidden, condition)
            prediction = self.sit.final_layer(hidden, condition)
        if self.fusion == "none":
            return prediction
        alpha = self.mixing_weight(mean)
        return (1-alpha) * mean + alpha * prediction
