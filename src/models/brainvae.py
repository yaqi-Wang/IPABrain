from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _normalize(in_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    groups = min(num_groups, in_channels)
    while in_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=1e-6, affine=True)


class GaussianLatent2D:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor) -> None:
        self.mean = mean
        self.logvar = logvar.clamp(min=-30.0, max=20.0)

    def sample(self) -> torch.Tensor:
        std = torch.exp(0.5 * self.logvar)
        return self.mean + torch.randn_like(std) * std

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        return 0.5 * (torch.exp(self.logvar) + self.mean.pow(2) - 1.0 - self.logvar)


class Downsample1D(nn.Module):
    def __init__(self, channels: int, with_conv: bool = True) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.conv = nn.Conv1d(channels, channels, kernel_size=7, stride=2, padding=3) if with_conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:
            return self.conv(x)
        return F.avg_pool1d(x, kernel_size=2, stride=2)


class Upsample1D(nn.Module):
    def __init__(self, channels: int, with_conv: bool = True) -> None:
        super().__init__()
        if with_conv:
            self.up = nn.ConvTranspose1d(
                channels,
                channels,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=1,
            )
        else:
            self.up = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.up is not None:
            return self.up(x)
        return F.interpolate(x, scale_factor=2.0, mode="nearest")


class ResnetBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        conv_shortcut: bool = False,
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = _normalize(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = _normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        elif conv_shortcut:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(_swish(self.norm1(x)))
        h = self.conv2(self.dropout(_swish(self.norm2(h))))
        return self.shortcut(x) + h


class AttnBlock1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = _normalize(channels)
        self.q = nn.Conv1d(channels, channels, kernel_size=1)
        self.k = nn.Conv1d(channels, channels, kernel_size=1)
        self.v = nn.Conv1d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h).permute(0, 2, 1)
        k = self.k(h)
        v = self.v(h)
        _, channels, _ = k.shape
        weight = torch.bmm(q, k) * (channels ** -0.5)
        weight = F.softmax(weight, dim=-1)
        h = torch.bmm(v, weight.permute(0, 2, 1))
        return x + self.proj_out(h)


class LinAttnBlock1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.to_qkv = nn.Conv1d(channels, channels * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        k = k.softmax(dim=-1)
        context = torch.einsum("bcn,bdn->bcd", k, v)
        out = torch.einsum("bcd,bcn->bdn", context, q)
        return x + self.to_out(out)


def make_attention(channels: int, attn_type: str) -> nn.Module:
    if attn_type == "none":
        return nn.Identity()
    if attn_type == "linear":
        return LinAttnBlock1D(channels)
    if attn_type == "vanilla":
        return AttnBlock1D(channels)
    raise ValueError(f"Unknown attention type: {attn_type}")


class NeuroEncoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int = 1,
        ch_mult: Sequence[int] = (2, 4, 4),
        ada_length: int,
        num_res_blocks: int,
        num_down_blocks: int,
        attn_resolutions: Sequence[int] = (),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        z_channels: int,
        mid_attn_type: str = "linear",
        block_attn_type: str = "none",
        **_: object,
    ) -> None:
        super().__init__()
        del out_ch
        self.ada_length = ada_length
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.num_down_blocks = num_down_blocks
        self.output_length = ada_length // (2**num_down_blocks)

        self.conv_in = nn.Conv1d(in_channels, ch, kernel_size=7, stride=1, padding=3)
        self.ada_maxpool = nn.AdaptiveMaxPool1d(ada_length)

        curr_len = ada_length
        in_ch_mult = (1, *tuple(ch_mult))
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock1D(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_len in attn_resolutions:
                    attn.append(make_attention(block_in, block_attn_type))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < num_down_blocks:
                down.downsample = Downsample1D(block_in, resamp_with_conv)
                curr_len = curr_len // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(block_in, block_in, dropout=dropout)
        self.mid.attn_1 = make_attention(block_in, mid_attn_type)
        self.mid.block_2 = ResnetBlock1D(block_in, block_in, dropout=dropout)

        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, z_channels, kernel_size=7, stride=1, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ada_maxpool(self.conv_in(x))
        hs = [h]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level < self.num_down_blocks:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = self.mid.block_1(hs[-1])
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        return self.conv_out(_swish(self.norm_out(h)))


class NeuroDecoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: Sequence[int] = (2, 4, 4),
        num_res_blocks: int,
        num_up_blocks: int,
        attn_resolutions: Sequence[int] = (),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        z_channels: int,
        hidden_length: int,
        mid_attn_type: str = "linear",
        block_attn_type: str = "none",
        give_pre_end: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        del in_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.num_up_blocks = num_up_blocks
        self.give_pre_end = give_pre_end

        block_in = ch * ch_mult[-1]
        curr_len = hidden_length
        self.conv_in = nn.Conv1d(z_channels, block_in, kernel_size=7, stride=1, padding=3)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(block_in, block_in, dropout=dropout)
        self.mid.attn_1 = make_attention(block_in, mid_attn_type)
        self.mid.block_2 = ResnetBlock1D(block_in, block_in, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock1D(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_len in attn_resolutions:
                    attn.append(make_attention(block_in, block_attn_type))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level >= self.num_resolutions - num_up_blocks:
                up.upsample = Upsample1D(block_in, resamp_with_conv)
                curr_len = curr_len * 2
            self.up.insert(0, up)

        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, out_ch, kernel_size=7, stride=1, padding=3)

    def forward(self, z: torch.Tensor, target_length: int) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level >= self.num_resolutions - self.num_up_blocks:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = F.interpolate(h, size=target_length, mode="linear", align_corners=False)
        return self.conv_out(_swish(self.norm_out(h)))


def _token_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


def soft_clip_loss(preds: torch.Tensor, targets: torch.Tensor, temp: float = 0.125) -> torch.Tensor:
    preds = F.normalize(preds.flatten(1), dim=-1)
    targets = F.normalize(targets.flatten(1), dim=-1)
    target_sim = (targets @ targets.T) / temp
    pred_sim = (preds @ targets.T) / temp
    loss1 = -(pred_sim.log_softmax(-1) * target_sim.softmax(-1)).sum(-1).mean()
    loss2 = -(pred_sim.T.log_softmax(-1) * target_sim.softmax(-1)).sum(-1).mean()
    return (loss1 + loss2) * 0.5


def _default_ddconfig(
    *,
    latent_tokens: int,
    hidden_channels: int,
    hidden_dim: int,
    num_down_blocks: int = 1,
) -> dict[str, object]:
    return {
        "z_channels": latent_tokens,
        "in_channels": 1,
        "out_ch": 1,
        "ch": hidden_channels,
        "ch_mult": [2, 4, 4],
        "ada_length": hidden_dim * (2**num_down_blocks),
        "num_res_blocks": 2,
        "num_down_blocks": num_down_blocks,
        "num_up_blocks": 1,
        "attn_resolutions": [],
        "dropout": 0.0,
        "resamp_with_conv": True,
        "mid_attn_type": "linear",
        "block_attn_type": "none",
    }


class BrainVAE(nn.Module):
    def __init__(
        self,
        latent_tokens: int = 256,
        embed_dim: int = 1664,
        hidden_channels: int = 128,
        hidden_dim: int | None = None,
        linear_dim: int = 2048,
        kl_weight: float = 0.001,
        clip_weight: float = 1000.0,
        free_bits: float = 0.0,
        ddconfig: dict[str, object] | None = None,
        semantic_source: str = "sample",
        clip_reg_weight: float = 0.0,
        recon_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if semantic_source not in {"sample", "mean"}:
            raise ValueError("semantic_source must be sample (legacy) or mean")
        for name, value in (("clip_reg_weight", clip_reg_weight), ("recon_weight", recon_weight),
                            ("clip_weight", clip_weight), ("kl_weight", kl_weight), ("free_bits", free_bits)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        hidden_dim_for_defaults = 4096 if hidden_dim is None else hidden_dim
        self.ddconfig = (
            _default_ddconfig(
                latent_tokens=latent_tokens,
                hidden_channels=hidden_channels,
                hidden_dim=hidden_dim_for_defaults,
            )
            if ddconfig is None
            else dict(ddconfig)
        )

        self.latent_tokens = int(self.ddconfig.get("z_channels", latent_tokens))
        self.embed_dim = embed_dim
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.free_bits = float(free_bits)
        # Defaults preserve old checkpoints and existing training callers.
        # New mean-supervised experiments enable these explicitly in their config.
        self.semantic_source = semantic_source
        self.clip_reg_weight = float(clip_reg_weight)
        self.recon_weight = float(recon_weight)

        num_down_blocks = int(self.ddconfig.get("num_down_blocks", 0))
        ada_length = int(self.ddconfig["ada_length"])
        inferred_hidden_dim = ada_length // (2**num_down_blocks)
        if hidden_dim is not None and hidden_dim != inferred_hidden_dim:
            raise ValueError(
                "hidden_dim must match ddconfig ada_length // 2**num_down_blocks "
                f"({hidden_dim} != {inferred_hidden_dim})"
            )
        self.hidden_dim = inferred_hidden_dim

        self.encoder = NeuroEncoder(**self.ddconfig)
        decoder_config = dict(self.ddconfig)
        decoder_config["hidden_length"] = self.hidden_dim
        self.decoder = NeuroDecoder(**decoder_config)

        self.pre_projector_mean = _token_mlp(self.hidden_dim, linear_dim, embed_dim)
        self.pre_projector_logvar = _token_mlp(self.hidden_dim, linear_dim, embed_dim)
        self.post_projector = _token_mlp(embed_dim, linear_dim, self.hidden_dim)

    def encode(self, fmri: torch.Tensor) -> GaussianLatent2D:
        if fmri.ndim == 2:
            fmri = fmri.unsqueeze(1)
        hidden = self.encoder(fmri)
        mean = self.pre_projector_mean(hidden)
        logvar = self.pre_projector_logvar(hidden)
        return GaussianLatent2D(mean, logvar)

    def decode(self, latent: torch.Tensor, target_length: int) -> torch.Tensor:
        hidden = self.post_projector(latent)
        return self.decoder(hidden, target_length)

    def forward(
        self,
        fmri: torch.Tensor,
        clip_target: torch.Tensor,
        sample_posterior: bool = True,
        kl_weight_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if fmri.ndim == 2:
            fmri = fmri.unsqueeze(1)
        posterior = self.encode(fmri)
        if clip_target.shape != posterior.mean.shape:
            raise ValueError(f"CLIP target shape {tuple(clip_target.shape)} differs from mean {tuple(posterior.mean.shape)}")
        latent = posterior.sample() if sample_posterior else posterior.mode()
        # Sampling remains the VAE reconstruction branch, independently of
        # which tensor receives image-semantic supervision.
        recon = self.decode(latent, fmri.shape[-1])
        recon_loss = F.mse_loss(recon, fmri, reduction="sum") / fmri.shape[0]
        raw_kl_per_dim = posterior.kl()
        kl_per_dim = raw_kl_per_dim
        if self.free_bits > 0.0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        kl_loss = kl_per_dim.sum() / fmri.shape[0]
        semantic = posterior.mean if self.semantic_source == "mean" else latent
        clip_loss = soft_clip_loss(semantic, clip_target) if self.clip_weight else latent.new_tensor(0.0)
        active_kl_weight = self.kl_weight if kl_weight_override is None else kl_weight_override
        if not math.isfinite(active_kl_weight) or active_kl_weight < 0:
            raise ValueError("Active KL weight must be finite and nonnegative")
        loss = self.recon_weight * recon_loss + active_kl_weight * kl_loss + self.clip_weight * clip_loss
        losses = {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "clip_loss": clip_loss,
        }
        if self.semantic_source == "mean" or self.clip_reg_weight > 0 or self.recon_weight != 1:
            # Raw-coordinate regression: average over batch, patches and channels.
            # Do NOT normalize mean/target or apply the VAE latent scale factor.
            clip_reg_loss = F.mse_loss(posterior.mean, clip_target, reduction="mean")
            losses["loss"] = loss + self.clip_reg_weight * clip_reg_loss
            losses.update({"clip_reg_loss": clip_reg_loss,
                           "weighted_clip_reg_loss": self.clip_reg_weight * clip_reg_loss,
                           "weighted_clip_loss": self.clip_weight * clip_loss,
                           "weighted_recon_loss": self.recon_weight * recon_loss,
                           "weighted_kl_loss": active_kl_weight * kl_loss})
            # Observability only; no extra loss or gradient on variance/norm.
            with torch.no_grad():
                mean, target = posterior.mean.detach().float(), clip_target.detach().float()
                flat_mean, flat_target = mean.flatten(1), target.flatten(1)
                mean_norm, target_norm = flat_mean.norm(dim=1), flat_target.norm(dim=1)
                losses.update({"mean_clip_cosine": F.cosine_similarity(flat_mean, flat_target, dim=-1).mean(),
                               "mean_token_cosine": F.cosine_similarity(mean, target, dim=-1).mean(),
                               "mean_norm": mean_norm.mean(), "target_norm": target_norm.mean(),
                               "mean_norm_ratio": (mean_norm / target_norm.clamp_min(1e-8)).mean(),
                               "logvar_mean": posterior.logvar.detach().float().mean(),
                               "kl_unclamped_loss": raw_kl_per_dim.detach().sum() / fmri.shape[0],
                               "kl_active_fraction": (raw_kl_per_dim.detach() > self.free_bits).float().mean()})
        # Return shape/API unchanged: the first value is still z (or mean when
        # sample_posterior=False); inference continues to use encode(...).mode().
        return latent, recon, losses
