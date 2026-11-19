"""Strict loading of upstream BrainVAE checkpoints, with no key dropping."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from src.models.brainvae_factory import build_brainvae
from src.utils.config import load_yaml


def read_synbrain_checkpoint(path: Path) -> tuple[dict, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("BrainVAE checkpoint must be a state dictionary or a dictionary containing model")
    raw = next((payload[key] for key in ("model", "state_dict", "model_state_dict")
                if key in payload and isinstance(payload[key], Mapping)), payload)
    weights = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("BrainVAE state dictionary contains non-tensor entries")
        normalized = key
        while normalized.startswith("module."):
            normalized = normalized[len("module."):]
        if normalized in weights:
            raise ValueError(f"Duplicate checkpoint key after removing module prefix: {normalized}")
        weights[normalized] = value
    for key in ("encoder.conv_in.weight", "pre_projector_mean.0.weight",
                "pre_projector_mean.2.weight", "pre_projector_mean.8.weight"):
        if key not in weights:
            raise ValueError(f"Not an upstream BrainVAE checkpoint: missing {key}; "
                             "select Stage1 BrainVAE, rather than the Stage2 S2N mapper")
    metadata = {key: payload[key] for key in ("epoch", "subject") if key in payload}
    return weights, metadata


def checkpoint_dimensions(weights: dict) -> dict:
    conv = weights["encoder.conv_out.weight"]
    mean_input = weights["pre_projector_mean.0.weight"]
    mean_middle = weights["pre_projector_mean.2.weight"]
    mean_output = weights["pre_projector_mean.8.weight"]
    attention = "vanilla" if "encoder.mid.attn_1.q.weight" in weights else (
        "linear" if "encoder.mid.attn_1.to_qkv.weight" in weights else "none")
    return {"hidden_dim": int(mean_input.shape[0]), "linear_dim": int(mean_middle.shape[0]),
            "embed_dim": int(mean_output.shape[0]), "latent_tokens": int(conv.shape[0]),
            "hidden_channels": int(weights["encoder.conv_in.weight"].shape[0]),
            "attention": attention}


def external_model_config(weights: dict, default_config: dict, author_config: Path | None = None) -> dict:
    config = {**default_config, "ddconfig": dict(default_config["ddconfig"])}
    if config.get("architecture") != "synbrain":
        raise ValueError("External SynBrain checkpoints require architecture=synbrain")
    dimensions = checkpoint_dimensions(weights)
    if author_config is not None:
        author = load_yaml(author_config)
        model = author.get("model", author)
        params = model.get("params", model)
        config["ddconfig"] = dict(params.get("ddconfig", params))
    for key in ("hidden_dim", "linear_dim", "embed_dim", "latent_tokens", "hidden_channels"):
        config[key] = dimensions[key]
    config["activation_checkpointing"] = False
    # Do not guess pool length/downsampling from parameter dimensions: both
    # affect the forward operator but are not encoded in the state dictionary.
    return config


def validate_synbrain_weights(weights: dict, model_config: dict) -> None:
    try:
        with torch.device("meta"):
            expected = build_brainvae(model_config).state_dict()
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(f"Author checkpoint needs its matching ddconfig: {error}. "
                         "Set external_source.author_model_config to the author's YAML; "
                         "no pooling/downsampling configuration is guessed.") from error
    missing, unexpected = sorted(expected.keys() - weights.keys()), sorted(weights.keys() - expected.keys())
    mismatched = [f"{key}: {tuple(weights[key].shape)} != {tuple(expected[key].shape)}"
                  for key in expected.keys() & weights.keys() if weights[key].shape != expected[key].shape]
    if missing or unexpected or mismatched:
        raise ValueError(f"Strict SynBrain compatibility failed; missing={missing[:8]}, "
                         f"unexpected={unexpected[:8]}, shapes={mismatched[:8]}. "
                         "Use the original model/config; incompatible layers are never skipped.")
    for key, value in weights.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite pretrained weights: {key}")
