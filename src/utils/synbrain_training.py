"""Opt-in execution improvements for fresh SynBrain training runs."""
from __future__ import annotations

import numpy as np
import torch

from src.models.synbrain_blocks import AttnBlock
from src.models.brainvae_clip_encoder import SDPAAttnBlock


def configure_synbrain_execution(model, training):
    backend = training.get("attention_backend", "original")
    if backend not in {"original", "sdpa"}:
        raise ValueError(f"Unknown SynBrain attention backend: {backend}")
    if backend == "sdpa":
        replacements = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, AttnBlock):
                replacement = SDPAAttnBlock(module.in_channels).to(
                    device=module.q.weight.device, dtype=module.q.weight.dtype)
                replacement.load_state_dict(module.state_dict(), strict=True)
                replacement.train(module.training)
                parent, child = name.rsplit(".", 1)
                setattr(model.get_submodule(parent), child, replacement)
                replacements += 1
        if not replacements:
            raise ValueError("SDPA was requested but no SynBrain attention blocks were found")
    return backend


def preload_shared_arrays(train, val):
    """Store each source image once; keep all individual fMRI repeats."""
    train.fmri = np.array(train.fmri, dtype=np.float32, copy=True)
    train.clip = np.array(train.clip, dtype=np.float32, copy=True)
    val.fmri, val.clip = train.fmri, train.clip
    return train.fmri.nbytes + train.clip.nbytes
