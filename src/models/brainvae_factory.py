"""Select a VAE implementation without changing existing checkpoint defaults."""
from __future__ import annotations

from src.models.brainvae import BrainVAE


def build_brainvae(config: dict):
    options = dict(config)
    architecture = options.pop("architecture", "project")
    if architecture == "project":
        return BrainVAE(**options)
    if architecture == "synbrain":
        from src.models.brainvae_synbrain import SynBrainVAE
        return SynBrainVAE(**options)
    if architecture == "neuroflow":
        from src.models.neuroflow_neurovae import NeuroFlowNeuroVAE
        return NeuroFlowNeuroVAE(**options)
    raise ValueError(f"Unknown BrainVAE architecture: {architecture}")
