"""Select a VAE implementation without changing existing checkpoint defaults."""
from __future__ import annotations

from src.models.brainvae import BrainVAE
from src.utils.config import brainvae_architecture


def build_brainvae(config: dict):
    options = dict(config)
    architecture = brainvae_architecture(options)
    options.pop("architecture", None)
    if architecture == "project":
        return BrainVAE(**options)
    if architecture == "ppbe":
        from src.models.brainvae_ppbe import PerceptualBrainVAE
        return PerceptualBrainVAE(**options)
    if architecture == "neuroflow":
        from src.models.neuroflow_neurovae import NeuroFlowNeuroVAE
        return NeuroFlowNeuroVAE(**options)
    raise ValueError(f"Unknown BrainVAE architecture: {architecture}")
