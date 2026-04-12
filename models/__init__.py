from .unet import UNet, build_unet
from .proposed1 import ProposedModelV1, build_proposed1
from .proposed2 import ProposedModelV2, build_proposed2, adapt_checkpoint_to_temporal
from .discriminator import PatchDiscriminator, build_discriminator
from .losses import (
    CombinedL2SSIMLoss,
    ForegroundEdgeLoss,
    ForegroundEdgePerceptualLoss,
    KSpaceConsistencyLoss,
    PerceptualLoss,
    TemporalConsistencyLoss,
)

# 'proposed' always refers to the current default (V2)
ProposedModel = ProposedModelV2
build_proposed = build_proposed2

__all__ = [
    "UNet",
    "build_unet",
    "ProposedModelV1",
    "build_proposed1",
    "ProposedModelV2",
    "build_proposed2",
    "ProposedModel",
    "build_proposed",
    "adapt_checkpoint_to_temporal",
    "PatchDiscriminator",
    "build_discriminator",
    "CombinedL2SSIMLoss",
    "ForegroundEdgeLoss",
    "ForegroundEdgePerceptualLoss",
    "KSpaceConsistencyLoss",
    "PerceptualLoss",
    "TemporalConsistencyLoss",
]
