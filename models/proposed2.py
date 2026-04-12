"""
ProposedModelV2 — April 2026.

Architecture improvements:
  - PixelShuffleBlock now uses a full cross-channel 3×3 smooth conv (NOT depthwise)
    after each shuffle stage, mixing the four sub-pixel channels and breaking the
    2×2 periodic grid that ICNR init alone cannot remove.  A parallel antialiased
    bilinear branch is added so the network has a structurally smooth reference to
    fall back on, preventing residual checkerboard from dominating.
  - scale parameter controls the number of PixelShuffle stages:
      scale=4  (default) → 2 stages → 128 → 256 → 512 px  (4× SR)
      scale=8             → 3 stages → 128 → 256 → 512 → 1024 px  (8× SR)
  - in_channels=3 by default for temporal stacking (t-1, t, t+1); set 1 for
    single-frame inference / baselines.
  - Sigmoid output removed: it clips gradients near tissue boundaries.
"""

import math

import torch
import torch.nn as nn
import torch.nn.init as init


def icnr_init(weight: torch.Tensor, upscale: int = 2) -> torch.Tensor:
    c_out, c_in, kh, kw = weight.shape
    base_c_out = c_out // (upscale**2)
    subkernel = torch.empty(base_c_out, c_in, kh, kw)
    init.kaiming_normal_(subkernel, mode="fan_out", nonlinearity="relu")
    return subkernel.repeat_interleave(upscale**2, dim=0)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.excitation(self.squeeze(x))


class AttentiveResBlock(nn.Module):
    def __init__(self, n_feats: int = 64, reduction: int = 16, res_scale: float = 0.1) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
        )
        self.se = SEBlock(n_feats, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.se(self.body(x))


class PixelShuffleBlock(nn.Module):
    def __init__(self, n_feats: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        # Full 3×3 conv (NOT depthwise) — mixes all four sub-pixel channels,
        # breaking the 2×2 periodic grid that ICNR alone cannot fully remove.
        self.smooth = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        # Parallel antialiased bilinear branch — provides a structurally smooth
        # reference so residual checkerboard artefacts cannot dominate.
        self.aa_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.aa_conv = nn.Conv2d(n_feats, n_feats, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ps = self.act(self.smooth(self.shuffle(self.conv(x))))
        aa = self.aa_conv(self.aa_up(x))
        return ps + aa


class ProposedModelV2(nn.Module):
    """
    MRI super-resolution with AttentiveResBlocks and PixelShuffle upscaling.

    scale=4 (default) → 2 stages → 128→512 px (4× SR, matches real HR data).
    scale=8            → 3 stages → 128→1024 px (8× SR, legacy configuration).

    in_channels=3 (default) — expects temporal stack (t-1, t, t+1).
    in_channels=1           — single-frame; use for baselines.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_res_blocks: int = 16,
        n_feats: int = 64,
        reduction: int = 16,
        res_scale: float = 0.1,
        scale: int = 4,
    ) -> None:
        super().__init__()
        n_stages = int(math.log2(scale))
        if 2 ** n_stages != scale or n_stages < 1:
            raise ValueError(f"scale must be a power of 2 ≥ 2, got {scale}")

        self.head = nn.Conv2d(in_channels, n_feats, kernel_size=3, padding=1)
        body = [AttentiveResBlock(n_feats, reduction=reduction, res_scale=res_scale) for _ in range(n_res_blocks)]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)
        self.up_stages = nn.ModuleList([PixelShuffleBlock(n_feats) for _ in range(n_stages)])
        self.output_conv = nn.Conv2d(n_feats, 1, kernel_size=3, padding=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    init.constant_(module.bias, 0)
        with torch.no_grad():
            for block in self.up_stages:
                block.conv.weight.copy_(icnr_init(block.conv.weight, upscale=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head = self.head(x)
        out = self.body(head) + head
        for stage in self.up_stages:
            out = stage(out)
        return self.output_conv(out)


def build_proposed2(
    n_res_blocks: int = 16,
    n_feats: int = 64,
    reduction: int = 16,
    res_scale: float = 0.1,
    in_channels: int = 3,
    scale: int = 4,
) -> ProposedModelV2:
    return ProposedModelV2(
        in_channels=in_channels,
        n_res_blocks=n_res_blocks,
        n_feats=n_feats,
        reduction=reduction,
        res_scale=res_scale,
        scale=scale,
    )


def adapt_checkpoint_to_temporal(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Expand a single-channel head weight to 3-channel for temporal input.

    Deprecated: new checkpoints are always built with in_channels=3. This helper
    remains for loading legacy single-channel checkpoints.
    """
    new_state = dict(state_dict)
    head_w = state_dict["head.weight"]  # (n_feats, 1, 3, 3)
    new_head = torch.zeros(head_w.shape[0], 3, *head_w.shape[2:], dtype=head_w.dtype)
    new_head[:, 1:2, :, :] = head_w  # centre channel = current frame t
    new_state["head.weight"] = new_head
    return new_state


# Backwards-compatible aliases
ProposedModel = ProposedModelV2
build_proposed = build_proposed2
