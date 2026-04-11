"""
ProposedModelV2 — April 2026.

Improvements over ProposedModelV1:
  - Depthwise smooth conv after each PixelShuffle stage eliminates checkerboard
    artifacts that ICNR init alone cannot prevent during training.
  - Sigmoid output removed: it clips gradients at tissue boundaries (values near
    0 and 1), which is where the diagnostically relevant signal lives.
  - in_channels parameter: supports temporal stacking of 3 adjacent frames as a
    3-channel input for temporally consistent dynamic inference.
  - adapt_checkpoint_to_temporal(): converts a V1/V2 single-channel checkpoint to
    3-channel by copying weights to the centre channel and zeroing the flanking
    channels, so the model can be fine-tuned with temporal context.
"""

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
        scale = self.excitation(self.squeeze(x))
        return x * scale


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
        out = self.se(self.body(x))
        return x + self.res_scale * out


class PixelShuffleBlock(nn.Module):
    def __init__(self, n_feats: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        # Depthwise conv after shuffle: mixes the sub-pixel channels together,
        # breaking the periodic grid pattern that causes checkerboard artifacts.
        # Uses groups=n_feats so it adds very few parameters (~9 per channel).
        self.smooth = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, groups=n_feats)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.smooth(self.shuffle(self.conv(x))))


class ProposedModelV2(nn.Module):
    """
    8× MRI super-resolution model: 3 PixelShuffle stages (128 → 256 → 512 → 1024).

    in_channels=1  for standard single-frame inference.
    in_channels=3  for temporal inference (frames t-1, t, t+1 stacked as channels).
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_res_blocks: int = 16,
        n_feats: int = 64,
        reduction: int = 16,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, n_feats, kernel_size=3, padding=1)
        body = [AttentiveResBlock(n_feats, reduction=reduction, res_scale=res_scale) for _ in range(n_res_blocks)]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)
        self.up1 = PixelShuffleBlock(n_feats)
        self.up2 = PixelShuffleBlock(n_feats)
        self.up3 = PixelShuffleBlock(n_feats)
        # Sigmoid removed: it clips gradients near tissue boundaries (values
        # close to 0 and 1), which is where most diagnostic detail lives.
        # The loss functions already constrain the output range via L1/SSIM.
        self.output_conv = nn.Conv2d(n_feats, 1, kernel_size=3, padding=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    init.constant_(module.bias, 0)
        with torch.no_grad():
            for block in (self.up1, self.up2, self.up3):
                block.conv.weight.copy_(icnr_init(block.conv.weight, upscale=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head = self.head(x)
        body = self.body(head) + head
        out = self.up1(body)
        out = self.up2(out)
        out = self.up3(out)
        return self.output_conv(out)


def build_proposed2(
    n_res_blocks: int = 16,
    n_feats: int = 64,
    reduction: int = 16,
    res_scale: float = 0.1,
    in_channels: int = 1,
) -> ProposedModelV2:
    return ProposedModelV2(
        in_channels=in_channels,
        n_res_blocks=n_res_blocks,
        n_feats=n_feats,
        reduction=reduction,
        res_scale=res_scale,
    )


def adapt_checkpoint_to_temporal(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Expand a single-channel head weight to accept 3 temporal input channels.

    The original weights are copied to the centre channel (frame t). Flanking
    channels (t-1 and t+1) are zeroed so the model initially relies on the
    current frame and gradually learns to use temporal context during fine-tuning.
    """
    new_state = dict(state_dict)
    head_w = state_dict["head.weight"]  # (n_feats, 1, 3, 3)
    new_head = torch.zeros(head_w.shape[0], 3, *head_w.shape[2:], dtype=head_w.dtype)
    new_head[:, 1:2, :, :] = head_w  # channel 1 = current frame t
    new_state["head.weight"] = new_head
    return new_state


# Backwards-compatible aliases so existing scripts importing 'build_proposed' or
# 'ProposedModel' from this module continue to get V2.
ProposedModel = ProposedModelV2
build_proposed = build_proposed2
