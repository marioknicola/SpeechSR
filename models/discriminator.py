import torch
import torch.nn as nn


def _sn_conv(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    """Spectrally normalised conv + LeakyReLU block."""
    return nn.Sequential(
        nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)),
        nn.LeakyReLU(0.2, inplace=True),
    )


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator with spectral normalisation.

    Produces a spatial grid of real/fake scores rather than a single scalar.
    Each output location covers a receptive field of ~286×286 pixels on a
    1024×1024 input (6 stride-2 layers → 16×16 output grid). The adversarial
    loss is averaged over the grid.

    Spectral normalisation stabilises training without needing a gradient
    penalty term (WGAN-GP), keeping the training loop simple.
    """

    def __init__(self, in_channels: int = 1, base_filters: int = 64) -> None:
        super().__init__()
        b = base_filters
        self.net = nn.Sequential(
            # First layer: no spectral norm (standard practice for patch discriminators)
            nn.Conv2d(in_channels, b, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(b,     b * 2, stride=2),
            _sn_conv(b * 2, b * 4, stride=2),
            _sn_conv(b * 4, b * 4, stride=2),
            _sn_conv(b * 4, b * 8, stride=2),
            _sn_conv(b * 8, b * 8, stride=2),
            nn.utils.spectral_norm(nn.Conv2d(b * 8, 1, kernel_size=4, stride=1, padding=1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_discriminator(base_filters: int = 64) -> PatchDiscriminator:
    return PatchDiscriminator(in_channels=1, base_filters=base_filters)
