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
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.conv(x)))


class ProposedModel(nn.Module):

    def __init__(
        self,
        channels: int = 1,
        n_res_blocks: int = 16,
        n_feats: int = 64,
        reduction: int = 16,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(channels, n_feats, kernel_size=3, padding=1)
        body = [AttentiveResBlock(n_feats, reduction=reduction, res_scale=res_scale) for _ in range(n_res_blocks)]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)
        self.up1 = PixelShuffleBlock(n_feats)
        self.up2 = PixelShuffleBlock(n_feats)
        self.up3 = PixelShuffleBlock(n_feats)
        self.output_conv = nn.Conv2d(n_feats, channels, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()
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
        return self.output_act(self.output_conv(out))


def build_proposed(
    n_res_blocks: int = 16,
    n_feats: int = 64,
    reduction: int = 16,
    res_scale: float = 0.1,
) -> ProposedModel:
    return ProposedModel(
        channels=1,
        n_res_blocks=n_res_blocks,
        n_feats=n_feats,
        reduction=reduction,
        res_scale=res_scale,
    )
