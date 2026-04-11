import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False

try:
    import torchvision.models as tvm
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class CombinedL2SSIMLoss(nn.Module):
    def __init__(self, alpha_l2: float = 0.7) -> None:
        super().__init__()
        self.alpha_l2 = alpha_l2
        self.l2 = nn.MSELoss()

    def _ssim_fallback(self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        c1 = 0.01**2
        c2 = 0.03**2
        mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
        mu_y = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)
        sigma_x = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size // 2) - mu_x * mu_x
        sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size // 2) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2))
        return ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l2_loss = self.l2(pred, target)
        ssim_score = ssim(pred, target, data_range=1.0, size_average=True) if HAS_MSSSIM else self._ssim_fallback(pred, target)
        return self.alpha_l2 * l2_loss + (1 - self.alpha_l2) * (1 - ssim_score)


class SobelEdgeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _edges(self, img: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(img, self.sobel_x.to(img.device), padding=1)
        gy = F.conv2d(img, self.sobel_y.to(img.device), padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._edges(pred), self._edges(target))


class LaplacianLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("laplacian", laplacian)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kernel = self.laplacian.to(pred.device)
        return F.l1_loss(F.conv2d(pred, kernel, padding=1), F.conv2d(target, kernel, padding=1))


class ForegroundEdgeLoss(nn.Module):
    def __init__(
        self,
        alpha_l1: float = 0.25,
        alpha_sobel: float = 0.35,
        alpha_laplacian: float = 0.25,
        alpha_ssim: float = 0.15,
        fg_threshold: float = 0.05,
        fg_steepness: float = 5.0,
    ) -> None:
        super().__init__()
        total = alpha_l1 + alpha_sobel + alpha_laplacian + alpha_ssim
        self.alpha_l1 = alpha_l1 / total
        self.alpha_sobel = alpha_sobel / total
        self.alpha_laplacian = alpha_laplacian / total
        self.alpha_ssim = alpha_ssim / total
        self.fg_threshold = fg_threshold
        self.fg_steepness = fg_steepness
        self.sobel = SobelEdgeLoss()
        self.laplacian = LaplacianLoss()

    def _mask(self, target: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fg_steepness * (target - self.fg_threshold))

    def _ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if HAS_MSSSIM:
            return 1 - ssim(pred, target, data_range=1.0, size_average=True)
        c1 = 0.01**2
        c2 = 0.03**2
        mu_x = F.avg_pool2d(pred, 11, stride=1, padding=5)
        mu_y = F.avg_pool2d(target, 11, stride=1, padding=5)
        sigma_x = F.avg_pool2d(pred * pred, 11, stride=1, padding=5) - mu_x * mu_x
        sigma_y = F.avg_pool2d(target * target, 11, stride=1, padding=5) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(pred * target, 11, stride=1, padding=5) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2))
        return 1 - ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = self._mask(target)
        l1_masked = (torch.abs(pred - target) * mask).sum() / (mask.sum() + 1e-8)
        sobel_loss = self.sobel(pred, target)
        laplacian_loss = self.laplacian(pred, target)
        ssim_loss = self._ssim_loss(pred, target)
        return (
            self.alpha_l1 * l1_masked
            + self.alpha_sobel * sobel_loss
            + self.alpha_laplacian * laplacian_loss
            + self.alpha_ssim * ssim_loss
        )


class PerceptualLoss(nn.Module):
    """
    VGG19 perceptual loss using relu2_2 and relu3_4 feature maps.

    Encourages the SR output to match the HR target in feature space,
    recovering mid-frequency texture that pixel-level losses average away.
    Grayscale inputs are duplicated to 3-channel RGB before passing through
    VGG and normalised to ImageNet statistics.
    """

    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self) -> None:
        super().__init__()
        if not HAS_TORCHVISION:
            raise ImportError("torchvision is required for PerceptualLoss. Install with: pip install torchvision")

        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1).features
        # relu2_2 ends at index 9; relu3_4 ends at index 18
        self.slice1 = nn.Sequential(*list(vgg)[:10]).eval()
        self.slice2 = nn.Sequential(*list(vgg)[10:19]).eval()
        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self._STD).view(1, 3, 1, 1))

    def _to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W) in [0, 1] → (B, 3, H, W) normalised to ImageNet stats
        return (x.repeat(1, 3, 1, 1) - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_rgb = self._to_rgb(pred)
        target_rgb = self._to_rgb(target)
        # slice2 continues from slice1's output, so they must be chained
        pred_f1 = self.slice1(pred_rgb)
        tgt_f1 = self.slice1(target_rgb)
        loss = F.l1_loss(pred_f1, tgt_f1)
        loss = loss + F.l1_loss(self.slice2(pred_f1), self.slice2(tgt_f1))
        return loss


class KSpaceConsistencyLoss(nn.Module):
    """
    Enforces low-frequency consistency between the SR output and LR input.

    Computes the FFT of each image, locates the DC peak (which can be
    off-centre in MRI data due to phase offsets), and compares normalised
    32×32 patches centred on that peak. Normalisation makes the comparison
    scale-invariant across different image sizes (e.g., 128×128 LR vs
    1024×1024 SR). Real and imaginary parts are compared separately so the
    loss is differentiable end-to-end.
    """

    def __init__(self, region_size: int = 32) -> None:
        super().__init__()
        self.half = region_size // 2

    def _peak_region(self, image: torch.Tensor) -> torch.Tensor:
        kspace = torch.fft.fftshift(torch.fft.fft2(image))  # (B, 1, H, W) complex
        mag = kspace.abs()
        B, _, H, W = mag.shape
        half = self.half
        regions = []
        for b in range(B):
            flat_idx = int(mag[b, 0].argmax())
            cy, cx = flat_idx // W, flat_idx % W
            # Clamp so the 32×32 window never falls outside the image
            y0 = min(max(cy - half, 0), H - 2 * half)
            x0 = min(max(cx - half, 0), W - 2 * half)
            patch = kspace[b, 0, y0:y0 + 2 * half, x0:x0 + 2 * half]
            # Normalise by peak magnitude so comparison is scale-invariant
            patch = patch / (patch.abs().max() + 1e-8)
            regions.append(patch)
        return torch.stack(regions, dim=0)  # (B, region_size, region_size) complex

    def forward(self, sr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        sr_region = self._peak_region(sr)
        lr_region = self._peak_region(lr)
        return F.l1_loss(sr_region.real, lr_region.real) + F.l1_loss(sr_region.imag, lr_region.imag)


class ForegroundEdgePerceptualLoss(nn.Module):
    """
    Extends ForegroundEdgeLoss with a VGG19 perceptual term.

    The perceptual component recovers mid-frequency texture that L1, SSIM,
    and hand-crafted edge terms tend to average away — the primary cause of
    the over-smoothing observed in the Proposed model. Enable via
    --perceptual-loss in train.py; the pixel-level weights are unchanged.

    max_percep_size: if the image is larger than this, it is downsampled before
    the VGG pass only. This keeps validation fast when full 1024×1024 images are
    used without affecting training patches (which are typically ≤512×512).
    """

    def __init__(
        self,
        alpha_l1: float = 0.25,
        alpha_sobel: float = 0.35,
        alpha_laplacian: float = 0.25,
        alpha_ssim: float = 0.15,
        alpha_percep: float = 0.1,
        fg_threshold: float = 0.05,
        fg_steepness: float = 5.0,
        max_percep_size: int = 512,
    ) -> None:
        super().__init__()
        self.alpha_percep = alpha_percep
        self.max_percep_size = max_percep_size
        self.pixel_loss = ForegroundEdgeLoss(
            alpha_l1=alpha_l1,
            alpha_sobel=alpha_sobel,
            alpha_laplacian=alpha_laplacian,
            alpha_ssim=alpha_ssim,
            fg_threshold=fg_threshold,
            fg_steepness=fg_steepness,
        )
        self.perceptual = PerceptualLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p, t = pred, target
        if p.shape[-1] > self.max_percep_size:
            p = F.interpolate(p, size=self.max_percep_size, mode="bilinear", align_corners=False)
            t = F.interpolate(t, size=self.max_percep_size, mode="bilinear", align_corners=False)
        return self.pixel_loss(pred, target) + self.alpha_percep * self.perceptual(p, t)
