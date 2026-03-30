import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False


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
        gx = F.conv2d(img, self.sobel_x, padding=1)
        gy = F.conv2d(img, self.sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._edges(pred), self._edges(target))


class LaplacianLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("laplacian", laplacian)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_lap = F.conv2d(pred, self.laplacian, padding=1)
        target_lap = F.conv2d(target, self.laplacian, padding=1)
        return F.l1_loss(pred_lap, target_lap)


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
