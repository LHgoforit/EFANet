# evaluation/metrics.py
# ------------------------------------------------------------------
#  Unified PSNR / SSIM / LPIPS computation for EFANet experiments
#  All tensors are assumed to be float32 in [0, 1] and shaped
#  (B, C, H, W) on the same device (CPU or CUDA).
# ------------------------------------------------------------------

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

__all__ = ["psnr", "ssim", "lpips", "MetricMeter"]

# ------------------------------------------------------------- #
#                Peak Signal-to-Noise Ratio                      #
# ------------------------------------------------------------- #
def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    psnr_val = 10.0 * torch.log10(max_val**2 / mse.clamp_min(1e-12))
    return psnr_val  # (B,)


# ------------------------------------------------------------- #
#           Structural Similarity Index (PyTorch)               #
# ------------------------------------------------------------- #
def _gaussian_window(window_size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    return window.expand(3, 1, window_size, window_size).contiguous()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    max_val: float = 1.0,
) -> torch.Tensor:
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    window = _gaussian_window(window_size, sigma, pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=3)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=3)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=3) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=3) - mu12

    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean(dim=[1, 2, 3])  # (B,)


# ------------------------------------------------------------- #
#              Learned Perceptual Image Patch Sim.              #
# ------------------------------------------------------------- #
_lpips_net = None


def _init_lpips(net: str = "alex") -> None:
    global _lpips_net
    if _lpips_net is None:
        import lpips

        _lpips_net = lpips.LPIPS(net=net)
        _lpips_net = _lpips_net.eval().to("cuda" if torch.cuda.is_available() else "cpu")


def lpips(pred: torch.Tensor, target: torch.Tensor, net: str = "alex") -> torch.Tensor:
    _init_lpips(net)
    with torch.no_grad():
        d = _lpips_net(pred * 2 - 1, target * 2 - 1)  # lpips expects [-1, 1]
    return d.view(-1)  # (B,)


# ------------------------------------------------------------- #
#                Metric Accumulator / Meter                     #
# ------------------------------------------------------------- #
class MetricMeter:
    """Running online mean for PSNR / SSIM / LPIPS."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sum: Dict[str, float] = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        self._count: int = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate metrics for a batch."""
        pred = pred.float()
        target = target.float()
        self._sum["psnr"] += psnr(pred, target).sum().item()
        self._sum["ssim"] += ssim(pred, target).sum().item()
        self._sum["lpips"] += lpips(pred, target).sum().item()
        self._count += pred.size(0)

    def value(self) -> Tuple[float, float, float]:
        if self._count == 0:
            return 0.0, 0.0, 0.0
        return (
            self._sum["psnr"] / self._count,
            self._sum["ssim"] / self._count,
            self._sum["lpips"] / self._count,
        )
