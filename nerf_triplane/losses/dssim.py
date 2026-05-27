"""Differential SSIM (D-SSIM) loss for patch-based rendering.

Stage-B main training in the architecture diagram lists "L1 / D-SSIM +
LPIPS(face & mouth) ..." as the photometric stack. This module supplies
the D-SSIM term; L1 and LPIPS already exist in the trainer.

Inputs/outputs are normalized RGB patches in [0, 1] with shape [B, 3, H, W].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    if img1.shape != img2.shape:
        raise ValueError(f"D-SSIM input shape mismatch: {img1.shape} vs {img2.shape}")
    if img1.dim() != 4:
        raise ValueError(f"D-SSIM expects [B, C, H, W], got {img1.shape}")

    C = img1.size(1)
    g1d = _gaussian_kernel(window_size, sigma, img1.device, img1.dtype)
    window = (g1d[:, None] @ g1d[None, :])[None, None, ...].expand(C, 1, window_size, window_size)

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


class DSSIMLoss(nn.Module):
    """D-SSIM = (1 - SSIM) / 2, scaled by `lambda_dssim`."""

    def __init__(self, lambda_dssim: float = 0.0, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.lambda_dssim = float(lambda_dssim)
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.lambda_dssim == 0.0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        ssim_val = _ssim(pred, target, self.window_size, self.sigma, data_range=1.0)
        return self.lambda_dssim * 0.5 * (1.0 - ssim_val)
