# src/efanet/layers.py
# ------------------------------------------------------------------
#  Building blocks shared by BFSA, DSDC, CCFFM and the main EFANet
# ------------------------------------------------------------------

from __future__ import annotations
import math
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Helper: weight initialisation à la Kaiming
# ------------------------------------------------------------------
def _kaiming_init(module: nn.Module, a: float = 0, mode: str = "fan_out") -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ------------------------------------------------------------------
# Basic conv helpers
# ------------------------------------------------------------------
def conv3x3(in_ch: int, out_ch: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False)


# ------------------------------------------------------------------
# Channel Shuffle (ShuffleNet-style)
# ------------------------------------------------------------------
def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    b, c, h, w = x.size()
    if c % groups != 0:  # pragma: no cover
        raise ValueError(f"Channels ({c}) not divisible by groups ({groups})")
    x = x.reshape(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.reshape(b, c, h, w)


# ------------------------------------------------------------------
# Depthwise-Separable Conv block (DWConv + BN + ReLU)
# ------------------------------------------------------------------
class DWConvBNReLU(nn.Sequential):
    """
    3×3 depthwise convolution followed by BN + ReLU.
    """

    def __init__(self, channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.apply(_kaiming_init)


# ------------------------------------------------------------------
# Dual-Scale Depthwise-Separable Convolution (DSDC)
# ------------------------------------------------------------------
class DSDC(nn.Module):
    """
    Two parallel depthwise-separable conv branches with channel shuffle.
    Used in EFANet paper to enlarge receptive fields without extra params.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branch1 = DWConvBNReLU(channels)
        self.branch2 = DWConvBNReLU(channels)

        self.proj = nn.Sequential(
            nn.BatchNorm2d(channels),
            conv1x1(channels, channels),
        )
        self.apply(_kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y = y1 + y2
        y = channel_shuffle(y, groups=2)
        return self.proj(y)


# ------------------------------------------------------------------
# PixelShuffle Upsampler (X2, X4, X8)
# ------------------------------------------------------------------
class UpsamplePixelShuffle(nn.Sequential):
    """
    Flexible pixel-shuffle upsampler. `scale` must be power of two.
    """

    def __init__(self, in_ch: int, scale: int) -> None:
        super().__init__()
        if scale & (scale - 1) != 0:  # pragma: no cover
            raise ValueError("Scale must be power-of-two.")
        num_steps = int(math.log2(scale))
        layers = []
        ch = in_ch
        for _ in range(num_steps):
            layers.append(conv3x3(ch, ch * 4))
            layers.append(nn.PixelShuffle(2))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.apply(_kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.layers(x)


# ------------------------------------------------------------------
# Residual wrapper with optional squeeze-and-excitation
# ------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden_ch: Optional[int] = None,
        use_se: bool = False,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        hidden_ch = hidden_ch or in_ch
        self.conv1 = conv3x3(in_ch, hidden_ch)
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(hidden_ch, in_ch)
        self.bn2 = nn.BatchNorm2d(in_ch)

        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch, in_ch // reduction, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch // reduction, in_ch, 1, bias=True),
                nn.Sigmoid(),
            )

        self.apply(_kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_se:
            out = out * self.se(out)
        return x + out
