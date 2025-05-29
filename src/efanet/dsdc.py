# src/efanet/dsdc.py
# ------------------------------------------------------------------
#  Dual-Scale Depthwise-Separable Convolution (DSDC)
#  Matches §3.2 of the EFANet paper.
# ------------------------------------------------------------------

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple

from .layers import conv1x1, _kaiming_init, channel_shuffle


def _dwconv_bn_relu(channels: int, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        ),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
    )


class DSDC(nn.Module):
    """
    Dual-scale depthwise-separable convolution block.

    Two parallel DWConv branches capture complementary receptive fields
    (dilation 1 and 2 by default).  Outputs are summed, shuffled across
    channel groups, and projected back with a 1×1 convolution.

    Parameters
    ----------
    channels : int
        Input/output channel count.
    dilations : Tuple[int, int], default=(1, 2)
        Dilation factors for the two branches.
    shuffle_groups : int, default=2
        Groups used by ShuffleNet-style channel shuffle.
    """

    def __init__(
        self,
        channels: int,
        dilations: Tuple[int, int] = (1, 2),
        shuffle_groups: int = 2,
    ) -> None:
        super().__init__()

        d1, d2 = dilations
        self.b1 = _dwconv_bn_relu(channels, dilation=d1)
        self.b2 = _dwconv_bn_relu(channels, dilation=d2)

        self.post = nn.Sequential(
            nn.BatchNorm2d(channels),
            conv1x1(channels, channels),
        )
        self.shuffle_groups = shuffle_groups

        self.apply(_kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        y = self.b1(x) + self.b2(x)          # element-wise sum
        y = channel_shuffle(y, self.shuffle_groups)
        y = self.post(y)
        return y
