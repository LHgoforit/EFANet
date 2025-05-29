# src/efanet/efanet.py
# ------------------------------------------------------------------
#  EFANet full model implementation.
# ------------------------------------------------------------------

from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
from .layers import conv3x3, conv1x1, _kaiming_init, UpsamplePixelShuffle
from .dsdc import DSDC
from .bfsa import BFSA
from .ccffm import CCFFM

class EFAM(nn.Module):
    def __init__(self, channels: int, dsdc_scales: int = 2, bfsa_reduction: int = 4) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.input_proj = conv1x1(channels, channels)
        self.dsdc = DSDC(channels, dilations=(1, 2)[:dsdc_scales])
        self.bfsa = BFSA(channels, reduction=bfsa_reduction)
        self.mlp = nn.Sequential(
            conv1x1(channels, channels),
            nn.GELU(),
            conv1x1(channels, channels),
        )
        self.apply(_kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.input_proj(self.norm(x))
        res = self.dsdc(res) + res
        res = self.bfsa(res)
        return x + self.mlp(res)

class EFANet(nn.Module):
    def __init__(
        self,
        scale: int,
        channels: int = 64,
        num_blocks: int = 13,
        dsdc_scales: int = 2,
        bfsa_reduction: int = 4,
    ) -> None:
        super().__init__()
        if scale not in [4, 8, 16]:
            raise ValueError("Scale must be one of [4, 8, 16]")

        self.shallow = conv3x3(3, channels)
        self.deep_blocks = nn.Sequential(
            *[EFAM(channels, dsdc_scales, bfsa_reduction) for _ in range(num_blocks)]
        )
        self.deep_proj = conv3x3(channels, channels)
        self.fusion = CCFFM(channels)
        self.reconstruction = conv3x3(channels, channels)
        self.upsample = UpsamplePixelShuffle(channels, scale)
        self.output = conv3x3(channels, 3)
        self.apply(_kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow_feat = self.shallow(x)
        deep_feat = self.deep_proj(self.deep_blocks(shallow_feat))
        fused = self.fusion(shallow_feat, deep_feat)
        recon = self.reconstruction(fused)
        up = self.upsample(recon)
        out = self.output(up)
        return torch.clamp(out, 0.0, 1.0)
