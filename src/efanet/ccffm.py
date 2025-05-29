# src/efanet/ccffm.py
# ------------------------------------------------------------------
#  Cross-Channel Feature Fusion Module (CCFFM)
#  Implements the routing scheme described in §3.3 of the EFANet paper.
# ------------------------------------------------------------------

from __future__ import annotations
import torch
import torch.nn as nn

from .layers import conv1x1, conv3x3, _kaiming_init


class CCFFM(nn.Module):
    """
    Cross-Channel Feature Fusion Module.

    Parameters
    ----------
    channels : int
        Channel count of both shallow and deep feature maps.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # 1×1 conv applied after vertical split (three parallel branches)
        self.fuse_conv = nn.ModuleList([conv1x1(channels * 2, channels) for _ in range(3)])

        # 3×3 conv after cross routing
        self.cross_conv1 = conv3x3(channels * 3, channels)
        self.cross_conv2 = conv3x3(channels * 3, channels)

        self.out_proj = conv1x1(channels * 2, channels)
        self.apply(_kaiming_init)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _split_vertical(x: torch.Tensor) -> list[torch.Tensor]:
        """Split tensor into three parts along the height (top, mid, bottom)."""
        return torch.chunk(x, 3, dim=2)  # dim=2 corresponds to H

    @staticmethod
    def _split_horizontal(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split tensor into two parts along the width (left, right)."""
        return torch.chunk(x, 2, dim=3)  # dim=3 corresponds to W

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, fs: torch.Tensor, fd: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        fs : torch.Tensor (B, C, H, W)
            Shallow feature map.
        fd : torch.Tensor (B, C, H, W)
            Deep feature map.

        Returns
        -------
        torch.Tensor
            Fused feature map of shape (B, C, H, W).
        """
        # 1) vertical split (top / mid / bottom)
        fs_parts = self._split_vertical(fs)
        fd_parts = self._split_vertical(fd)

        branch_feats = []
        for idx in range(3):
            concat = torch.cat([fs_parts[idx], fd_parts[idx]], dim=1)
            branch_feats.append(self.fuse_conv[idx](concat))  # (B, C, H/3, W)

        # 2) horizontal split for each branch
        G1_left, G1_right = self._split_horizontal(branch_feats[0])
        G2_left, G2_right = self._split_horizontal(branch_feats[1])
        G3_left, G3_right = self._split_horizontal(branch_feats[2])

        # 3) cross routing and 3×3 conv
        #    Fhat1: [G1', G2, G3]   (right of B1, left of B2/B3)
        #    Fhat2: [G1,  G2',G3']  (left of  B1, right of B2/B3)
        Fhat1 = self.cross_conv1(torch.cat([G1_right, G2_left, G3_left], dim=1))
        Fhat2 = self.cross_conv2(torch.cat([G1_left, G2_right, G3_right], dim=1))

        # 4) channel concat + projection back to C
        fused = torch.cat([Fhat1, Fhat2], dim=1)  # (B, 2C, H, W)
        return self.out_proj(fused)
