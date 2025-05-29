# src/efanet/bfsa.py
# ------------------------------------------------------------------
#  Bilinear Fusion Self-Attention (BFSA) as described in the EFANet
#  paper (§3.2).  Channel-wise attention is computed with *linear*
#  complexity w.r.t. spatial resolution:
#
#      A =  Kᵀ · V   ∈ ℝ^{C×C}
#      W = σ(A₁) + σ(A₂)         (bilinear fusion over two subsets)
#      Y =  Q · Wᵀ
#
#  Output retains the original shape (B, C, H, W) and includes a
#  residual connection plus optional channel-shuffle for extra mixing.
# ------------------------------------------------------------------

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple

from .layers import conv1x1, channel_shuffle, _kaiming_init


class BFSA(nn.Module):
    """
    Bilinear Fusion Self-Attention block.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    reduction : int, default=4
        Internal channel reduction for the Q/K/V 1×1 projections
        (C_r = C // reduction).
    shuffle_groups : int, default=2
        Number of groups used in the post-attention channel shuffle.
    """

    def __init__(self, channels: int, reduction: int = 4, shuffle_groups: int = 2) -> None:
        super().__init__()
        c_red = max(channels // reduction, 1)

        self.proj_qkv = conv1x1(channels, c_red * 3)  # fused Q/K/V
        self.proj_out = conv1x1(c_red, channels)

        self.shuffle_groups = shuffle_groups
        self.sigmoid = nn.Sigmoid()

        self.apply(_kaiming_init)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _split_subsets(
        A: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split correlation matrix along the second dimension into two
        complementary halves (C₁, C₂).
        """
        C = A.size(1)
        mid = C // 2
        return A[:, :mid], A[:, mid:]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        B, C, H, W = x.shape
        # QKV projection ------------------------------------------------
        qkv = self.proj_qkv(x)  # (B, 3C_r, H, W)
        q_red, k_red, v_red = torch.chunk(qkv, 3, dim=1)  # each (B, C_r, H, W)

        # Flatten spatial dims ------------------------------------------
        hw = H * W
        q_r = q_red.reshape(B, -1, hw).permute(0, 2, 1)  # (B, HW, C_r)
        k_r = k_red.reshape(B, -1, hw)                   # (B, C_r, HW)
        v_r = v_red.reshape(B, -1, hw).permute(0, 2, 1)  # (B, HW, C_r)

        # Channel-only correlation (C_r × C_r) --------------------------
        A = torch.bmm(k_r, v_r)  # (B, C_r, C_r)

        # Bilinear fusion ----------------------------------------------
        A1, A2 = self._split_subsets(A)
        W = self.sigmoid(A1) + self.sigmoid(A2)  # (B, C_r, C_r)

        # Attention output ---------------------------------------------
        y_r = torch.bmm(q_r, W.transpose(1, 2))  # (B, HW, C_r)
        y = y_r.permute(0, 2, 1).reshape(B, -1, H, W)  # (B, C_r, H, W)

        # Projection back to C channels --------------------------------
        y = self.proj_out(y)  # (B, C, H, W)

        # Optional channel shuffle for better mixing -------------------
        y = channel_shuffle(y, groups=self.shuffle_groups)

        return x + y  # residual connection
