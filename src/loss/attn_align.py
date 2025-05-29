# src/loss/attn_align.py
# ------------------------------------------------------------------
# Attention Alignment Loss
# Encourages consistency between predicted attention maps and
# the structure of the high-resolution ground truth.
# Designed to guide BFSA in EFANet.
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionAlignmentLoss(nn.Module):
    """
    Attention Alignment Loss (AAL).

    This loss penalizes spatial misalignment between attention maps (A_pred)
    and normalized HR-derived guidance maps (A_gt). It is computed as
    element-wise L1 or cosine distance across spatial maps.

    Parameters
    ----------
    mode : str, default='l1'
        The alignment mode to compare attention maps. Options: ['l1', 'cosine'].
    reduction : str, default='mean'
        Specifies the reduction to apply to the output: 'mean' | 'sum'.
    normalize : bool, default=True
        Whether to apply softmax normalization to both attention maps.
    """

    def __init__(self, mode: str = 'l1', reduction: str = 'mean', normalize: bool = True):
        super().__init__()
        if mode not in ['l1', 'cosine']:
            raise ValueError("Unsupported mode: choose from ['l1', 'cosine']")
        if reduction not in ['mean', 'sum']:
            raise ValueError("Unsupported reduction: choose from ['mean', 'sum']")
        self.mode = mode
        self.reduction = reduction
        self.normalize = normalize

    def forward(self, A_pred: torch.Tensor, A_gt: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        A_pred : Tensor
            Predicted attention map, shape [B, C, H, W].
        A_gt : Tensor
            Ground-truth guidance map derived from HR, shape [B, C, H, W].

        Returns
        -------
        Tensor
            Scalar alignment loss.
        """
        if self.normalize:
            A_pred = F.softmax(A_pred.view(A_pred.size(0), A_pred.size(1), -1), dim=-1).view_as(A_pred)
            A_gt = F.softmax(A_gt.view(A_gt.size(0), A_gt.size(1), -1), dim=-1).view_as(A_gt)

        if self.mode == 'l1':
            loss = torch.abs(A_pred - A_gt)
        elif self.mode == 'cosine':
            A_pred_norm = F.normalize(A_pred, p=2, dim=1)
            A_gt_norm = F.normalize(A_gt, p=2, dim=1)
            loss = 1.0 - (A_pred_norm * A_gt_norm).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()
