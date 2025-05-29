# src/loss/charbonnier.py
# ------------------------------------------------------------------
#  Charbonnier Loss implementation
#  Commonly used for image restoration tasks due to robustness
#  to outliers and smooth gradient near zero.
#  Reference: "Deep Laplacian Pyramid Networks for Fast and Accurate
#  Super-Resolution", CVPR 2017 (Lai et al.)
# ------------------------------------------------------------------

import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (a differentiable variant of L1 loss):
        L(x, y) = sqrt((x - y)^2 + epsilon^2)

    Parameters
    ----------
    epsilon : float, default=1e-3
        A small constant for numerical stability.
    reduction : str, default='mean'
        Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'.
    """

    def __init__(self, epsilon: float = 1e-3, reduction: str = 'mean') -> None:
        super().__init__()
        self.epsilon = epsilon
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon ** 2)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
