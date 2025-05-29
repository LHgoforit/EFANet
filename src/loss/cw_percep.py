# src/loss/cw_percep.py
# ------------------------------------------------------------------
# Channel-wise Perceptual Loss (CW-Percep)
# As used in EFANet: measures perceptual similarity by comparing
# intermediate activations from a pretrained VGG network,
# but emphasizes channel-wise structural fidelity.
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List


class VGGFeatureExtractor(nn.Module):
    """
    VGG19 feature extractor for perceptual loss computation.

    Parameters
    ----------
    layer_ids : List[int]
        List of VGG layer indices from which features are extracted.
    use_input_norm : bool, default=True
        Whether to normalize inputs using ImageNet statistics.
    """

    def __init__(self, layer_ids: List[int], use_input_norm: bool = True) -> None:
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = nn.ModuleList([vgg[:i + 1].eval() for i in layer_ids])
        for param in self.parameters():
            param.requires_grad = False

        self.use_input_norm = use_input_norm
        if use_input_norm:
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


class ChannelWisePerceptualLoss(nn.Module):
    """
    Channel-Wise Perceptual Loss (CW-Percep).

    Given intermediate feature maps from a pretrained network (e.g. VGG),
    this loss computes a weighted L1 distance across channels
    to emphasize semantic fidelity per channel.

    Parameters
    ----------
    layer_ids : List[int], default=[2, 7, 16]
        VGG layer indices to extract features from.
    weights : List[float], optional
        Optional weighting for each layer loss.
    reduction : str, default='mean'
        Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'.
    """

    def __init__(
        self,
        layer_ids: List[int] = [2, 7, 16],
        weights: List[float] = None,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        self.vgg = VGGFeatureExtractor(layer_ids)
        self.reduction = reduction
        self.weights = weights if weights is not None else [1.0] * len(layer_ids)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feats = self.vgg(pred)
        target_feats = self.vgg(target)

        loss = 0.0
        for i, (f_pred, f_target) in enumerate(zip(pred_feats, target_feats)):
            # Channel-wise L1 norm
            channel_diff = torch.abs(f_pred - f_target).mean(dim=(2, 3))  # shape: [B, C]
            layer_loss = channel_diff.mean(dim=1)  # mean over channels
            weighted_loss = self.weights[i] * layer_loss

            if self.reduction == 'mean':
                loss += weighted_loss.mean()
            elif self.reduction == 'sum':
                loss += weighted_loss.sum()
            else:
                raise ValueError(f"Unsupported reduction mode: {self.reduction}")

        return loss
