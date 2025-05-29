# src/data/transforms.py
# ------------------------------------------------------------------
# Custom Transform Utilities for Face Super-Resolution
# Includes paired transforms for consistent LR–HR augmentation.
# Applied during training of EFANet on facial datasets (CelebA, Helen, FFHQ).
# ------------------------------------------------------------------

import random
from PIL import Image
import torchvision.transforms.functional as TF


class PairedRandomHorizontalFlip:
    """
    Apply horizontal flip to both HR and LR images with a given probability.

    Parameters
    ----------
    prob : float, default=0.5
        Probability of applying the horizontal flip.
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, lr: Image.Image, hr: Image.Image):
        if random.random() < self.prob:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)
        return lr, hr


class PairedRandomRotation:
    """
    Apply the same random rotation to both HR and LR images.

    Parameters
    ----------
    degrees : int, default=10
        Maximum rotation angle in degrees (±degrees).
    """

    def __init__(self, degrees: int = 10):
        self.degrees = degrees

    def __call__(self, lr: Image.Image, hr: Image.Image):
        angle = random.uniform(-self.degrees, self.degrees)
        lr = TF.rotate(lr, angle, resample=Image.BICUBIC)
        hr = TF.rotate(hr, angle, resample=Image.BICUBIC)
        return lr, hr


class PairedRandomCrop:
    """
    Apply a synchronized crop to both HR and LR images.

    Parameters
    ----------
    hr_crop_size : int
        Size of HR image crop. LR crop will be scaled accordingly.
    scale : int
        Super-resolution scale factor (e.g., 4, 8, 16).
    """

    def __init__(self, hr_crop_size: int, scale: int):
        self.hr_crop_size = hr_crop_size
        self.lr_crop_size = hr_crop_size // scale
        self.scale = scale

    def __call__(self, lr: Image.Image, hr: Image.Image):
        hr_w, hr_h = hr.size
        if hr_w < self.hr_crop_size or hr_h < self.hr_crop_size:
            raise ValueError("HR crop size larger than image dimensions.")

        hr_left = random.randint(0, hr_w - self.hr_crop_size)
        hr_top = random.randint(0, hr_h - self.hr_crop_size)
        lr_left = hr_left // self.scale
        lr_top = hr_top // self.scale

        hr_crop = TF.crop(hr, hr_top, hr_left, self.hr_crop_size, self.hr_crop_size)
        lr_crop = TF.crop(lr, lr_top, lr_left, self.lr_crop_size, self.lr_crop_size)
        return lr_crop, hr_crop


class PairedToTensor:
    """
    Convert PIL Images of LR and HR pairs to Tensors in [0, 1] range.
    """

    def __call__(self, lr: Image.Image, hr: Image.Image):
        return TF.to_tensor(lr), TF.to_tensor(hr)
