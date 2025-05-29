import os
import numpy as np
from PIL import Image
import torchvision.utils as vutils
import torch


def save_image_grid(images, out_path, nrow=None, padding=2, normalize=True, value_range=(0, 1)):
    """
    Save a grid of images to a file.

    Args:
        images (Tensor or list of PIL.Image): Images to save.
        out_path (str): Destination file path.
        nrow (int): Number of images in each row.
        padding (int): Space between images.
        normalize (bool): Normalize image pixels to [0, 1].
        value_range (tuple): Min and max values for normalization.
    """
    if isinstance(images, list):
        images = torch.stack([torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255. for img in images])

    grid = vutils.make_grid(
        images,
        nrow=nrow or len(images),
        padding=padding,
        normalize=normalize,
        range=value_range
    )
    ndarr = grid.mul(255).clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy()
    Image.fromarray(ndarr).save(out_path)


def tile_image_row(img_list, out_path, resize_h=None, resize_w=None):
    """
    Horizontally concatenate a list of PIL images into a single row image and save.

    Args:
        img_list (list of PIL.Image): List of images to tile.
        out_path (str): Path to save the resulting image.
        resize_h (int): Optional height to resize each image.
        resize_w (int): Optional width to resize each image.
    """
    if resize_h and resize_w:
        img_list = [img.resize((resize_w, resize_h), Image.BICUBIC) for img in img_list]

    total_width = sum(img.width for img in img_list)
    max_height = max(img.height for img in img_list)
    row_img = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in img_list:
        row_img.paste(img, (x_offset, 0))
        x_offset += img.width

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    row_img.save(out_path)


def tensor_to_pil(tensor):
    """
    Convert a Tensor image [C x H x W] to a PIL Image (uint8).
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().clamp(0, 1)
        tensor = tensor.permute(1, 2, 0).numpy()
    img = (tensor * 255.0).round().astype(np.uint8)
    return Image.fromarray(img)


def save_comparison_row(images_dict, out_path, order=None, resize=(256, 256)):
    """
    Save a horizontal comparison of different methods as a single row image.

    Args:
        images_dict (dict): Keys are method names, values are PIL images or tensors.
        out_path (str): Destination file path.
        order (list): Optional display order of methods.
        resize (tuple): Target (H, W) for each image.
    """
    ordered_keys = order or list(images_dict.keys())
    img_list = []

    for key in ordered_keys:
        img = images_dict[key]
        if isinstance(img, torch.Tensor):
            img = tensor_to_pil(img)
        img = img.resize(resize, Image.BICUBIC)
        img_list.append(img)

    tile_image_row(img_list, out_path)


def save_batch_comparisons(batch_dict_list, output_dir, postfix=".png", resize=(256, 256)):
    """
    Save a batch of visual comparisons.

    Args:
        batch_dict_list (list of dict): Each dict is a mapping {method_name: image}.
        output_dir (str): Save directory.
        postfix (str): File name suffix.
        resize (tuple): Resize each image to (H, W).
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx, img_dict in enumerate(batch_dict_list):
        save_comparison_row(
            img_dict,
            out_path=os.path.join(output_dir, f"cmp_{idx:02d}{postfix}"),
            resize=resize
        )
