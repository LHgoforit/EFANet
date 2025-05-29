import torch
import torch.nn.functional as F


def reshape_to_batches(x, num_groups):
    """
    Reshape input tensor into grouped batches for group-wise processing.
    Args:
        x (Tensor): Input tensor of shape (B, C, H, W)
        num_groups (int): Number of groups to split channels
    Returns:
        Tensor: Reshaped tensor of shape (B * num_groups, C // num_groups, H, W)
    """
    B, C, H, W = x.shape
    assert C % num_groups == 0, f"Channel size {C} must be divisible by num_groups={num_groups}"
    x = x.view(B, num_groups, C // num_groups, H, W)
    x = x.permute(0, 1, 3, 4, 2).reshape(B * num_groups, H, W, C // num_groups)
    x = x.permute(0, 3, 1, 2).contiguous()
    return x


def restore_from_batches(x, num_groups, original_batch):
    """
    Inverse of reshape_to_batches. Merge grouped batches back into original.
    Args:
        x (Tensor): Tensor of shape (B * G, C', H, W)
        num_groups (int): Number of groups G
        original_batch (int): Original batch size B
    Returns:
        Tensor: Tensor of shape (B, C' * G, H, W)
    """
    B = original_batch
    G = num_groups
    Cg, H, W = x.shape[1:]
    x = x.view(B, G, Cg, H, W)
    x = x.permute(0, 2, 1, 3, 4).reshape(B, G * Cg, H, W)
    return x


def repeat_interleave(x, times, dim):
    """
    Repeat each slice along a dimension (interleaving).
    Args:
        x (Tensor): Input tensor.
        times (int): Number of times to repeat.
        dim (int): Dimension to repeat along.
    Returns:
        Tensor: Repeated tensor.
    """
    return x.repeat_interleave(repeats=times, dim=dim)


def safe_cat(tensor_list, dim=1):
    """
    Safely concatenate a list of tensors along a specified dimension.
    Args:
        tensor_list (List[Tensor]): Tensors to concatenate.
        dim (int): Dimension along which to concatenate.
    Returns:
        Tensor: Concatenated tensor.
    """
    filtered = [t for t in tensor_list if t is not None]
    if not filtered:
        raise ValueError("safe_cat received an empty or all-None tensor list.")
    return torch.cat(filtered, dim=dim)


def resize_like(src, target, mode='bilinear', align_corners=False):
    """
    Resize src tensor to match spatial dimensions of target.
    Args:
        src (Tensor): Source tensor to resize (B, C, H1, W1)
        target (Tensor): Target tensor (B, C, H2, W2)
        mode (str): Interpolation method.
        align_corners (bool): Used for bilinear/nearest modes.
    Returns:
        Tensor: Resized src to match target H/W.
    """
    return F.interpolate(src, size=target.shape[2:], mode=mode, align_corners=align_corners)
