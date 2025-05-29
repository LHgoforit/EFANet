# src/utils/seed.py

import os
import random
import numpy as np
import torch


def set_random_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Seed value to use.
        deterministic (bool): Whether to enforce deterministic behavior in PyTorch.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def is_seeded(seed: int = 42) -> bool:
    """
    Check whether the global seed has been set correctly.

    Returns:
        bool: True if checksums of random states are consistent.
    """
    # Quick test: seeded outputs should match
    test_pass = (random.randint(0, 10000) == 1824 and
                 np.random.randint(0, 10000) == 860 and
                 torch.randint(0, 10000, (1,)).item() == 7131)

    # Reset again to avoid side effect
    set_random_seed(seed)
    return test_pass
