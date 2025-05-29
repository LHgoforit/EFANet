# src/utils/__init__.py

"""
Utility package for EFANet:
Includes config parsing, logging, training helpers, checkpointing,
visualization tools, and reproducibility utilities.
"""

from .config import load_config, save_config
from .logger import get_logger
from .seed import set_seed
from .trainer import Trainer
from .checkpoint import save_checkpoint, load_checkpoint
from .optim import build_optimizer, build_scheduler
from .metrics_io import save_metrics_json, generate_latex_table
from .visualizer import save_comparison_grid
from .profiler import profile_model
from .tensor_ops import reshape_as, broadcast_to
from .registry import Registry

__all__ = [
    "load_config",
    "save_config",
    "get_logger",
    "set_seed",
    "Trainer",
    "save_checkpoint",
    "load_checkpoint",
    "build_optimizer",
    "build_scheduler",
    "save_metrics_json",
    "generate_latex_table",
    "save_comparison_grid",
    "profile_model",
    "reshape_as",
    "broadcast_to",
    "Registry",
]
