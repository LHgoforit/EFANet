# src/utils/config.py

import os
import yaml
import argparse
from easydict import EasyDict as edict


def load_config(config_path: str) -> edict:
    """
    Load YAML configuration file into EasyDict for attribute-style access.

    Args:
        config_path (str): Path to the .yaml configuration file.

    Returns:
        edict: Parsed configuration.
    """
    assert os.path.isfile(config_path), f"Config file not found: {config_path}"
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    cfg = edict(cfg_dict)
    _validate_config(cfg)
    return cfg


def save_config(cfg: edict, out_path: str) -> None:
    """
    Save EasyDict configuration back to a YAML file.

    Args:
        cfg (edict): Configuration to save.
        out_path (str): Destination file path.
    """
    with open(out_path, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def _validate_config(cfg: edict) -> None:
    """
    Basic validation for required fields in the configuration.
    Raises AssertionError if validation fails.
    """
    assert 'model' in cfg and 'name' in cfg.model, "Missing model.name"
    assert 'dataset' in cfg and 'train' in cfg.dataset, "Missing dataset.train"
    assert 'optim' in cfg and 'type' in cfg.optim, "Missing optim.type"
    assert 'train' in cfg and 'epochs' in cfg.train, "Missing train.epochs"


def parse_args() -> argparse.Namespace:
    """
    Command-line argument parser for training/testing scripts.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='EFANet Config Loader')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to the configuration YAML file.')
    args = parser.parse_args()
    return args
