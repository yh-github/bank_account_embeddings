"""Training utilities."""

import logging
import os
import sys
from typing import Any

import torch


def setup_logger(output_dir: str) -> logging.Logger:
    """Sets up a logger that writes to console and a file.

    Args:
        output_dir: Directory to save the log file.

    Returns:
        The configured logger.
    """
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    # Ensure dir
    os.makedirs(output_dir, exist_ok=True)

    # Check if logger already exists to avoid duplicate handlers
    logger = logging.getLogger('train')
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)

    fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def recursive_to_device(obj: Any, device: str) -> Any:
    """Recursively moves tensors in dicts/lists to the specified device.

    Args:
        obj: The object (Tensor, dict, list, or other) to move.
        device: The target device (e.g., 'cuda', 'cpu').

    Returns:
        The moved object with the same structure.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: recursive_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to_device(v, device) for v in obj]
    else:
        return obj
