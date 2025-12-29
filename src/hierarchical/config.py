"""Configuration management for hierarchical embeddings.

This module provides utilities for loading and accessing configuration
from YAML files or environment variables.

Usage:
    from hierarchical.config import load_config, get_config

    # Load from environment variable EMBEDDER_CONFIG or default
    config = load_config()

    # Load from specific file
    config = load_config('/path/to/config.yaml')

    # Access paths
    data_dir = config['paths']['data_dir']
"""

import os
from pathlib import Path
from typing import Any, cast

import yaml

# Global config cache
_config: dict[str, Any] | None = None

# Default config location relative to package
DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent.parent.parent / "config" / "default.yaml"
)


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Priority order:
    1. Explicit config_path argument
    2. EMBEDDER_CONFIG environment variable
    3. Default config file (config/default.yaml)

    Args:
        config_path: Optional path to config YAML file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If no config file is found.
        ValueError: If required paths are not set.
    """
    global _config

    # Determine config file path
    if config_path is None:
        config_path = os.environ.get("EMBEDDER_CONFIG")

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Please either:\n"
            "  1. Set EMBEDDER_CONFIG environment variable\n"
            "  2. Copy config/default.yaml to config/local.yaml and update paths"
        )

    with open(config_path) as f:
        _config = yaml.safe_load(f)

    return _config


def get_config() -> dict[str, Any]:
    """Get the currently loaded configuration.

    Returns:
        Configuration dictionary.

    Raises:
        RuntimeError: If config has not been loaded yet.
    """
    if _config is None:
        # Try to load default
        return load_config()
    return _config


def get_path(key: str, required: bool = True) -> Path | None:
    """Get a path from config, resolving relative paths.

    Args:
        key: Path key (e.g., 'data_dir', 'output_dir').
        required: If True, raise error if path is None.

    Returns:
        Resolved Path object, or None if not set and not required.

    Raises:
        ValueError: If required path is not set.
    """
    config = get_config()
    path_str = config.get("paths", {}).get(key)

    if path_str is None:
        if required:
            raise ValueError(
                f"Required path '{key}' not set in config.\n"
                "Please update your config file with the correct path."
            )
        return None

    path = Path(path_str)

    # Resolve relative paths relative to current working directory
    if not path.is_absolute():
        path = Path.cwd() / path

    return path


def get_banks() -> list[str]:
    """Get list of bank names from config.

    Returns:
        List of bank directory names.
    """
    config = get_config()
    return config.get("banks", [])


def get_model_config() -> dict[str, Any]:
    """Get model hyperparameters from config.

    Returns:
        Dictionary of model parameters.
    """
    config = get_config()
    return config.get("model", {})


def get_training_config() -> dict[str, Any]:
    """Get training parameters from config.

    Returns:
        Dictionary of training parameters.
    """
    config = get_config()
    return config.get("training", {})
