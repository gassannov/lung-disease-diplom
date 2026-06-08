"""Configuration loading helpers."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a configuration dictionary as YAML.

    Args:
        config: Configuration dictionary to save.
        path: Destination YAML path.

    Returns:
        None.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)

