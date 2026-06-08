"""Experiment logging helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def create_run_dir(output_dir: str | Path, name: str) -> Path:
    """Create a timestamped experiment directory.

    Args:
        output_dir: Parent directory for experiment runs.
        name: Human-readable experiment name.

    Returns:
        Created run directory path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    return run_dir


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save JSON data with stable formatting.

    Args:
        data: Serializable dictionary.
        path: Destination JSON path.

    Returns:
        None.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def append_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    """Append one metrics row to a CSV file.

    Args:
        metrics: Metrics dictionary for one epoch or evaluation stage.
        path: Destination CSV path.

    Returns:
        None.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([metrics])
    frame.to_csv(destination, mode="a", header=not destination.exists(), index=False)

