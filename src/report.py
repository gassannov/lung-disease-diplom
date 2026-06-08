"""Aggregate experiment outputs for future diploma reports."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.utils.logging import save_json


def collect_runs(runs_dir: str | Path) -> pd.DataFrame:
    """Collect test metrics from experiment run directories.

    Args:
        runs_dir: Parent reports/runs directory.

    Returns:
        Summary DataFrame with one row per completed run.
    """
    rows = []
    for run_dir in sorted(Path(runs_dir).glob("*")):
        metrics_path = run_dir / "test_metrics.json"
        config_path = run_dir / "config.yaml"
        if not metrics_path.exists() or not config_path.exists():
            continue
        import json

        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        rows.append(
            {
                "run": run_dir.name,
                "macro_f1": metrics.get("macro_f1"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "weighted_f1": metrics.get("weighted_f1"),
                "per_class_recall": metrics.get("per_class_recall"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Write aggregate experiment summaries.

    Args:
        None.

    Returns:
        None.
    """
    parser = ArgumentParser()
    parser.add_argument("--runs-dir", default="reports/runs")
    parser.add_argument("--output", default="reports/summary.csv")
    args = parser.parse_args()
    summary = collect_runs(args.runs_dir)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination, index=False)
    save_json({"runs": len(summary)}, destination.with_suffix(".json"))
    print(f"saved {len(summary)} runs to {destination}")


if __name__ == "__main__":
    main()

