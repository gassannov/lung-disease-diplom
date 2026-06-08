"""Evaluation entry point for saved torch experiments."""

from argparse import ArgumentParser
from pathlib import Path

import torch

from src.data.annotations import CLASS_NAMES
from src.train import build_loader, build_torch_model, run_epoch
from src.utils.config import load_config
from src.utils.logging import save_json
from src.utils.metrics import save_confusion_matrix


def main() -> None:
    """Evaluate a saved checkpoint on a configured split.

    Args:
        None.

    Returns:
        None.
    """
    parser = ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = load_config(run_dir / "config.yaml")
    device_name = config["training"].get("device", "cuda")
    device = torch.device(
        device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu"
    )
    loader = build_loader(config, args.split, False)
    model = build_torch_model(config).to(device)
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", map_location=device))
    criterion = torch.nn.CrossEntropyLoss()
    metrics = run_epoch(model, loader, criterion, device)
    save_json(
        {key: value for key, value in metrics.items() if key not in {"y_true", "y_pred"}},
        run_dir / f"{args.split}_metrics_recomputed.json",
    )
    save_confusion_matrix(
        metrics["y_true"],
        metrics["y_pred"],
        CLASS_NAMES,
        run_dir / "figures" / f"{args.split}_confusion_matrix_recomputed.png",
    )


if __name__ == "__main__":
    main()
