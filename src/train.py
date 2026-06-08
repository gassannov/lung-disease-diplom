"""Training entry point for all respiratory audio experiments."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data.annotations import CLASS_NAMES
from src.data.dataset import RespiratoryAudioDataset
from src.features.mfcc import build_feature_table
from src.losses import FocalLoss
from src.models.ast import build_ast
from src.models.crnn import CRNNClassifier
from src.models.ml import build_estimator, save_estimator
from src.models.resnet import build_resnet, count_parameters
from src.utils.config import load_config, save_config
from src.utils.logging import append_metrics, create_run_dir, save_json
from src.utils.metrics import (
    class_weights,
    compute_metrics,
    save_confusion_matrix,
    save_training_curves,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments.

    Args:
        seed: Integer random seed.

    Returns:
        None.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def labels_for_split(index_csv: str | Path, split: str) -> list[int]:
    """Read labels for a metadata split.

    Args:
        index_csv: Metadata CSV path.
        split: Split name.

    Returns:
        List of integer labels.
    """
    frame = pd.read_csv(index_csv)
    return frame.loc[frame["split"] == split, "label"].astype(int).tolist()


def build_loader(config: dict[str, Any], split: str, sampler: bool) -> DataLoader:
    """Build a DataLoader for a torch experiment.

    Args:
        config: Experiment configuration.
        split: Split name.
        sampler: Whether to use weighted sampling.

    Returns:
        Configured DataLoader.
    """
    data_config = config["data"]
    feature_config = config["features"]
    training_config = config["training"]
    dataset = RespiratoryAudioDataset(
        data_config["index_csv"],
        split,
        int(data_config["sample_rate"]),
        float(data_config["duration"]),
        "ast" if config["model"]["type"] == "ast" else "logmel",
        int(feature_config["mel_bins"]),
        int(feature_config["window_ms"]),
        int(feature_config["hop_ms"]),
        int(feature_config.get("ast_max_length", 1024)),
        training_config.get("augmentation"),
    )
    weighted_sampler = None
    shuffle = split == "train"
    if sampler and split == "train":
        weights = class_weights(dataset.frame["label"].astype(int).tolist(), len(CLASS_NAMES))
        sample_weights = [weights[int(label)] for label in dataset.frame["label"]]
        weighted_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=shuffle,
        sampler=weighted_sampler,
        num_workers=2,
        pin_memory=True,
    )


def build_torch_model(config: dict[str, Any]) -> nn.Module:
    """Create a torch model from experiment configuration.

    Args:
        config: Experiment configuration.

    Returns:
        PyTorch model.
    """
    model_config = config["model"]
    if model_config["type"] == "resnet":
        return build_resnet(
            model_config["backbone"],
            len(CLASS_NAMES),
            bool(model_config.get("pretrained", False)),
        )
    if model_config["type"] == "crnn":
        return CRNNClassifier(
            len(CLASS_NAMES),
            model_config.get("rnn_type", "gru"),
            int(model_config.get("hidden_size", 128)),
            int(model_config.get("num_layers", 2)),
        )
    if model_config["type"] == "ast":
        return build_ast(
            model_config["pretrained_model_name_or_path"],
            len(CLASS_NAMES),
            bool(model_config.get("freeze_backbone", False)),
        )
    raise ValueError(f"Unknown torch model type: {model_config['type']}")


def forward_logits(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Run a model and normalize Hugging Face and PyTorch outputs.

    Args:
        model: Classifier module.
        features: Input features.

    Returns:
        Logits tensor.
    """
    outputs = model(features)
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Run one train or evaluation epoch.

    Args:
        model: Classifier module.
        loader: DataLoader for one split.
        criterion: Loss function.
        device: Torch device.
        optimizer: Optimizer or None for evaluation.

    Returns:
        Loss and prediction metrics.
    """
    model.train(optimizer is not None)
    losses = []
    y_true = []
    y_pred = []
    for features, labels in tqdm(loader, leave=False):
        features = features.to(device)
        labels = labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits = forward_logits(model, features)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
    metrics["loss"] = float(np.mean(losses))
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    return metrics


def train_torch(config: dict[str, Any], run_dir: Path) -> None:
    """Train a PyTorch model and save artifacts.

    Args:
        config: Experiment configuration.
        run_dir: Run directory for artifacts.

    Returns:
        None.
    """
    training_config = config["training"]
    device_name = training_config.get("device", "cuda")
    device = torch.device(
        device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu"
    )
    train_loader = build_loader(
        config,
        "train",
        bool(training_config.get("weighted_sampler", False)),
    )
    val_loader = build_loader(config, "val", False)
    test_loader = build_loader(config, "test", False)
    model = build_torch_model(config).to(device)
    save_json(
        {"total_params": count_parameters(model), "device": str(device)},
        run_dir / "model_info.json",
    )

    train_labels = labels_for_split(config["data"]["index_csv"], "train")
    weights = None
    if bool(training_config.get("weighted_loss", False)):
        weights = torch.tensor(
            class_weights(train_labels, len(CLASS_NAMES)),
            dtype=torch.float32,
            device=device,
        )
    if training_config.get("loss", "cross_entropy") == "focal":
        criterion = FocalLoss(float(training_config.get("focal_gamma", 2.0)), weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training_config["lr"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    best_macro_f1 = -1.0
    for epoch in range(1, int(training_config["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        }
        append_metrics(row, run_dir / "metrics.csv")
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best.pt")
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", map_location=device))
    test_metrics = run_epoch(model, test_loader, criterion, device)
    save_json(
        {key: value for key, value in test_metrics.items() if key not in {"y_true", "y_pred"}},
        run_dir / "test_metrics.json",
    )
    save_confusion_matrix(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        CLASS_NAMES,
        run_dir / "figures" / "confusion_matrix.png",
    )
    save_training_curves(run_dir / "metrics.csv", run_dir / "figures" / "training_curves.png")


def train_ml(config: dict[str, Any], run_dir: Path) -> None:
    """Train a classical ML baseline and save artifacts.

    Args:
        config: Experiment configuration.
        run_dir: Run directory for artifacts.

    Returns:
        None.
    """
    features, labels, frame = build_feature_table(
        config["data"]["index_csv"],
        int(config["data"]["sample_rate"]),
        float(config["data"]["duration"]),
        int(config["features"]["mfcc"]),
    )
    train_mask = frame["split"].eq("train").to_numpy()
    val_mask = frame["split"].eq("val").to_numpy()
    test_mask = frame["split"].eq("test").to_numpy()
    estimator = build_estimator(
        config["model"].get("estimator", "logistic_regression"),
        config["training"].get("class_weight"),
        int(config["training"]["seed"]),
    )
    estimator.fit(features[train_mask], labels[train_mask])
    save_estimator(estimator, str(run_dir / "model.joblib"))
    for split, mask in {"val": val_mask, "test": test_mask}.items():
        predictions = estimator.predict(features[mask])
        metrics = compute_metrics(labels[mask].tolist(), predictions.tolist(), CLASS_NAMES)
        save_json(metrics, run_dir / f"{split}_metrics.json")
        save_confusion_matrix(
            labels[mask].tolist(),
            predictions.tolist(),
            CLASS_NAMES,
            run_dir / "figures" / f"{split}_confusion_matrix.png",
        )


def main() -> None:
    """Run an experiment from a YAML configuration.

    Args:
        None.

    Returns:
        None.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["training"]["seed"]))
    run_dir = create_run_dir(config["output_dir"], config["name"])
    save_config(config, run_dir / "config.yaml")
    if config["model"]["type"] == "ml":
        train_ml(config, run_dir)
    else:
        train_torch(config, run_dir)
    print(f"saved run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
