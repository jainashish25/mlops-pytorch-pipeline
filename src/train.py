"""Training entrypoint for the CIFAR-10 classifier.

- Reads hyperparameters from configs/training_config.yaml (path overridable via
  the CONFIG_PATH environment variable or a mounted volume at /app/configs).
- Logs per-epoch metrics to stdout as JSON lines.
- Saves the best checkpoint (by validation loss) to a configurable path.
- Supports early stopping.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import get_model


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_config_path() -> Path:
    """Resolve the config path: env var > mounted volume > repo default."""
    env = os.environ.get("CONFIG_PATH")
    candidates = [Path(env)] if env else []
    candidates += [Path("/app/configs/training_config.yaml"),
                   Path("configs/training_config.yaml")]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No config found in: {[str(c) for c in candidates]}")


def log(**kwargs) -> None:
    """Emit one structured JSON line to stdout."""
    print(json.dumps(kwargs), flush=True)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


def main() -> None:
    config_path = resolve_config_path()
    config = load_config(str(config_path))
    log(event="config_loaded", path=str(config_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(event="device_selected", device=str(device))

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 2),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    patience = config["training"]["early_stopping_patience"]
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / config["output"]["model_name"]

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        log(epoch=epoch + 1,
            train_loss=round(train_loss, 4), train_accuracy=round(train_acc, 4),
            val_loss=round(val_loss, 4), val_accuracy=round(val_acc, 4))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "architecture": config["model"]["architecture"],
                "num_classes": config["model"]["num_classes"],
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, save_path)
            log(event="checkpoint_saved", path=str(save_path), val_loss=round(val_loss, 4))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(event="early_stopping", epoch=epoch + 1)
                break

    log(event="training_complete", best_val_loss=round(best_val_loss, 4),
        checkpoint=str(save_path))


if __name__ == "__main__":
    main()
