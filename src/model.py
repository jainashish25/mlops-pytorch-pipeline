"""Model definitions for the CIFAR-10 image classifier.

Provides `get_model(architecture, num_classes)` used by both training and
serving. Supports a ResNet-18 (adapted for 32x32 CIFAR images) and a small
custom CNN.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """A compact CNN baseline for CIFAR-10 (3x32x32 -> num_classes)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                             # 32 -> 16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                             # 16 -> 8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for 32x32 inputs.

    The stock torchvision ResNet-18 uses a 7x7 stride-2 conv and a maxpool
    designed for 224x224 ImageNet images, which throws away too much spatial
    information on 32x32 CIFAR. We swap in a 3x3 stride-1 stem and drop the
    initial maxpool, a standard CIFAR adaptation.
    """
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """Factory: return a model by name.

    Args:
        architecture: "resnet18" or "simplecnn".
        num_classes: number of output classes (10 for CIFAR-10).
    """
    architecture = architecture.lower()
    if architecture == "resnet18":
        return _resnet18_cifar(num_classes)
    if architecture in ("simplecnn", "cnn"):
        return SimpleCNN(num_classes)
    raise ValueError(f"Unknown architecture: {architecture!r} (use 'resnet18' or 'simplecnn')")
