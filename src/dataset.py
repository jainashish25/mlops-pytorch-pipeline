"""CIFAR-10 data loading: transforms + DataLoaders.

Extends the assignment starter snippet with configurable workers and a class
name lookup used by the serving layer.
"""
from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10 channel statistics.
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return train (augmented) or eval transforms."""
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for CIFAR-10.

    Downloads the dataset into `data_dir` if not already present.
    """
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=get_transforms(train=True),
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=get_transforms(train=False),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
