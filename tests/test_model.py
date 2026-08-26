"""Unit tests for the model factory and forward shapes."""
import torch

from src.model import get_model


def test_resnet18_output_shape():
    model = get_model("resnet18", num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_simplecnn_output_shape():
    model = get_model("simplecnn", num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_custom_num_classes():
    model = get_model("resnet18", num_classes=5)
    out = model(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, 5)


def test_unknown_architecture_raises():
    try:
        get_model("does-not-exist")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown architecture")
