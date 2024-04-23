import torch

from minerva.analysis.metrics import PixelAccuracy


def test_pixel_accuracy1():
    metric = PixelAccuracy()

    # Test case 1: All correct predictions
    preds = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([0, 1, 2, 3])
    metric(preds, target)
    assert metric.compute() == 1.0


def test_pixel_accuracy2():
    metric = PixelAccuracy()

    # Test case 2: All incorrect predictions
    preds = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([3, 2, 1, 0])
    metric(preds, target)
    assert metric.compute() == 0.0


def test_pixel_accuracy3():
    metric = PixelAccuracy()

    # Test case 3: Partially correct predictions
    preds = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([0, 1, 1, 0])
    metric(preds, target)
    assert metric.compute() == 0.5
