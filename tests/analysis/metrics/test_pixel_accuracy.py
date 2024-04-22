import torch

from minerva.analysis.metrics import PixelAccuracy


def test_pixel_accuracy():
    metric = PixelAccuracy()

    # Test case 1: All correct predictions
    preds = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([0, 1, 2, 3])
    metric.update(preds, target)
    assert metric.compute() == 1.0

    # Test case 2: All incorrect predictions
    preds = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([3, 2, 1, 0])
    metric.update(preds, target)
    assert metric.compute() == 0.0

    # Test case 3: Partially correct predictions
    preds = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([0, 1, 1, 0])
    metric.update(preds, target)
    assert metric.compute() == 0.5

    # Test case 4: Empty predictions
    preds = torch.tensor([])
    target = torch.tensor([])
    metric.update(preds, target)
    assert metric.compute() == 1.0

    # Test case 5: Empty target
    preds = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([])
    metric.update(preds, target)
    assert metric.compute() == 0.0

    # Test case 6: Empty predictions and target
    preds = torch.tensor([])
    target = torch.tensor([])
    metric.update(preds, target)
    assert metric.compute() == 1.0
