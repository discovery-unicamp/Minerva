import torch
from minerva.analysis.metrics import BalancedAccuracy

def test_balanced_accuracy():
    # Test case 1: Basic binary classification with perfect accuracy
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([0, 1, 0, 1])
    metric = BalancedAccuracy(num_classes=2, task='binary')
    metric.update(y_pred, y_true)
    expected = 1.0
    assert torch.isclose(metric.compute().float(), torch.tensor(expected).float()), "Test case 1 failed"

    # Test case 2: Basic binary classification with random accuracy
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([1, 0, 1, 0])
    metric = BalancedAccuracy(num_classes=2, task='binary')
    metric.update(y_pred, y_true)
    expected = 0.0
    assert torch.isclose(metric.compute().float(), torch.tensor(expected).float()), "Test case 2 failed"

    # Test case 3: Multiclass classification with perfect accuracy
    y_true = torch.tensor([0, 1, 2, 0, 1, 2])
    y_pred = torch.tensor([0, 1, 2, 0, 1, 2])
    metric = BalancedAccuracy(num_classes=3, task='multiclass')
    metric.update(y_pred, y_true)
    expected = 1.0
    assert torch.isclose(metric.compute().float(), torch.tensor(expected).float()), "Test case 3 failed"

    # Test case 4: Multiclass classification with missing classes in y_pred
    y_true = torch.tensor([0, 1, 2, 0, 1, 2])
    y_pred = torch.tensor([0, 1, 0, 0, 1, 0])  # class 2 missing in y_pred
    metric = BalancedAccuracy(num_classes=3, task='multiclass')
    metric.update(y_pred, y_true)
    expected = 0.6666666666666666
    assert torch.isclose(metric.compute().float(), torch.tensor(expected).float()), "Test case 4 failed"

    # Test case 5: Multiclass classification with some classes not present in the batch
    y_true = torch.tensor([0, 0, 0, 0, 0, 0])
    y_pred = torch.tensor([0, 0, 0, 0, 0, 0])  # only one class present
    metric = BalancedAccuracy(num_classes=3, task='multiclass')
    metric.update(y_pred, y_true)
    expected = 1.0
    assert torch.isclose(metric.compute().float(), torch.tensor(expected).float()), "Test case 5 failed"

    # Test case 6: Multiclass classification with NaN values
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([0, 0, 0, 0])  # class 1 not predicted
    metric = BalancedAccuracy(num_classes=2, task='multiclass')
    metric.update(y_pred, y_true)
    expected = 0.5
    assert torch.isclose(metric.compute().float(), torch.tensor(expected).float()), f"Test case 6 failed with result {metric.compute()}"

    # Test case 7: Multiclass classification with a complex case of missing classes
    y_true = torch.tensor([0, 1, 2, 3, 4, 5])
    y_pred = torch.tensor([0, 0, 2, 2, 4, 5])  # missing classes 1 and 3 in y_pred
    metric = BalancedAccuracy(num_classes=6, task='multiclass')
    metric.update(y_pred, y_true)
    expected = 0.6666666666666666
    assert torch.isclose(metric.compute().float(), torch.tensor(expected).float()), f"Test case 7 failed with result {metric.compute()}"

    # Test case 8: Multiclass classification where y_true has classes not in y_pred
    y_true = torch.tensor([0, 1, 2, 3, 4, 5])
    y_pred = torch.tensor([0, 0, 2, 3, 4, 4])  # class 1 and 5 not predicted
    metric = BalancedAccuracy(num_classes=6, task='multiclass')
    metric.update(y_pred, y_true)
    expected = 0.6666666666666666
    assert torch.isclose(metric.compute().float(), torch.tensor(expected).float()), f"Test case 8 failed with result {metric.compute()}"

    # Test case 9: Complex case with missing classes
    y_true = torch.tensor([4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3])
    y_pred = torch.tensor([3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3])
    metric = BalancedAccuracy(num_classes=6, task='multiclass')
    metric.update(y_pred, y_true)
    expected = 0.6666666666666666
    assert torch.isclose(metric.compute().float(), torch.tensor(expected).float()), f"Test case 9 failed with result {metric.compute()}"