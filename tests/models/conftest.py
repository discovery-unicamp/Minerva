import torchmetrics
import pytest

@pytest.fixture
def simple_torchmetrics():
    return dict(
        train_metrics={"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)},
        val_metrics={"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)},
        test_metrics={"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)},
    )