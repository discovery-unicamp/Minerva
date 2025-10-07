import random
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
import torchmetrics
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from minerva.analysis.metrics import BalancedAccuracy
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)


def generate_logits_with_n_correct(num_classes: int, size: int, num_correct: int):
    """
    Generate logits and true labels for a classification task, ensuring a specific number of correct predictions.
    """
    if not (0 <= num_correct <= size):
        raise ValueError("num_correct must be between 0 and size.")

    y_true = np.random.randint(0, num_classes, size)
    logits = np.random.randn(size, num_classes)
    correct_indices = random.sample(range(size), num_correct)

    for i in range(size):
        true_class = y_true[i]
        if i in correct_indices:
            logits[i] = np.random.randn(num_classes)
            logits[i][true_class] = max(logits[i]) + 1.0
        else:
            logits[i] = np.random.randn(num_classes)
            if np.argmax(logits[i]) == true_class:
                other_classes = [j for j in range(num_classes) if j != true_class]
                swap_class = random.choice(other_classes)
                logits[i][true_class], logits[i][swap_class] = (
                    logits[i][swap_class],
                    logits[i][true_class],
                )

    return y_true, logits


class MyModel(L.LightningModule):
    def __init__(self, logits: list, batch_size):
        super().__init__()
        self.logits = torch.Tensor(logits)
        self.batch_size = batch_size
        self._i = 0

    def forward(self, *args, **kwargs):
        if self._i + self.batch_size > len(self.logits):
            stride = self._i + self.batch_size - len(self.logits) + 1
        else:
            stride = self.batch_size
        logs = self.logits[self._i : self._i + stride]
        self._i += stride
        if self._i >= len(self.logits):
            self._i = 0
        return logs


class MyDataModule(L.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )


# Separate parametrize decorators = Cartesian product
@pytest.mark.parametrize("batch_size", [1, 2, 7])
@pytest.mark.parametrize("num_samples", [10, 30])
@pytest.mark.parametrize("num_correct", [0, 1, 10])
@pytest.mark.parametrize("num_classes", [4, 6])
def test_accuracy_pipeline(batch_size, num_samples, num_correct, num_classes):
    y_true, logits = generate_logits_with_n_correct(
        num_classes, num_samples, num_correct
    )

    x = torch.arange(0, len(logits))  # Not used, but needed for the DataLoader
    y_true_tensor = torch.Tensor(y_true)
    logits = torch.Tensor(logits)
    y_pred = np.argmax(logits.numpy(), axis=1)

    dataset = TensorDataset(x, y_true_tensor)
    model = MyModel(logits.numpy().tolist(), batch_size=batch_size)
    dm = MyDataModule(dataset, batch_size=batch_size)
    trainer = L.Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
    )

    pipeline = SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        save_run_status=False,
        classification_metrics={
            "accuracy": torchmetrics.Accuracy(
                num_classes=num_classes, task="multiclass", average="micro"
            ),
            "f1-macro": torchmetrics.F1Score(
                num_classes=num_classes, average="macro", task="multiclass"
            ),
            "f1-micro": torchmetrics.F1Score(
                num_classes=num_classes, average="micro", task="multiclass"
            ),
            "precision": torchmetrics.Precision(
                num_classes=num_classes, average="macro", task="multiclass"
            ),
            "recall": torchmetrics.Recall(
                num_classes=num_classes, average="macro", task="multiclass"
            ),
            "balanced_accuracy": BalancedAccuracy(
                num_classes=num_classes, task="multiclass"
            ),
        },
        apply_metrics_per_sample=False,
    )

    r = pipeline.run(
        task="evaluate",
        data=dm,
    )

    expected_acc = num_correct / num_samples
    expected_acc_sklearn = accuracy_score(y_true, y_pred)
    expected_f1_macro_sklearn = f1_score(y_true, y_pred, average="macro")
    expected_f1_micro_sklearn = f1_score(y_true, y_pred, average="micro")
    expected_precision_sklearn = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    expected_recall_sklearn = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    expected_balanced_acc_sklearn = balanced_accuracy_score(y_true, y_pred)

    np.testing.assert_almost_equal(
        np.mean(r["classification"]["accuracy"][0]), expected_acc
    )
    np.testing.assert_almost_equal(
        np.mean(r["classification"]["accuracy"][0]), expected_acc_sklearn
    )
    np.testing.assert_almost_equal(
        np.mean(r["classification"]["f1-macro"][0]), expected_f1_macro_sklearn
    )
    np.testing.assert_almost_equal(
        np.mean(r["classification"]["f1-micro"][0]), expected_f1_micro_sklearn
    )
    np.testing.assert_almost_equal(
        np.mean(r["classification"]["precision"][0]), expected_precision_sklearn
    )
    np.testing.assert_almost_equal(
        np.mean(r["classification"]["recall"][0]), expected_recall_sklearn
    )
    np.testing.assert_almost_equal(
        np.mean(r["classification"]["balanced_accuracy"][0]),
        expected_balanced_acc_sklearn,
    )


@pytest.mark.parametrize("batch_size", [1, 2, 7])
@pytest.mark.parametrize("num_samples", [10, 30])
@pytest.mark.parametrize("num_correct", [0, 1, 10])
@pytest.mark.parametrize("num_classes", [4, 6])
def test_accuracy_pipeline_per_sample(
    batch_size, num_samples, num_correct, num_classes
):
    y_true, logits = generate_logits_with_n_correct(
        num_classes, num_samples, num_correct
    )

    x = torch.arange(0, len(logits))
    y_true_tensor = torch.Tensor(y_true)
    logits_tensor = torch.Tensor(logits)
    dataset = TensorDataset(x, y_true_tensor)

    model = MyModel(logits_tensor.numpy().tolist(), batch_size=batch_size)
    datamodule = MyDataModule(dataset, batch_size=batch_size)

    trainer = L.Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
    )

    pipeline = SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        save_run_status=False,
        classification_metrics={
            "accuracy": torchmetrics.Accuracy(
                num_classes=num_classes, task="multiclass"
            ),
            "f1-macro": torchmetrics.F1Score(
                num_classes=num_classes, average="macro", task="multiclass"
            ),
            "f1-micro": torchmetrics.F1Score(
                num_classes=num_classes, average="micro", task="multiclass"
            ),
            "precision": torchmetrics.Precision(
                num_classes=num_classes, average="macro", task="multiclass"
            ),
            "recall": torchmetrics.Recall(
                num_classes=num_classes, average="macro", task="multiclass"
            ),
            "balanced_accuracy": BalancedAccuracy(
                num_classes=num_classes, task="multiclass"
            ),
        },
        apply_metrics_per_sample=True,
    )

    result = pipeline.run(task="evaluate", data=datamodule)

    actual_accuracy = np.mean(result["classification"]["accuracy"])
    expected_accuracy = num_correct / num_samples
    np.testing.assert_almost_equal(actual_accuracy, expected_accuracy, decimal=4)
