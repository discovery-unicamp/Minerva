import lightning as L
import torch
import torchmetrics

from minerva.models.nets import ResNet
from minerva.utils.data import RandomDataModule


def test_resnet50():
    # Test the class instantiation
    model = ResNet(type="50", img_channel=3, num_classes=1000)
    assert model is not None

    # Generate a random input tensor (B, C, H, W)
    input_shape = (2, 3, 224, 224)
    x = torch.rand(*input_shape)

    # Test the forward method
    output = model(x)
    expected_output_shape = (2, 1000)  # For classification, output matches num_classes

    print(output.shape == expected_output_shape, output.shape)
    assert (
        output.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output.shape}"

    # Test the training step
    target = torch.rand(expected_output_shape)
    loss = model.training_step((x, target), 0).item()
    assert loss is not None
    assert loss >= 0, f"Expected non-negative loss, but got {loss}"

def test_resnet101():
    # Test the class instantiation
    model = ResNet(type="101", img_channel=3, num_classes=1000)
    assert model is not None

    # Generate a random input tensor (B, C, H, W)
    input_shape = (2, 3, 224, 224)
    x = torch.rand(*input_shape)

    # Test the forward method
    output = model(x)
    expected_output_shape = (2, 1000)  # For classification, output matches num_classes

    print(output.shape == expected_output_shape, output.shape)
    assert (
        output.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output.shape}"

    # Test the training step
    target = torch.rand(expected_output_shape)
    loss = model.training_step((x, target), 0).item()
    assert loss is not None
    assert loss >= 0, f"Expected non-negative loss, but got {loss}"

def test_resnet152():
    # Test the class instantiation
    model = ResNet(type="101", img_channel=3, num_classes=1000)
    assert model is not None

    # Generate a random input tensor (B, C, H, W)
    input_shape = (3, 3, 224, 224)
    x = torch.rand(*input_shape)

    # Test the forward method
    output = model(x)
    expected_output_shape = (3, 1000)  # For classification, output matches num_classes

    print(output.shape == expected_output_shape, output.shape)
    assert (
        output.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output.shape}"

    # Test the training step
    target = torch.rand(expected_output_shape)
    loss = model.training_step((x, target), 0).item()
    assert loss is not None
    assert loss >= 0, f"Expected non-negative loss, but got {loss}"

def test_resnet50_train_metrics():
    metrics = {
        "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=1000),
        "top_k_accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=1000, top_k=5),
    }

    data_module = RandomDataModule(
        data_shape=(3, 224, 224),
        label_shape=None,
        num_classes=1000,
        num_train_samples=10,
        batch_size=2,
    )

    model = ResNet(
        type="50",
        img_channel=3,
        num_classes=1000,
        train_metrics=metrics,
    )
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, devices=1)

    # Run training
    trainer.fit(model, data_module)

    assert "train_accuracy" in trainer.logged_metrics
    assert "train_top_k_accuracy" in trainer.logged_metrics
    assert "train_loss" in trainer.logged_metrics

def test_resnet101_train_metrics():
    metrics = {
        "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=1000),
        "top_k_accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=1000, top_k=5),
    }

    data_module = RandomDataModule(
        data_shape=(3, 500, 500),
        label_shape=None,
        num_classes=1000,
        num_train_samples=10,
        batch_size=3,
    )

    model = ResNet(
        type="101",
        img_channel=3,
        num_classes=1000,
        train_metrics=metrics,
    )
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, devices=1)

    # Run training
    trainer.fit(model, data_module)

    assert "train_accuracy" in trainer.logged_metrics
    assert "train_top_k_accuracy" in trainer.logged_metrics
    assert "train_loss" in trainer.logged_metrics

def test_resnet152_train_metrics():
    metrics = {
        "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=1000),
        "top_k_accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=1000, top_k=5),
    }

    data_module = RandomDataModule(
        data_shape=(3, 768, 768),
        label_shape=None,
        num_classes=1000,
        num_train_samples=10,
        batch_size=2,
    )

    model = ResNet(
        type="152",
        img_channel=3,
        num_classes=1000,
        train_metrics=metrics,
    )
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, devices=1)

    # Run training
    trainer.fit(model, data_module)

    assert "train_accuracy" in trainer.logged_metrics
    assert "train_top_k_accuracy" in trainer.logged_metrics
    assert "train_loss" in trainer.logged_metrics
