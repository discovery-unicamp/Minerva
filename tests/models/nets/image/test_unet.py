import lightning as L
import torch
import torchmetrics

from minerva.models.nets import UNet
from minerva.data.data_module_tools import RandomDataModule


def test_unet():
    # Test the class instantiation
    model = UNet()
    assert model is not None

    # Generate a random input tensor (B, C, H, W) and the random mask of the
    # same shape
    input_shape = (2, 1, 500, 500)
    x = torch.rand(*input_shape)
    mask = torch.rand(*input_shape)

    # Test the forward method
    output = model(x)
    assert (
        output.shape == input_shape
    ), f"Expected output shape {input_shape}, but got {output.shape}"

    # Test the training step
    loss = model.training_step((x, mask), 0).item()
    assert loss is not None
    assert loss >= 0, f"Expected non-negative loss, but got {loss}"


def test_unet_train_metrics():
    metrics = {
        "mse": torchmetrics.MeanSquaredError(squared=True),
        "mae": torchmetrics.MeanAbsoluteError(),
    }

    # Generate a random input tensor (B, C, H, W) and the random mask of the
    # same shape
    data_module = RandomDataModule(
        data_shape=(1, 128, 128),
        label_shape=(1, 128, 128),
        num_train_samples=2,
        batch_size=2,
    )

    # Test the class instantiation
    model = UNet(train_metrics=metrics)
    trainer = L.Trainer(fast_dev_run=True, devices=1, accelerator="cpu", max_epochs=1)

    assert data_module is not None
    assert model is not None
    assert trainer is not None

    # Do fit
    trainer.fit(model, data_module)

    assert "train_mse" in trainer.logged_metrics
    assert "train_mae" in trainer.logged_metrics
    assert "train_loss" in trainer.logged_metrics
