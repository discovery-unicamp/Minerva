import pytest
import torch
import torch.nn as nn
from minerva.models.ssl.simsiam import SimSiam
from unittest.mock import MagicMock


@pytest.fixture
def dummy_backbone():
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1),
    )


@pytest.fixture
def dummy_projection_head():
    return nn.Sequential(nn.Linear(16, 64))


@pytest.fixture
def dummy_prediction_head():
    return nn.Sequential(nn.Linear(64, 64))


@pytest.fixture
def simsiam_model(dummy_backbone, dummy_projection_head, dummy_prediction_head):
    model = SimSiam(
        backbone=dummy_backbone,
        projection_head=dummy_projection_head,
        prediction_head=dummy_prediction_head,
        learning_rate=0.0001,
        weight_decay=1e-6,
    )
    model.log = MagicMock()  # Mock the logger
    return model


def test_simsiam_forward(simsiam_model):
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    z, p = simsiam_model(x)
    assert z.shape == (batch_size, 64)
    assert p.shape == (batch_size, 64)


def test_training_step(simsiam_model):
    batch = ((torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)), torch.zeros(4))
    loss = simsiam_model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)


def test_configure_optimizers(simsiam_model):
    optimizer = simsiam_model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[0]["weight_decay"] == 1e-6
