import pytest
import torch
import torch.nn as nn
from minerva.models.ssl.barlowtwins import BarlowTwins


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
def barlowtwins_model(dummy_backbone, dummy_projection_head):
    return BarlowTwins(
        backbone=dummy_backbone,
        projection_head=dummy_projection_head,
        learning_rate=0.0001,
        weight_decay=1e-6,
    )


def test_barlowtwins_forward(barlowtwins_model):
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    projections = barlowtwins_model(x)
    assert projections.shape == (batch_size, 64)


def test_training_step(barlowtwins_model):
    batch = ((torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)), torch.zeros(4))
    loss = barlowtwins_model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)


def test_configure_optimizers(barlowtwins_model):
    optimizer = barlowtwins_model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[0]["weight_decay"] == 1e-6
