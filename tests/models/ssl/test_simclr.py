import pytest
import torch
import torch.nn as nn
from torch import optim
from torch.nn.functional import normalize

from minerva.models.ssl.simclr import SimCLR
from minerva.optimizers.lars import LARS


@pytest.fixture
def dummy_backbone():
    return nn.Sequential(nn.Conv2d(3, 16, kernel_size=3), nn.AdaptiveAvgPool2d((1, 1)))


@pytest.fixture
def dummy_projection_head():
    return nn.Sequential(nn.Linear(16, 64))


@pytest.fixture
def simclr_model(dummy_backbone, dummy_projection_head):
    return SimCLR(
        backbone=dummy_backbone,
        projection_head=dummy_projection_head,
        temperature=0.5,
        lr=1e-3,
    )


def test_simclr_forward(simclr_model):
    batch_size = 4
    x = (torch.randn(batch_size, 3, 32, 32), torch.randn(batch_size, 3, 32, 32))
    projections = simclr_model(x)
    assert projections[0].shape == (batch_size, 64) and projections[1].shape == (
        batch_size,
        64,
    )


def test_training_step(simclr_model):
    batch = ((torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)), torch.zeros(4))
    loss = simclr_model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)


def test_configure_optimizers(simclr_model):
    optimizer = simclr_model.configure_optimizers()
    assert isinstance(optimizer, LARS)
    assert optimizer.param_groups[0]["lr"] == 1e-3


def test_predict_step(simclr_model):
    batch = ((torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)), None)
    predictions = simclr_model.predict_step(batch, 0)
    assert predictions[0].shape == (4, 64) and predictions[1].shape == (4, 64)
