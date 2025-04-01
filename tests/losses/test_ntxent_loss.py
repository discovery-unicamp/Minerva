import pytest
import torch

from minerva.losses.xtent_loss import NTXentLoss


def test_ntxent_loss():
    ntxent_loss = NTXentLoss(temperature=0.5)
    x = torch.rand(10, 32)
    loss = ntxent_loss(x, x)
    assert loss is not None
    assert loss.item() is not None


def test_invalid_temperature():
    with pytest.raises(ValueError):
        NTXentLoss(temperature=0.0)


def test_temp_lower_esp():
    with pytest.raises(ValueError):
        NTXentLoss(temperature=1e-9)


def test_forward():
    ntxent_loss = NTXentLoss(temperature=0.0001)
    y_0 = torch.eye(4, 128)
    y_1 = y_0.clone()
    loss = ntxent_loss(y_0, y_1)
    assert loss is not None
    assert loss.item() is not None
    assert loss.item() == pytest.approx(0.0)
