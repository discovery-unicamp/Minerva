import pytest
import torch

from minerva.schedulers.diffusionts_lr_sch import (
    ReduceLROnPlateauWithWarmup,
)


@pytest.fixture
def optimizer():
    model = torch.nn.Linear(2, 1)
    return torch.optim.SGD(model.parameters(), lr=0.1)


def test_plateau_warmup_increases_learning_rate(optimizer):
    scheduler = ReduceLROnPlateauWithWarmup(optimizer, warmup=2, warmup_lr=0.3)

    scheduler.step(1.0)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2)


def test_plateau_reduces_learning_rate_when_loss_stops_improving(optimizer):
    scheduler = ReduceLROnPlateauWithWarmup(optimizer, patience=1, factor=0.5)

    scheduler.step(1.0)
    scheduler.step(1.0)
    scheduler.step(1.0)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_plateau_keeps_learning_rate_when_loss_improves(optimizer):
    scheduler = ReduceLROnPlateauWithWarmup(optimizer, patience=0)

    scheduler.step(1.0)
    scheduler.step(0.5)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_plateau_respects_minimum_learning_rate(optimizer):
    scheduler = ReduceLROnPlateauWithWarmup(
        optimizer, patience=0, factor=0.1, min_lr=0.05
    )

    scheduler.step(1.0)
    scheduler.step(1.0)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_plateau_rejects_invalid_factor(optimizer):
    with pytest.raises(ValueError, match="Factor"):
        ReduceLROnPlateauWithWarmup(optimizer, factor=1.0)
