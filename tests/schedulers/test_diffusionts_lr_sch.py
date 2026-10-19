import pytest
import torch
from copy import deepcopy

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


def test_plateau_restores_warmup_without_changing_learning_rate_steps(optimizer):
    scheduler = ReduceLROnPlateauWithWarmup(optimizer, warmup=4, warmup_lr=0.5)
    scheduler.step(1.0)
    scheduler.step(1.0)
    optimizer_state = deepcopy(optimizer.state_dict())
    scheduler_state = deepcopy(scheduler.state_dict())
    restored_optimizer = torch.optim.SGD(torch.nn.Linear(2, 1).parameters(), lr=0.1)
    restored_scheduler = ReduceLROnPlateauWithWarmup(
        restored_optimizer, warmup=4, warmup_lr=0.5
    )

    restored_optimizer.load_state_dict(optimizer_state)
    restored_scheduler.load_state_dict(scheduler_state)
    scheduler.step(1.0)
    restored_scheduler.step(1.0)

    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )