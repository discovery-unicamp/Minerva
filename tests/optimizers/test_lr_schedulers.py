import pytest
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import _LRScheduler

from minerva.optimizers.lr_schedulers import PolyLRScheduler


@pytest.fixture
def optimizer():
    model = torch.nn.Linear(2, 1)
    return SGD(model.parameters(), lr=0.1)


def test_initial_lr(optimizer):
    scheduler = PolyLRScheduler(optimizer, max_iter=10)
    lrs = scheduler.get_last_lr()
    assert pytest.approx(lrs[0]) == 0.1  # initial lr must match optimizer lr


def test_lr_decay(optimizer):
    scheduler = PolyLRScheduler(optimizer, max_iter=10, power=1.0)
    scheduler.step()  # epoch=0 (returns base lr)
    initial_lr = scheduler.get_last_lr()[0]
    scheduler.step()  # epoch=1 (returns base lr)
    initial_lr = scheduler.get_last_lr()[0]
    scheduler.step()  # epoch=2 (decayed)
    new_lr = scheduler.get_last_lr()[0]

    assert new_lr < initial_lr
    assert scheduler.min_lr <= new_lr <= 0.1


def test_min_lr_clamp():
    model = torch.nn.Linear(2, 1)
    optimizer = SGD(model.parameters(), lr=0.1)

    # Required for resume with last_epoch >= 0
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    scheduler = PolyLRScheduler(
        optimizer, max_iter=2, power=2.0, min_lr=0.05, last_epoch=0
    )
    lrs = []
    for _ in range(5):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    # Invariants we care about
    assert all(lr >= 0.05 for lr in lrs)
    assert 0.05 in [round(lr, 5) for lr in lrs]


def test_multiple_base_lrs():
    params = [
        torch.nn.Parameter(torch.randn(2, 2, requires_grad=True)),
        torch.nn.Parameter(torch.randn(2, 2, requires_grad=True)),
    ]
    optimizer = SGD(
        [
            {"params": [params[0]], "lr": 0.1},
            {"params": [params[1]], "lr": 0.01},
        ]
    )
    scheduler = PolyLRScheduler(optimizer, max_iter=5, power=1.0, min_lr=0.001)

    lrs_history = []
    for _ in range(6):
        scheduler.step()
        lrs_history.append(scheduler.get_last_lr())

    # Both parameter groups decayed, never below min_lr
    for lrs in lrs_history:
        assert len(lrs) == 2
        assert all(lr >= 0.001 for lr in lrs)


def test_last_epoch_resume():
    model = torch.nn.Linear(2, 1)
    optimizer = SGD(model.parameters(), lr=0.1)

    # Required for resume with last_epoch >= 0
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    scheduler = PolyLRScheduler(optimizer, max_iter=10, last_epoch=5)

    # At construction, should compute LR as if at epoch=5
    lr_at_5 = scheduler.get_last_lr()[0]
    scheduler.step()  # move to epoch=6
    lr_at_6 = scheduler.get_last_lr()[0]

    assert lr_at_6 < lr_at_5
