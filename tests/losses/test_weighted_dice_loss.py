import pytest
import torch
import torch.nn.functional as F
from minerva.losses.weighted_dice_loss import WeightedDiceLoss, BinaryWeightedDiceLoss


def test_dice_loss_perfect_complete():
    num_classes = 4
    H, W = 4, 4
    target = torch.tensor(
        [[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]]
    )  # shape: (1, 4, 4)

    # One-hot → logits very high for the correct class
    pred = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float() * 1000

    loss_fn = WeightedDiceLoss(num_classes=num_classes, smooth=1e-6)
    loss = loss_fn(pred, target)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_dice_loss_total_wrong_prediction():
    num_classes = 4
    H, W = 4, 4
    target = torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]])

    pred = torch.zeros(1, num_classes, H, W)

    # map incorrectly: 0→1, 1→2, 2→3, 3→0
    pred[:, 1, 0:2, 0:2] = 1000.0
    pred[:, 2, 0:2, 2:4] = 1000.0
    pred[:, 3, 2:4, 0:2] = 1000.0
    pred[:, 0, 2:4, 2:4] = 1000.0

    loss_fn = WeightedDiceLoss(num_classes=num_classes, smooth=1.0)
    loss = loss_fn(pred, target)

    assert loss.item() > 0.6


def test_dice_loss_missing_class():
    num_classes = 4
    H, W = 4, 4

    target = torch.zeros(1, H, W).long()  # only class 0
    pred = torch.zeros(1, num_classes, H, W)
    pred[:, 0, :, :] = 1000.0  # correct prediction for class 0

    loss_fn = WeightedDiceLoss(num_classes=num_classes, smooth=1.0)
    loss = loss_fn(pred, target)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_binary_dice_loss_perfect_match():
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])  # shape (1,1,2,2)
    pred_logits = torch.tensor(
        [[[[-100.0, 100.0], [100.0, -100.0]]]]
    )  # high logits, inverted for sigmoid

    loss_fn = BinaryWeightedDiceLoss(smooth=1e-6)
    loss = loss_fn(pred_logits, target)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_binary_dice_loss_complete_mismatch():
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    pred_logits = torch.tensor([[[[100.0, -100.0], [-100.0, 100.0]]]])

    loss_fn = BinaryWeightedDiceLoss(smooth=1.0)
    loss = loss_fn(pred_logits, target)

    assert loss.item() > 0.6


def test_binary_dice_loss_no_foreground():
    target = torch.zeros((1, 1, 2, 2))
    pred_logits = torch.zeros((1, 1, 2, 2))  # any value

    loss_fn = BinaryWeightedDiceLoss(smooth=1.0)
    loss = loss_fn(pred_logits, target)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)
