"""Functional API for losses."""

import torch


# Borrowed from https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/_functional.py
def dice_score(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert y_hat.size() == y.size()
    if dims is not None:
        intersection = torch.sum(y_hat * y, dim=dims)
        cardinality = torch.sum(y_hat + y, dim=dims)
    else:
        intersection = torch.sum(y_hat * y)
        cardinality = torch.sum(y_hat + y)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score
