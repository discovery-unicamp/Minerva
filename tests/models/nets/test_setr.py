import pytest
import torch

from minerva.models.nets.setr import SETR_PUP


def test_wisenet_loss():
    model = SETR_PUP()
    batch_size = 2
    x = torch.rand(2, 3, 512, 512)
    mask = torch.rand(2, 1, 512, 512).long()

    # Do the training step
    loss = model.training_step((x, mask), 0).item()
    assert loss is not None
    assert loss >= 0, f"Expected non-negative loss, but got {loss}"


def test_wisenet_predict():
    model = SETR_PUP()
    batch_size = 2
    mask_shape = (batch_size, 1000, 512, 512)  # (2, 1, 500, 500)
    x = torch.rand(2, 3, 512, 512)
    mask = torch.rand(2, 1, 512, 512).long()

    # Do the prediction step
    preds = model.predict_step((x, mask), 0)
    assert preds is not None
    assert (
        preds[0].shape == mask_shape
    ), f"Expected shape {mask_shape}, but got {preds[0].shape}"
