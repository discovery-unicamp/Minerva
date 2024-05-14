import pytest
import torch

from minerva.models.nets.setr import SETR_PUP


def test_setr_loss():
    model = SETR_PUP(image_size=16)
    batch_size = 2
    x = torch.rand(2, 3, 16, 16)
    mask = torch.rand(2, 1, 16, 16).long()

    # Do the training step
    loss = model.training_step((x, mask), 0).item()
    assert loss is not None
    assert loss >= 0, f"Expected non-negative loss, but got {loss}"


def test_setr_predict():
    model = SETR_PUP(image_size=16)
    batch_size = 2
    mask_shape = (batch_size, 1000, 16, 16)  # (2, 1, 500, 500)
    x = torch.rand(2, 3, 16, 16)
    mask = torch.rand(2, 1, 16, 16).long()

    # Do the prediction step
    preds = model.predict_step((x, mask), 0)
    assert preds is not None
    assert (
        preds.shape == mask_shape
    ), f"Expected shape {mask_shape}, but got {preds[0].shape}"


if __name__ == "__main__":
    test_setr_loss()
    test_setr_predict()
