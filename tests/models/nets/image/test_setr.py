import warnings

import pytest
import torch
from torchinfo import summary

from minerva.models.nets import SETR_PUP


def test_setr_loss():
    model = SETR_PUP(img_size=(16, 16))
    batch_size = 2
    x = torch.rand(batch_size, 3, 16, 16)
    mask = torch.rand(batch_size, 1, 16, 16).long()

    # Do the training step
    loss = model.training_step((x, mask), 0).item()
    assert loss is not None
    assert loss >= 0, f"Expected non-negative loss, but got {loss}"


def test_setr_predict():
    model = SETR_PUP(img_size=(16, 16))
    batch_size = 2
    mask_shape = (batch_size, 6, 16, 16)
    x = torch.rand(batch_size, 3, 16, 16)
    mask = torch.rand(batch_size, 1, 16, 16).long()

    # Do the prediction step
    preds = model.predict_step((x, mask), 0)
    assert preds is not None
    assert (
        preds.shape == mask_shape
    ), f"Expected shape {mask_shape}, but got {preds.shape}"


def test_eval_step_with_slide(monkeypatch):
    batch_size = 2
    height, width = 8, 8

    model = SETR_PUP(img_size=(height, width), use_sliding_inference=True)
    model = model.cpu()  # move model to CPU

    # Mock slide inference to return a constant prediction
    def fake_slide_inference(img_np, crop_size, stride, ori_shape):
        return torch.ones(ori_shape, dtype=torch.uint8).numpy()

    monkeypatch.setattr(model, "_slide_inference", fake_slide_inference)

    # Mock metric and log
    dummy_metric = torch.nn.Module()
    dummy_metric.update = lambda preds, gt: None
    model.val_metrics = {"dummy_metric": dummy_metric}
    model.test_metrics = {"dummy_metric": dummy_metric}
    model.log = lambda *args, **kwargs: None

    img = torch.rand(batch_size, 3, height, width)
    gt = torch.zeros(batch_size, 1, height, width, dtype=torch.long)

    # Run for both val and test
    for step_name in ["val", "test"]:
        metrics_before = getattr(model, f"{step_name}_metrics")
        model._eval_step_with_slide((img, gt), step_name)
        metrics_after = getattr(model, f"{step_name}_metrics")
        assert metrics_before is metrics_after  # metrics dict unchanged
