import pytest
import torch

from minerva.models.nets.image.sam import Sam

def test_sam_loss():
    model = Sam()
    batch = [{
        "image": torch.rand(3, 255, 701),
        "label": torch.rand(1, 255, 701).long(),
        "original_size": (255, 701),
        "multimask_output": True,
    }]

    # Do the training step
    loss = model.training_step(batch, 0).item()
    assert loss is not None
    assert loss >= 0, f"Expected non-negative loss, but got {loss}"

def test_sam_predict():
    model = Sam()
    mask_shape = (1, 3, 255, 701)
    batch = [{
        "image": torch.rand(3, 255, 701),
        "label": torch.rand(1, 255, 701).long(),
        "original_size": (255, 701),
        "multimask_output": True,
    }]

    # Do the prediction step
    preds = model.predict_step(batch, 0)
    assert preds is not None
    assert (
        preds[0]['masks_logits'].shape == mask_shape
    ), f"Expected shape {mask_shape}, but got {preds[0]['masks_logits'].shape}"