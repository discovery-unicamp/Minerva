import pytest
import torch
import numpy as np

from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone
from minerva.models.ssl.byol import BYOL


@pytest.fixture
def dummy_input():
    # Dummy input with batch size 2 and 3-channel images
    x = np.random.rand(4, 3, 256, 256)
    return torch.tensor(x, dtype=torch.float32)


@pytest.fixture
def byol_model():
    return BYOL()


def test_forward_output_shape(byol_model, dummy_input):
    out = byol_model.forward(dummy_input)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == dummy_input.shape[0]
    assert out.shape[1] == 256  # Output dimension from BYOLPredictionHead


def test_forward_momentum_output_shape(byol_model, dummy_input):
    out = byol_model.forward_momentum(dummy_input)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == dummy_input.shape[0]
    assert out.shape[1] == 256


def test_forward_and_momentum_are_different(byol_model, dummy_input):
    out_normal = byol_model.forward(dummy_input)
    out_momentum = byol_model.forward_momentum(dummy_input)
    # Outputs shouldn't be exactly the same
    assert not torch.allclose(out_normal, out_momentum, atol=1e-3)


def test_training_step_runs(byol_model, dummy_input):
    x0, x1 = dummy_input[0:2], dummy_input[2:4]
    loss = byol_model.training_step((x0, x1), batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()  # Check for NaN or Inf


def test_cosine_schedule_behaviour(byol_model):
    start = byol_model.cosine_schedule(
        step=0, max_steps=100, start_value=0.5, end_value=1.0
    )
    end = byol_model.cosine_schedule(
        step=99, max_steps=100, start_value=0.5, end_value=1.0
    )

    assert np.isclose(start, 0.5)
    assert np.isclose(end, 1.0)
