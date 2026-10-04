import pytest
import torch

from minerva.models.nets.time_series.diffusionts.diffusionts_transformer import (
    Transformer,
)
from minerva.models.ssl.diffusionts import DiffusionTS


@pytest.fixture
def small_transformer():
    return Transformer(
        n_feat=2,
        n_channel=16,
        n_layer_enc=1,
        n_layer_dec=1,
        n_embd=8,
        n_heads=2,
        max_len=16,
        attn_pdrop=0,
        resid_pdrop=0,
    )


@pytest.fixture
def diffusionts(small_transformer):
    return DiffusionTS(small_transformer, seq_length=16, feature_size=2, timesteps=4)


def test_diffusionts_training_loss(diffusionts):
    x = torch.rand(2, 16, 2)

    loss = diffusionts(x)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_diffusionts_noise_can_be_removed(diffusionts):
    x = torch.rand(2, 16, 2)
    noise = torch.randn_like(x)
    timesteps = torch.tensor([0, 1])
    noisy = diffusionts.q_sample(x, timesteps, noise)

    restored = diffusionts.predict_start_from_noise(noisy, timesteps, noise)

    torch.testing.assert_close(restored, x)


def test_diffusionts_invalid_channel_count(diffusionts):
    x = torch.rand(2, 16, 3)

    with pytest.raises(AssertionError, match="number of variable"):
        diffusionts(x)


@pytest.mark.parametrize("sampling_steps", [2, 4])
def test_diffusionts_generate_samples(small_transformer, sampling_steps):
    model = DiffusionTS(
        small_transformer,
        seq_length=16,
        feature_size=2,
        timesteps=4,
        sampling_timesteps=sampling_steps,
    )

    samples = model.generate_mts(batch_size=2)

    assert samples.shape == (2, 16, 2)
    assert torch.isfinite(samples).all()


def test_diffusionts_accumulates_two_batches_per_step(diffusionts):
    diffusionts.configure_optimizers()
    batch = torch.rand(2, 16, 2)

    diffusionts.training_step(batch, 0)
    assert diffusionts.step_counter == 0

    diffusionts.training_step(batch, 1)
    assert diffusionts.step_counter == 1
