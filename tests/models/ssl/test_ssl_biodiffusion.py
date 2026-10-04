import pytest
import torch

from minerva.models.nets.time_series.biodiffusion.unet1d import Unet1D_cls_free
from minerva.models.ssl.biodiffusion import BioDiffusion


@pytest.fixture
def small_bio_unet():
    # Only the three core parameters are stored by get_init_config().
    return Unet1D_cls_free(dim=8, num_classes=3, channels=2)


def test_biodiffusion_forward(small_bio_unet):
    model = BioDiffusion(small_bio_unet, noise_steps=4)
    x = torch.rand(2, 2, 16)
    timesteps = torch.tensor([0, 1])

    output = model(x, timesteps)

    assert output.shape == x.shape


def test_biodiffusion_training_loss(small_bio_unet):
    model = BioDiffusion(small_bio_unet, noise_steps=4)
    x = torch.rand(2, 2, 16)
    labels = torch.tensor([[0], [1]])

    loss = model.training_step((x, labels))

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_biodiffusion_sample(small_bio_unet):
    model = BioDiffusion(small_bio_unet, noise_steps=4, channels=2, n_timesteps=16)

    samples = model.sample(batch_size=2, activity=-1)

    assert samples.shape == (2, 2, 16)
    assert torch.isfinite(samples).all()


def test_biodiffusion_padding_preserves_input(small_bio_unet):
    model = BioDiffusion(small_bio_unet, signal_padding=3)
    x = torch.rand(2, 2, 16)

    result = model.unpad_signal(model.pad_signal(x))

    torch.testing.assert_close(result, x)


def test_biodiffusion_config_round_trip(small_bio_unet):
    model = BioDiffusion(small_bio_unet, noise_steps=4, channels=2)
    restored = BioDiffusion(**model.get_init_config())
    restored.load_state_dict(model.state_dict())
    x = torch.rand(2, 2, 16)
    timesteps = torch.tensor([0, 1])

    torch.testing.assert_close(restored(x, timesteps), model(x, timesteps))
