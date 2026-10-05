import pytest
import torch

from minerva.models.nets.time_series.biodiffusion.unet1d import Unet1D_cls_free


@pytest.fixture
def small_bio_unet():
    # Only the three core parameters are stored by get_init_config().
    return Unet1D_cls_free(dim=8, num_classes=3, channels=2)


def test_biodiffusion_unet_forward(small_bio_unet):
    x = torch.rand(2, 2, 16)
    timesteps = torch.tensor([0, 1])

    output = small_bio_unet(x, timesteps)

    assert output.shape == x.shape


def test_biodiffusion_unet_forward_with_labels(small_bio_unet):
    x = torch.rand(2, 2, 16)
    timesteps = torch.tensor([0, 1])
    labels = torch.tensor([0, 2])

    output = small_bio_unet(x, timesteps, labels)

    assert output.shape == x.shape


def test_biodiffusion_unet_init_config(small_bio_unet):
    config = small_bio_unet.get_init_config()

    assert config == {"dim": 8, "num_classes": 3, "channels": 2}


def test_biodiffusion_dropping_all_labels_matches_unconditional_output(small_bio_unet):
    x = torch.rand(2, 2, 16)
    timesteps = torch.tensor([0, 1])
    labels = torch.tensor([0, 2])

    dropped = small_bio_unet(x, timesteps, labels, cond_drop_prob=1)
    unconditional = small_bio_unet(x, timesteps)

    torch.testing.assert_close(dropped, unconditional)


def test_biodiffusion_zero_guidance_returns_unconditional_output(small_bio_unet):
    x = torch.rand(2, 2, 16)
    timesteps = torch.tensor([0, 1])
    labels = torch.tensor([0, 2])

    output = small_bio_unet.forward_with_cond_scale(x, timesteps, labels, cond_scale=0)
    unconditional = small_bio_unet(x, timesteps)

    torch.testing.assert_close(output, unconditional)


def test_biodiffusion_unet_backward_reaches_input(small_bio_unet):
    x = torch.rand(2, 2, 16, requires_grad=True)
    timesteps = torch.tensor([0, 1])

    small_bio_unet(x, timesteps).square().mean().backward()

    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0

def test_biodiffusion_full_drop_probability_runs():
    model = Unet1D_cls_free(dim=8, num_classes=3, channels=2, cond_drop_prob=1)
    x = torch.rand(2, 2, 16)
    timesteps = torch.tensor([0, 1])
    labels = torch.tensor([0, 2])
    
    output = model(
        x,
        timesteps,
        labels,
        cond_drop_prob=1.0,
    )

    assert output.shape == x.shape

def test_biodiffusion_without_condition_dropout():
    model = Unet1D_cls_free(dim=8, num_classes=3, channels=2, cond_drop_prob=1)
    x = torch.rand(2, 2, 16)
    timesteps = torch.tensor([0, 1])
    labels = torch.tensor([0, 2])
    
    output = model(
        x,
        timesteps,
        labels,
        cond_drop_prob=0.0,
    )

    assert output.shape == x.shape