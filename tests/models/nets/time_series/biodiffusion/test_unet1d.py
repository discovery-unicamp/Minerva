from unittest.mock import Mock

import pytest
import torch

from minerva.models.nets.time_series.biodiffusion.unet1d import Unet1D_cls_free


@pytest.fixture
def small_bio_unet():
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

    assert config == {
        "dim": 8,
        "num_classes": 3,
        "cond_drop_prob": 0.5,
        "init_dim": None,
        "out_dim": None,
        "dim_mults": (1, 2, 4, 8),
        "channels": 2,
        "resnet_block_groups": 8,
        "learned_variance": False,
        "learned_sinusoidal_cond": False,
        "random_fourier_features": False,
        "learned_sinusoidal_dim": 16,
        "n_timesteps": 100,
    }


def test_biodiffusion_unet_custom_config_round_trip():
    dim_mults = [1, 2]
    model = Unet1D_cls_free(
        dim=8,
        num_classes=3,
        cond_drop_prob=0.25,
        init_dim=8,
        out_dim=2,
        dim_mults=dim_mults,
        channels=2,
        resnet_block_groups=4,
        learned_variance=True,
        learned_sinusoidal_cond=True,
        random_fourier_features=True,
        learned_sinusoidal_dim=8,
        n_timesteps=32,
    )
    config = model.get_init_config()
    assert config["dim_mults"] == [1, 2]
    assert config["cond_drop_prob"] == 0.25
    assert config["learned_variance"] is True
    assert config["random_fourier_features"] is True
    assert config["n_timesteps"] == 32

    dim_mults.append(4)
    config["dim_mults"].append(8)
    assert model.get_init_config()["dim_mults"] == [1, 2]

    restored = Unet1D_cls_free(**model.get_init_config())
    restored.load_state_dict(model.state_dict())
    x = torch.rand(2, 2, 16)
    timesteps = torch.tensor([0, 1])
    labels = torch.tensor([0, 2])
    torch.testing.assert_close(
        restored(x, timesteps, labels, cond_drop_prob=0),
        model(x, timesteps, labels, cond_drop_prob=0),
    )


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
