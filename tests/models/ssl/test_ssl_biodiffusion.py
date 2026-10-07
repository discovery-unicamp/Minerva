from unittest.mock import patch

import pytest
import torch

from minerva.models.nets.time_series.biodiffusion.unet1d import Unet1D_cls_free
from minerva.models.ssl.biodiffusion import BioDiffusion


@pytest.fixture
def small_bio_unet():
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


def test_biodiffusion_forward_disables_dropout_when_requested(small_bio_unet):
    model = BioDiffusion(small_bio_unet, noise_steps=4)
    x = torch.rand(2, 2, 16)
    timesteps = torch.tensor([0, 1])
    labels = torch.tensor([0, 1])

    with patch.object(
        small_bio_unet, "forward", wraps=small_bio_unet.forward
    ) as forward:
        model(x, timesteps, labels, with_cond_drop=False)

    assert forward.call_args.kwargs["cond_drop_prob"] == 0


def test_biodiffusion_conditional_sample_uses_requested_activity(small_bio_unet):
    model = BioDiffusion(small_bio_unet, noise_steps=4, channels=2, n_timesteps=16)

    with patch.object(model, "forward", wraps=model.forward) as forward:
        model.sample(batch_size=2, activity=1)

    conditional_calls = [call for call in forward.call_args_list if len(call.args) == 3]
    assert len(conditional_calls) == model.noise_steps
    for call in conditional_calls:
        torch.testing.assert_close(call.args[2], torch.ones(2, dtype=torch.int32))


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


def test_biodiffusion_config_round_trip_preserves_custom_unet():
    unet = Unet1D_cls_free(
        dim=8,
        num_classes=3,
        channels=2,
        cond_drop_prob=0.25,
        dim_mults=(1, 2),
        resnet_block_groups=4,
    )
    model = BioDiffusion(unet, noise_steps=4, channels=2)

    restored = BioDiffusion(**model.get_init_config())
    restored.load_state_dict(model.state_dict())

    assert restored.model.get_init_config() == unet.get_init_config()
