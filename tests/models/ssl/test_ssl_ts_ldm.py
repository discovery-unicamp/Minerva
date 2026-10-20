from types import SimpleNamespace
from unittest.mock import Mock

import lightning as L
import pytest
import torch

from minerva.models.nets.time_series.ts_ldm.autoencoder_kl_1d import AutoencoderKL1d
from minerva.models.nets.time_series.ts_ldm.unet_1d import UNetModel1d
from minerva.models.ssl.ts_ldm import TSLatentDiffusion


@pytest.fixture
def small_ldm_unet():
    return UNetModel1d(
        in_channels=2,
        out_channels=2,
        model_channels=32,
        num_res_blocks=1,
        attention_resolutions=[1],
        channel_mult=(1, 2),
    )


@pytest.fixture
def small_autoencoder():
    return AutoencoderKL1d(
        ddconfig=dict(
            ch=32,
            out_ch=2,
            ch_mult=(1, 2),
            num_res_blocks=1,
            attn_resolutions=[8],
            dropout=0,
            in_channels=2,
            z_channels=2,
            resolution=16,
            double_z=True,
        ),
        embed_dim=2,
        original_length=14,
    )


@pytest.fixture
def latent_diffusion(small_ldm_unet, small_autoencoder):
    return TSLatentDiffusion(
        small_ldm_unet,
        small_autoencoder,
        timesteps=4,
        channels=2,
        sequence_length=8,
        use_scheduler=False,
    )


def test_ts_ldm_freezes_autoencoder(latent_diffusion):
    latent_diffusion.train()
    parameters = latent_diffusion.first_stage_model.parameters()

    assert all(not parameter.requires_grad for parameter in parameters)


@pytest.mark.parametrize("mode", [True, False])
def test_ts_ldm_keeps_autoencoder_in_eval_mode(latent_diffusion, mode):
    latent_diffusion.eval()

    latent_diffusion.train(mode)

    assert latent_diffusion.training == mode
    assert latent_diffusion.model.training == mode
    assert not latent_diffusion.first_stage_model.training


def test_ts_ldm_disables_autoencoder_dropout_during_training(latent_diffusion):
    for layer in latent_diffusion.first_stage_model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0.2
    latent_diffusion.train()
    x = torch.rand(2, 2, 14)

    first = latent_diffusion.encode_first_stage(x)
    second = latent_diffusion.encode_first_stage(x)

    torch.testing.assert_close(first.mean, second.mean)
    torch.testing.assert_close(first.logvar, second.logvar)


def test_ts_ldm_training_loss(latent_diffusion):
    x = torch.rand(2, 2, 14)
    labels = torch.tensor([0, 1])

    loss = latent_diffusion.training_step((x, labels), 0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_ts_ldm_sample(latent_diffusion):
    samples = latent_diffusion.sample(batch_size=2, verbose=False)

    assert samples.shape == (2, 2, 14)
    assert torch.isfinite(samples).all()


def test_ts_ldm_optimizer_learning_rate(latent_diffusion):
    optimizer = latent_diffusion.configure_optimizers()

    assert optimizer.param_groups[0]["lr"] == latent_diffusion.learning_rate


@pytest.fixture
def scaled_latent_diffusion(small_ldm_unet, small_autoencoder):
    return TSLatentDiffusion(
        small_ldm_unet,
        small_autoencoder,
        timesteps=4,
        channels=2,
        sequence_length=8,
        scale_by_std=True,
        use_scheduler=False,
    )


def test_ts_ldm_initializes_scale_once(scaled_latent_diffusion):
    model = scaled_latent_diffusion
    model.trainer = L.Trainer(
        accelerator="cpu", devices=1, logger=False, enable_checkpointing=False
    )
    x = torch.rand(2, 2, 14)
    torch.manual_seed(42)
    z = model.get_first_stage_encoding(model.encode_first_stage(x))
    expected = 1.0 / z.flatten().std()

    torch.manual_seed(42)
    model.on_train_batch_start((x,), 0)
    model.on_train_batch_start((x * 2,), 1)

    torch.testing.assert_close(model.scale_factor, expected)


def test_ts_ldm_nonzero_rank_receives_scale(scaled_latent_diffusion):
    model = scaled_latent_diffusion
    broadcast = Mock(return_value=torch.tensor(2.5))
    model.trainer = SimpleNamespace(
        current_epoch=0,
        global_step=0,
        is_global_zero=False,
        strategy=SimpleNamespace(broadcast=broadcast),
    )
    original_buffer = model.scale_factor

    # Only rank zero needs a batch to compute the scale.
    model.on_train_batch_start(None, 0)

    broadcast.assert_called_once_with(original_buffer, src=0)
    assert model.scale_factor is original_buffer
    torch.testing.assert_close(model.state_dict()["scale_factor"], torch.tensor(2.5))
