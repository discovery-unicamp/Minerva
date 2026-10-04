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
    parameters = latent_diffusion.first_stage_model.parameters()

    assert all(not parameter.requires_grad for parameter in parameters)


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
