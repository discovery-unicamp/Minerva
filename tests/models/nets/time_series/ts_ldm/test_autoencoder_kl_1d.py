import pytest
import torch

from minerva.models.nets.time_series.ts_ldm.autoencoder_kl_1d import (
    AutoencoderKL1d,
    DiagonalGaussianDistribution1d,
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


def test_ts_ldm_autoencoder_forward(small_autoencoder):
    x = torch.rand(2, 2, 14)

    reconstruction, _ = small_autoencoder(x)

    assert reconstruction.shape == x.shape


def test_ts_ldm_autoencoder_latent_shape(small_autoencoder):
    x = torch.rand(2, 2, 16)

    posterior = small_autoencoder.encode(x)

    assert posterior.mean.shape == (2, 2, 8)


def test_ts_ldm_autoencoder_loss(small_autoencoder):
    x = torch.rand(2, 2, 14)
    reconstruction, posterior = small_autoencoder(x)

    loss, _, _, _ = small_autoencoder.get_loss(x, reconstruction, posterior)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_autoencoder_without_sampling_decodes_posterior_mean(small_autoencoder):
    x = torch.rand(2, 2, 14)

    reconstruction, posterior = small_autoencoder(x, sample_posterior=False)
    expected = small_autoencoder.adapter_unpad(
        small_autoencoder.decode(posterior.mode())
    )

    torch.testing.assert_close(reconstruction, expected)


def test_autoencoder_padding_repeats_last_value(small_autoencoder):
    x = torch.arange(14, dtype=torch.float32).reshape(1, 1, 14)

    padded = small_autoencoder.adapter_pad(x)

    torch.testing.assert_close(padded[..., -3:], torch.tensor([[[13.0, 13.0, 13.0]]]))


def test_autoencoder_loss_trains_latent_representation(small_autoencoder):
    x = torch.rand(2, 2, 14)
    reconstruction, posterior = small_autoencoder(x)
    loss, _, _, _ = small_autoencoder.get_loss(x, reconstruction, posterior)

    loss.backward()

    assert small_autoencoder.quant_conv.weight.grad.abs().sum() > 0


def test_gaussian_kl_for_shifted_mean():
    mean = torch.ones(2, 2, 8)
    log_variance = torch.zeros_like(mean)
    posterior = DiagonalGaussianDistribution1d(torch.cat([mean, log_variance], dim=1))

    loss = posterior.kl()

    # Each of the 16 latent values contributes 0.5 for mean=1 and variance=1.
    torch.testing.assert_close(loss, torch.full((2,), 8.0))
