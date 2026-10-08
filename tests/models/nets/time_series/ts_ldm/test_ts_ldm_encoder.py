import pytest
import torch

from minerva.models.nets.time_series.ts_ldm.autoencoder_kl_1d import AutoencoderKL1d
from minerva.models.nets.time_series.ts_ldm.ts_ldm_encoder import (
    TSLatentDiffusionEncoder,
)
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


@pytest.mark.parametrize("strategy", ["single", "double"])
def test_ts_ldm_encoder_forward(small_ldm_unet, small_autoencoder, strategy):
    backbone = TSLatentDiffusion(
        small_ldm_unet, small_autoencoder, timesteps=4, channels=2, sequence_length=8
    )
    model = TSLatentDiffusionEncoder(backbone, target_block=0, pass_strategy=strategy)
    x = torch.rand(2, 2, 14)

    output = model(x)

    assert output.shape == (2, 32, 8)


def test_ts_ldm_encoder_selects_deeper_block(small_ldm_unet, small_autoencoder):
    backbone = TSLatentDiffusion(
        small_ldm_unet, small_autoencoder, timesteps=4, channels=2, sequence_length=8
    )
    model = TSLatentDiffusionEncoder(backbone, target_block=3)
    x = torch.rand(2, 2, 14)

    output = model(x)

    assert output.shape == (2, 64, 4)
