import pytest
import torch

from minerva.models.nets.time_series.diffwave.diffwave_encoder import DiffWaveEncoder
from minerva.models.ssl.diffwave import DiffWave


@pytest.fixture
def small_diffwave():
    return DiffWave(
        in_channels=2,
        out_channels=2,
        res_channels=8,
        skip_channels=4,
        num_res_layers=2,
        dilation_cycle=2,
        diffusion_step_embed_dim_in=8,
        diffusion_step_embed_dim_mid=16,
        diffusion_step_embed_dim_out=8,
        T=4,
    )


@pytest.mark.parametrize("strategy", ["single", "double"])
def test_diffwave_encoder_forward(small_diffwave, strategy):
    model = DiffWaveEncoder(small_diffwave, target_block=0, pass_strategy=strategy)
    x = torch.rand(2, 2, 16)

    output = model(x)

    assert output.shape == (2, 8)


def test_diffwave_encoder_flatten_preserves_temporal_values(small_diffwave):
    model = DiffWaveEncoder(small_diffwave, flatten=True)
    x = torch.rand(2, 2, 16)

    output = model(x)
    pooled = DiffWaveEncoder(small_diffwave, flatten=False)(x)

    torch.testing.assert_close(output.reshape(2, 8, 16).mean(dim=-1), pooled)


def test_diffwave_encoder_uses_selected_timestep_and_block(small_diffwave):
    model = DiffWaveEncoder(small_diffwave, diffusion_timestep=2, target_block=1)
    x = torch.rand(2, 2, 16)

    output = model(x)
    expected = small_diffwave.simple_forward(x, target_time_step=2, target_res_layer=1)

    torch.testing.assert_close(output, expected)
