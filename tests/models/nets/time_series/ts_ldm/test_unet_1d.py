import pytest
import torch

from minerva.models.nets.time_series.ts_ldm.unet_1d import UNetModel1d


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


def test_ts_ldm_unet_forward(small_ldm_unet):
    x = torch.rand(2, 2, 8)
    timesteps = torch.tensor([0, 1])

    output = small_ldm_unet(x, timesteps)

    assert output.shape == x.shape


@pytest.mark.parametrize(
    "block, expected_shape",
    [(0, (2, 32, 8)), (2, (2, 32, 4)), (None, (2, 64, 4))],
)
def test_ts_ldm_unet_features(small_ldm_unet, block, expected_shape):
    x = torch.rand(2, 2, 8)
    timesteps = torch.tensor([0, 1])

    output = small_ldm_unet.forward_emb(x, timesteps, block=block)

    assert output.shape == expected_shape


def test_ts_ldm_unet_zero_output_can_learn(small_ldm_unet):
    x = torch.rand(2, 2, 8)
    timesteps = torch.tensor([0, 1])
    loss = (small_ldm_unet(x, timesteps) - 1).square().mean()

    loss.backward()

    assert small_ldm_unet.out[-1].weight.grad.abs().sum() > 0
