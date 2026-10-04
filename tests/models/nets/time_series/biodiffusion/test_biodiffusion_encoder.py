import pytest
import torch

from minerva.models.nets.time_series.biodiffusion.biodiffusion_encoder import (
    BioDiffusionEncoder,
)
from minerva.models.nets.time_series.biodiffusion.unet1d import Unet1D_cls_free
from minerva.models.ssl.biodiffusion import BioDiffusion


@pytest.fixture
def small_bio_unet():
    # Only the three core parameters are stored by get_init_config().
    return Unet1D_cls_free(dim=8, num_classes=3, channels=2)


@pytest.mark.parametrize("strategy", ["single", "double"])
def test_biodiffusion_encoder_forward(small_bio_unet, strategy):
    backbone = BioDiffusion(small_bio_unet, noise_steps=4, channels=2)
    model = BioDiffusionEncoder(
        backbone, diffusion_timestep=1, target_block=1, pass_strategy=strategy
    )
    x = torch.rand(2, 2, 16)

    output = model(x)

    assert output.shape == (2, 8, 8)


def test_biodiffusion_encoder_uses_selected_timestep_and_block(small_bio_unet):
    backbone = BioDiffusion(small_bio_unet, noise_steps=4, channels=2)
    model = BioDiffusionEncoder(backbone, diffusion_timestep=2, target_block=3)
    x = torch.rand(2, 2, 16)

    output = model(x)
    expected = backbone.simple_forward(x, t=2, target_block=3)

    torch.testing.assert_close(output, expected)
