import torch

from minerva.models.nets.time_series.diffwave.residual_blocks import (
    ResidualBlock,
    ResidualGroup,
)


def test_diffwave_residual_block_output_shapes():
    model = ResidualBlock(8, 4, dilation=2, diffusion_step_embed_dim_out=8)
    x = torch.rand(2, 8, 16)
    time_embedding = torch.rand(2, 8)

    residual, skip = model((x, time_embedding))

    assert residual.shape == x.shape
    assert skip.shape == (2, 4, 16)


def test_diffwave_residual_block_uses_label_embedding():
    model = ResidualBlock(8, 4, dilation=1, diffusion_step_embed_dim_out=8)
    x = torch.ones(2, 8, 16)
    time_embedding = torch.zeros(2, 8)
    labels = torch.ones(2, 128)

    _, conditioned = model((x, time_embedding), label_emb=labels)
    _, unconditioned = model((x, time_embedding))

    assert not torch.allclose(conditioned, unconditioned)


def test_diffwave_residual_group_normalizes_skip_output():
    model = ResidualGroup(
        res_channels=8,
        skip_channels=4,
        num_res_layers=2,
        diffusion_step_embed_dim_in=8,
        diffusion_step_embed_dim_mid=8,
        diffusion_step_embed_dim_out=8,
    )
    x = torch.rand(2, 8, 16)
    timesteps = torch.tensor([[0], [1]])

    output = model((x, timesteps))
    _, skip = model.forward_emb((x, timesteps))

    torch.testing.assert_close(output, skip / (2**0.5))


def test_diffwave_residual_group_stops_at_selected_layer():
    model = ResidualGroup(
        res_channels=8,
        skip_channels=4,
        num_res_layers=2,
        diffusion_step_embed_dim_in=8,
        diffusion_step_embed_dim_mid=8,
        diffusion_step_embed_dim_out=8,
    )
    x = torch.rand(2, 8, 16)
    timesteps = torch.tensor([[0], [1]])

    features, _ = model.forward_emb((x, timesteps), target_res_layer=0)
    features.sum().backward()

    assert model.residual_blocks[0].fc_t.weight.grad is not None
    assert model.residual_blocks[1].fc_t.weight.grad is None
