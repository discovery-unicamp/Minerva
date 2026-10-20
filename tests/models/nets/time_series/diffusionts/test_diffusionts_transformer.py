import torch

from minerva.models.nets.time_series.diffusionts.diffusionts_transformer import (
    Transformer,
)


def test_diffusionts_transformer_output_shapes():
    model = Transformer(
        n_feat=2,
        n_channel=16,
        n_layer_enc=1,
        n_layer_dec=1,
        n_embd=8,
        n_heads=2,
        max_len=16,
        attn_pdrop=0,
        resid_pdrop=0,
    )
    x = torch.rand(2, 16, 2)
    timesteps = torch.tensor([0, 1])

    trend, season = model(x, timesteps)

    assert trend.shape == x.shape
    assert season.shape == x.shape


def test_diffusionts_transformer_residual_components_preserve_output():
    model = Transformer(
        n_feat=2,
        n_channel=16,
        n_layer_enc=1,
        n_layer_dec=1,
        n_embd=8,
        n_heads=2,
        max_len=16,
    ).eval()
    x = torch.rand(2, 16, 2)
    timesteps = torch.tensor([0, 1])

    trend, season = model(x, timesteps)
    trend_part, season_part, residual = model(x, timesteps, return_res=True)

    torch.testing.assert_close(trend + season, trend_part + season_part + residual)


def test_diffusionts_transformer_backward_reaches_input():
    model = Transformer(
        n_feat=2,
        n_channel=16,
        n_layer_enc=1,
        n_layer_dec=1,
        n_embd=8,
        n_heads=2,
        max_len=16,
    )
    x = torch.rand(2, 16, 2, requires_grad=True)
    timesteps = torch.tensor([0, 1])
    trend, season = model(x, timesteps)

    (trend + season).square().mean().backward()

    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
