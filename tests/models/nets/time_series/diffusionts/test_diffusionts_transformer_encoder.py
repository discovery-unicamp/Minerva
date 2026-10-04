import pytest
import torch

from minerva.models.nets.time_series.diffusionts.diffusionts_transformer_encoder import (
    DiffusionTSEncoder,
)


@pytest.mark.parametrize("strategy", ["single", "double"])
def test_diffusionts_encoder_forward(strategy):
    model = DiffusionTSEncoder(
        n_feat=2,
        n_channel=16,
        n_layer_enc=1,
        n_layer_dec=1,
        n_embd=8,
        n_heads=2,
        max_len=16,
        diffusion_timestep=1,
        pass_strategy=strategy,
    )
    x = torch.rand(2, 16, 2)

    output = model(x)

    assert output.shape == (2, 16, 8)


def test_diffusionts_encoder_invalid_block():
    with pytest.raises(ValueError, match="target_block"):
        DiffusionTSEncoder(
            n_feat=2,
            n_channel=16,
            n_layer_enc=1,
            n_layer_dec=1,
            n_embd=8,
            n_heads=2,
            target_block=2,
        )


def test_diffusionts_encoder_stops_at_selected_block():
    model = DiffusionTSEncoder(
        n_feat=2,
        n_channel=16,
        n_layer_enc=2,
        n_layer_dec=1,
        n_embd=8,
        n_heads=2,
        max_len=16,
        target_block=1,
    )
    x = torch.rand(2, 16, 2)

    model(x).square().mean().backward()

    assert next(model.encoder.blocks[0].parameters()).grad is not None
    assert next(model.encoder.blocks[1].parameters()).grad is None


def test_diffusionts_double_pass_encodes_denoised_signal():
    model = DiffusionTSEncoder(
        n_feat=2,
        n_channel=16,
        n_layer_enc=1,
        n_layer_dec=1,
        n_embd=8,
        n_heads=2,
        max_len=16,
        diffusion_timestep=2,
        pass_strategy="double",
    ).eval()
    x = torch.rand(2, 16, 2)

    denoised = model.model_pass_forward(x, torch.ones(2, dtype=torch.long))
    expected, _ = model.encoder_partial_forward(
        denoised, torch.zeros(2, dtype=torch.long), encoder_block=1, additional=True
    )

    torch.testing.assert_close(model(x), expected)
