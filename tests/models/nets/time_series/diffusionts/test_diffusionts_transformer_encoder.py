import pytest
import torch

from minerva.models.nets.time_series.diffusionts.diffusionts_transformer_encoder import (
    DiffusionTSEncoder,
)


def _small_encoder_kwargs():
    return {
        "n_feat": 2,
        "n_channel": 16,
        "n_layer_enc": 1,
        "n_layer_dec": 1,
        "n_embd": 8,
        "n_heads": 2,
        "max_len": 16,
        "pass_strategy": "double",
    }


@pytest.mark.parametrize(
    ("strategy", "timestep"),
    [("single", 0), ("single", 1), ("double", 0), ("double", 1)],
)
def test_diffusionts_encoder_forward(strategy, timestep):
    model = DiffusionTSEncoder(
        n_feat=2,
        n_channel=16,
        n_layer_enc=1,
        n_layer_dec=1,
        n_embd=8,
        n_heads=2,
        max_len=16,
        diffusion_timestep=timestep,
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


@pytest.mark.parametrize("strategy", ["triple", "Single"])
def test_diffusionts_encoder_rejects_invalid_strategy(strategy):
    with pytest.raises(ValueError, match="pass_strategy"):
        DiffusionTSEncoder(n_feat=2, n_channel=16, pass_strategy=strategy)


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


def test_double_pass_starts_as_independent_copy_of_first_pass():
    encoder = DiffusionTSEncoder(**_small_encoder_kwargs())

    for first, second in (
        (encoder.emb, encoder.additional_emb),
        (encoder.pos_enc, encoder.additional_pos_enc),
        (encoder.encoder.blocks, encoder.additional_encoder_blocks),
    ):
        for key, value in first.state_dict().items():
            copied = second.state_dict()[key]
            torch.testing.assert_close(copied, value)
            assert copied.data_ptr() != value.data_ptr()

    second_weight = next(encoder.additional_emb.parameters())
    second_weight.sum().backward()
    assert next(encoder.emb.parameters()).grad is None
    assert second_weight.grad is not None


def test_double_pass_restores_both_sets_of_weights_from_encoder_checkpoint():
    source = DiffusionTSEncoder(**_small_encoder_kwargs())
    with torch.no_grad():
        source.additional_emb.sequential[1].weight.add_(1)
    restored = DiffusionTSEncoder(**_small_encoder_kwargs())

    restored.load_state_dict(source.state_dict())

    torch.testing.assert_close(
        restored.additional_emb.sequential[1].weight,
        source.additional_emb.sequential[1].weight,
    )
    assert not torch.allclose(
        restored.additional_emb.sequential[1].weight,
        restored.emb.sequential[1].weight,
    )
