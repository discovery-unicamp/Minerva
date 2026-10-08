import pytest
import torch

from minerva.models.loaders import FromPretrained
from minerva.models.nets.base import SimpleSupervisedModel
from minerva.models.nets.time_series.diffusionts.diffusionts_transformer import (
    Transformer,
)
from minerva.models.ssl.diffusionts import DiffusionTS

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


@pytest.fixture
def pretrained_transformer():
    config = _small_encoder_kwargs()
    config.pop("pass_strategy")
    return Transformer(**config)


@pytest.mark.parametrize("strategy", ["single", "double"])
def test_encoder_loads_diffusionts_pretraining_checkpoint(
    pretrained_transformer, tmp_path, strategy
):
    pretrained = DiffusionTS(
        pretrained_transformer, seq_length=16, feature_size=2, timesteps=4
    )
    checkpoint = tmp_path / "pretrained.ckpt"
    torch.save({"state_dict": pretrained.state_dict()}, checkpoint)
    config = {**_small_encoder_kwargs(), "pass_strategy": strategy}

    encoder = FromPretrained(
        model=DiffusionTSEncoder(**config),
        ckpt_path=checkpoint,
        filter_keys=["^model"],
        keys_to_rename={"model.": ""},
        strict=True,
        error_on_missing_keys=True,
    )

    torch.testing.assert_close(
        encoder.emb.state_dict(), pretrained_transformer.emb.state_dict()
    )
    if strategy == "double":
        torch.testing.assert_close(
            encoder.additional_emb.state_dict(), pretrained_transformer.emb.state_dict()
        )
        torch.testing.assert_close(
            encoder.additional_pos_enc.state_dict(),
            pretrained_transformer.pos_enc.state_dict(),
        )
        torch.testing.assert_close(
            encoder.additional_encoder_blocks.state_dict(),
            pretrained_transformer.encoder.blocks.state_dict(),
        )


@pytest.mark.parametrize("assign", [False, True])
def test_loaded_second_pass_parameters_are_independent(pretrained_transformer, assign):
    encoder = DiffusionTSEncoder(**_small_encoder_kwargs())

    encoder.load_state_dict(pretrained_transformer.state_dict(), assign=assign)

    first = encoder.emb.sequential[1].weight
    second = encoder.additional_emb.sequential[1].weight
    assert first.data_ptr() != second.data_ptr()


def test_supervised_checkpoint_preserves_finetuned_second_pass():
    source = SimpleSupervisedModel(
        backbone=DiffusionTSEncoder(**_small_encoder_kwargs()),
        fc=torch.nn.Linear(128, 3),
        loss_fn=torch.nn.CrossEntropyLoss(),
    )
    with torch.no_grad():
        source.backbone.additional_emb.sequential[1].weight.add_(1)
    restored = SimpleSupervisedModel(
        backbone=DiffusionTSEncoder(**_small_encoder_kwargs()),
        fc=torch.nn.Linear(128, 3),
        loss_fn=torch.nn.CrossEntropyLoss(),
    )

    restored.load_state_dict(source.state_dict(), strict=True)

    torch.testing.assert_close(restored.state_dict(), source.state_dict())


def test_nested_encoder_initializes_missing_second_pass(pretrained_transformer):
    source = torch.nn.ModuleDict({"backbone": pretrained_transformer})
    restored = torch.nn.ModuleDict(
        {"backbone": DiffusionTSEncoder(**_small_encoder_kwargs())}
    )

    restored.load_state_dict(source.state_dict(), strict=True)

    torch.testing.assert_close(
        restored["backbone"].additional_encoder_blocks.state_dict(),
        pretrained_transformer.encoder.blocks.state_dict(),
    )


def test_encoder_rejects_partial_second_pass_checkpoint():
    encoder = DiffusionTSEncoder(**_small_encoder_kwargs())
    state = encoder.state_dict()
    del state["additional_emb.sequential.1.weight"]

    with pytest.raises(RuntimeError, match="additional_emb.sequential.1.weight"):
        encoder.load_state_dict(state, strict=True)


@pytest.mark.parametrize("invalid", ["missing", "unexpected", "shape"])
def test_pretraining_load_preserves_strict_validation(pretrained_transformer, invalid):
    encoder = DiffusionTSEncoder(**_small_encoder_kwargs())
    state = pretrained_transformer.state_dict()
    if invalid == "missing":
        del state["emb.sequential.1.weight"]
    elif invalid == "unexpected":
        state["unknown.weight"] = torch.ones(1)
    else:
        state["emb.sequential.1.weight"] = torch.ones(1)

    with pytest.raises(RuntimeError):
        encoder.load_state_dict(state, strict=True)
