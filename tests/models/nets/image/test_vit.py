import pytest
import torch
import torch.nn as nn

from minerva.models.nets.image.vit_local.vit import VisionTransformer

# ----------------------------
# Helpers
# ----------------------------


@pytest.fixture
def dummy_input():
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def small_vit():
    """Smaller model for fast tests"""
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=2,
        num_heads=3,
        mlp_ratio=2.0,
    )


# ----------------------------
# Initialization tests
# ----------------------------


def test_init_default():
    model = VisionTransformer()
    assert model.embed_dim == 768
    assert isinstance(model.patch_embed, nn.Module)
    assert model.blocks is not None
    assert isinstance(model.blocks, nn.Sequential)


def test_init_no_class_token():
    model = VisionTransformer(class_token=False)
    assert model.cls_token is None
    assert model.num_prefix_tokens == 0


def test_init_with_reg_tokens():
    model = VisionTransformer(reg_tokens=2)
    assert model.reg_token is not None
    assert model.num_reg_tokens == 2
    assert model.num_prefix_tokens == 3  # 1 cls + 2 reg


def test_invalid_pos_embed():
    with pytest.raises(AssertionError):
        VisionTransformer(pos_embed="invalid")


# ----------------------------
# Positional embedding logic
# ----------------------------


def test_pos_embed_none(small_vit, dummy_input):
    small_vit.pos_embed = None
    x = small_vit.patch_embed(dummy_input)
    out = small_vit._pos_embed(x)
    assert out.ndim == 3


def test_pos_embed_with_class_token(small_vit, dummy_input):
    x = small_vit.patch_embed(dummy_input)
    out = small_vit._pos_embed(x)
    # should include class token
    assert out.shape[1] == small_vit.patch_embed.num_patches + 1


def test_pos_embed_no_embed_class():
    model = VisionTransformer(no_embed_class=True)
    x = torch.randn(1, 3, 224, 224)
    x = model.patch_embed(x)
    out = model._pos_embed(x)
    assert out.shape[1] == model.patch_embed.num_patches + model.num_prefix_tokens


# ----------------------------
# Forward pass tests
# ----------------------------


def test_forward_output_shape(small_vit, dummy_input):
    out = small_vit(dummy_input)
    assert out.shape[0] == dummy_input.shape[0]
    assert out.shape[-1] == small_vit.embed_dim


def test_forward_no_class_token():
    model = VisionTransformer(class_token=False)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape[1] == model.patch_embed.num_patches


def test_forward_with_reg_tokens():
    model = VisionTransformer(reg_tokens=2)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    expected_tokens = model.patch_embed.num_patches + model.num_prefix_tokens
    assert out.shape[1] == expected_tokens


# ----------------------------
# Dynamic image size behavior
# ----------------------------


def test_dynamic_img_size_forward():
    model = VisionTransformer(dynamic_img_size=True)
    x = torch.randn(1, 3, 128, 160)
    out = model(x)
    assert out.ndim == 3
    assert out.shape[-1] == model.embed_dim


# ----------------------------
# set_input_size tests
# ----------------------------


def test_set_input_size_changes_grid(small_vit):
    old_grid = small_vit.patch_embed.grid_size
    small_vit.set_input_size(img_size=(384, 384))
    new_grid = small_vit.patch_embed.grid_size
    assert new_grid != old_grid


def test_set_input_size_resamples_pos_embed(small_vit):
    old_len = small_vit.pos_embed.shape[1]
    small_vit.set_input_size(img_size=(384, 384))
    new_len = small_vit.pos_embed.shape[1]
    assert new_len != old_len


# ----------------------------
# Weight initialization
# ----------------------------


def test_init_weights_modes():
    for mode in ["", "jax", "jax_nlhb", "moco"]:
        model = VisionTransformer(weight_init=mode)
        assert model is not None


def test_init_weights_invalid_mode():
    model = VisionTransformer(weight_init="skip")
    with pytest.raises(AssertionError):
        model.init_weights("invalid")


# ----------------------------
# fix_init_weight behavior
# ----------------------------


def test_fix_init_weight_scales_weights(small_vit):
    # clone weights before
    w_before = small_vit.blocks[0].mlp.fc2.weight.clone()
    small_vit.fix_init_weight()
    w_after = small_vit.blocks[0].mlp.fc2.weight

    # weights should change after scaling
    assert not torch.equal(w_before, w_after)


# ----------------------------
# Patch dropout behavior
# ----------------------------


def test_patch_dropout_identity_when_zero():
    model = VisionTransformer(patch_drop_rate=0.0)
    assert isinstance(model.patch_drop, nn.Identity)


def test_patch_dropout_active_when_nonzero():
    model = VisionTransformer(patch_drop_rate=0.1)
    assert not isinstance(model.patch_drop, nn.Identity)


# ----------------------------
# Gradient checkpointing branch
# ----------------------------


def test_grad_checkpointing_path(small_vit, dummy_input):
    small_vit.grad_checkpointing = True
    out = small_vit(dummy_input)
    assert out is not None
