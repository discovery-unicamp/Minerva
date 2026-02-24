import pytest
import torch
import torch.nn as nn

from minerva.models.ssl.vitmae import MaskedAutoEncoderViT, VisionTransformer

# ----------------------------
# Fixtures
# ----------------------------


@pytest.fixture
def dummy_imgs():
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def small_backbone():
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=2,
        num_heads=3,
        mlp_ratio=2.0,
    )


@pytest.fixture
def mae_model(small_backbone):
    return MaskedAutoEncoderViT(
        backbone=small_backbone,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
    )


# ----------------------------
# Initialization tests
# ----------------------------


def test_init_default():
    model = MaskedAutoEncoderViT()
    assert model.backbone is not None
    assert isinstance(model.decoder_blocks, nn.ModuleList)
    assert model.mask_token.shape[-1] == model.decoder_embed.out_features


def test_initialize_weights_freezes_pos_embed(mae_model):
    assert mae_model.backbone.pos_embed.requires_grad is False
    assert mae_model.decoder_pos_embed.requires_grad is False


def test_mask_token_initialized(mae_model):
    assert torch.any(mae_model.mask_token != 0)


# ----------------------------
# Patchify / Unpatchify
# ----------------------------


def test_patchify_shape(mae_model, dummy_imgs):
    patches = mae_model.patchify(dummy_imgs)
    B, C, H, W = dummy_imgs.shape
    p = mae_model.backbone.patch_embed.patch_size[0]
    num_patches = (H // p) * (W // p)
    assert patches.shape[1] == num_patches


def test_unpatchify_inverse(mae_model, dummy_imgs):
    patches = mae_model.patchify(dummy_imgs)
    recon = mae_model.unpatchify(patches)
    assert recon.shape == dummy_imgs.shape


def test_patchify_invalid_size(mae_model):
    imgs = torch.randn(1, 3, 230, 224)
    with pytest.raises(AssertionError):
        mae_model.patchify(imgs)


# ----------------------------
# Random masking
# ----------------------------


def test_random_masking_shapes(mae_model: MaskedAutoEncoderViT):
    x = torch.randn(2, 196, 64)
    x_masked, mask, ids_restore = mae_model.random_masking(x, mask_ratio=0.5)

    assert x_masked.shape[0] == x.shape[0]
    assert mask.shape == (2, 196)
    assert ids_restore.shape == (2, 196)


def test_random_masking_ratio(mae_model):
    x = torch.randn(1, 100, 32)
    _, mask, _ = mae_model.random_masking(x, mask_ratio=0.3)
    removed = mask.sum().item()
    assert removed == pytest.approx(100 * 0.3, rel=0.1)


# ----------------------------
# Encoder forward
# ----------------------------


def test_forward_encoder_shapes(
    mae_model: MaskedAutoEncoderViT, dummy_imgs: torch.Tensor
):
    latent, mask, ids_restore = mae_model.forward_encoder(dummy_imgs, mask_ratio=0.5)
    assert latent.ndim == 3
    assert mask.ndim == 2
    assert ids_restore.ndim == 2


def test_forward_encoder_cls_token(mae_model, dummy_imgs):
    latent, _, _ = mae_model.forward_encoder(dummy_imgs, mask_ratio=0.5)
    # first token is cls token
    assert latent.shape[1] >= 1


# ----------------------------
# Decoder forward
# ----------------------------


def test_forward_decoder_shapes(mae_model, dummy_imgs):
    latent, mask, ids_restore = mae_model.forward_encoder(dummy_imgs, mask_ratio=0.5)
    pred = mae_model.forward_decoder(latent, ids_restore)

    assert pred.ndim == 3
    assert pred.shape[1] == ids_restore.shape[1]


def test_forward_decoder_removes_cls(mae_model, dummy_imgs):
    latent, _, ids_restore = mae_model.forward_encoder(dummy_imgs, mask_ratio=0.5)
    pred = mae_model.forward_decoder(latent, ids_restore)
    assert pred.shape[1] == ids_restore.shape[1]


# ----------------------------
# Loss computation
# ----------------------------


def test_forward_loss_scalar(mae_model, dummy_imgs):
    latent, mask, ids_restore = mae_model.forward_encoder(dummy_imgs, mask_ratio=0.5)
    pred = mae_model.forward_decoder(latent, ids_restore)
    loss = mae_model.forward_loss(dummy_imgs, pred, mask)
    assert loss.ndim == 0


def test_forward_loss_positive(mae_model, dummy_imgs):
    latent, mask, ids_restore = mae_model.forward_encoder(dummy_imgs, mask_ratio=0.5)
    pred = mae_model.forward_decoder(latent, ids_restore)
    loss = mae_model.forward_loss(dummy_imgs, pred, mask)
    assert loss.item() >= 0


def test_forward_loss_norm_pix(mae_model, dummy_imgs):
    mae_model.norm_pix_loss = True
    latent, mask, ids_restore = mae_model.forward_encoder(dummy_imgs, mask_ratio=0.5)
    pred = mae_model.forward_decoder(latent, ids_restore)
    loss = mae_model.forward_loss(dummy_imgs, pred, mask)
    assert loss.item() >= 0


# ----------------------------
# Full forward pass
# ----------------------------


def test_forward_full(mae_model, dummy_imgs):
    loss, pred, mask = mae_model(dummy_imgs, mask_ratio=0.75)
    assert loss.ndim == 0
    assert pred.ndim == 3
    assert mask.ndim == 2


def test_forward_different_mask_ratios(mae_model, dummy_imgs):
    for ratio in [0.25, 0.5, 0.75]:
        loss, pred, mask = mae_model(dummy_imgs, mask_ratio=ratio)
        assert pred.shape[1] == mask.shape[1]


# ----------------------------
# Determinism sanity check
# ----------------------------


def test_random_masking_different_each_call(mae_model):
    x = torch.randn(1, 100, 32)
    _, mask1, _ = mae_model.random_masking(x, 0.5)
    _, mask2, _ = mae_model.random_masking(x, 0.5)

    # extremely unlikely masks are identical
    assert not torch.equal(mask1, mask2)
