import pytest
import torch
import torch.nn as nn
import numpy as np

from minerva.models.nets.image.dino import DINOHead
from minerva.models.nets.image.dino.v2_vit import DinoVisionTransformer as V2ViT
from minerva.models.nets.image.dino.v3_vit import DinoVisionTransformer as V3ViT
from minerva.models.ssl.dino import (
    AugmentationConfig,
    CosineScheduler,
    DINOv2,
    DINOv3,
    GramConfig,
    LossConfig,
    MiscConfig,
    OptimConfig,
    apply_optim_scheduler,
    collate_data_and_cast,
    fuse_params_groups,
    get_params_groups_with_decay,
    get_vit_lr_decay_rate,
    init_config,
    linear_warmup_cosine_decay,
    remove_fsdp_compile_names,
)
from minerva.transforms.dino import MaskingGenerator


# -------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------
@pytest.fixture
def small_v2_backbone():
    return V2ViT(
        img_size=32,
        patch_size=16,
        embed_dim=64,
        depth=2,
        num_heads=2,
        ffn_ratio=2.0,
    )


@pytest.fixture
def small_v3_backbone():
    return V3ViT(
        img_size=32,
        patch_size=16,
        embed_dim=64,
        depth=2,
        num_heads=2,
        ffn_ratio=2.0,
    )


@pytest.fixture
def small_dino_head():
    return DINOHead(
        in_dim=64,
        out_dim=32,
        nlayers=2,
        hidden_dim=64,
        bottleneck_dim=32,
        dino_version=2,
    )


@pytest.fixture
def small_dino_head_v3():
    return DINOHead(
        in_dim=64,
        out_dim=32,
        nlayers=2,
        hidden_dim=64,
        bottleneck_dim=32,
        dino_version=3,
    )


# -------------------------------------------------------------
# Configuration and Helper Tests
# -------------------------------------------------------------
def test_configs_and_init():
    cfg = init_config(LossConfig, {"dino_loss_weight": 2.0})
    assert isinstance(cfg, LossConfig)
    assert cfg.dino_loss_weight == 2.0

    gram_cfg = init_config(GramConfig, None)
    assert isinstance(gram_cfg, GramConfig)

    aug_cfg = init_config(AugmentationConfig, None)
    assert isinstance(aug_cfg, AugmentationConfig)

    opt_cfg = init_config(OptimConfig, None)
    assert isinstance(opt_cfg, OptimConfig)

    misc_cfg = init_config(MiscConfig, None)
    assert isinstance(misc_cfg, MiscConfig)


def test_linear_warmup_cosine_decay():
    schedule = linear_warmup_cosine_decay(
        start=0.0,
        peak=1.0,
        end=0.1,
        warmup_iterations=10,
        total_iterations=100,
    )
    assert isinstance(schedule, np.ndarray)
    assert schedule[0] == pytest.approx(0.0)
    assert schedule[10] == pytest.approx(1.0)
    assert schedule[-1] == pytest.approx(0.1)


def test_cosine_scheduler():
    scheduler = CosineScheduler(
        base_value=1.0,
        final_value=0.1,
        total_iters=100,
        warmup_iters=10,
        start_warmup_value=0.0,
    )
    assert scheduler[0] == pytest.approx(0.0)
    assert scheduler[10] == pytest.approx(1.0)
    assert scheduler[99] == pytest.approx(0.1, rel=1e-1)


def test_get_vit_lr_decay_rate():
    decay_cls = get_vit_lr_decay_rate(
        "backbone.cls_token", lr_decay_rate=0.8, num_layers=4
    )
    assert decay_cls == pytest.approx(0.8**5)

    decay_blk0 = get_vit_lr_decay_rate(
        "backbone.blocks.0.mlp", lr_decay_rate=0.8, num_layers=4
    )
    assert decay_blk0 == pytest.approx(0.8**4)

    decay_blk3 = get_vit_lr_decay_rate(
        "backbone.blocks.3.mlp", lr_decay_rate=0.8, num_layers=4
    )
    assert decay_blk3 == pytest.approx(0.8**1)


def test_get_params_groups_with_decay(small_v2_backbone):
    groups = get_params_groups_with_decay(small_v2_backbone, lr_decay_rate=0.9)
    assert len(groups) > 0
    fused = fuse_params_groups(groups)
    assert len(fused) > 0


def test_remove_fsdp_compile_names():
    cleaned = remove_fsdp_compile_names(
        "_orig_mod._checkpoint_wrapped_module.layer.weight"
    )
    assert cleaned == "layer.weight"


def test_apply_optim_scheduler():
    param = nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [param],
                "lr": 1e-3,
                "weight_decay": 0.01,
                "is_last_layer": False,
                "lr_multiplier": 1.0,
                "wd_multiplier": 1.0,
            }
        ]
    )
    apply_optim_scheduler(optimizer, lr=5e-4, wd=0.02, last_layer_lr=1e-3)
    assert optimizer.param_groups[0]["lr"] == 5e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.02


def test_collate_data_and_cast():
    # Construct a dummy sample matching DataAugmentationDINO output
    def make_sample():
        return {
            "global_crops": [torch.randn(3, 32, 32), torch.randn(3, 32, 32)],
            "global_crops_teacher": [torch.randn(3, 32, 32), torch.randn(3, 32, 32)],
            "local_crops": [torch.randn(3, 16, 16), torch.randn(3, 16, 16)],
            "offsets": [],
        }

    samples = [(make_sample(), 0), (make_sample(), 0)]
    mask_gen = MaskingGenerator(input_size=(2, 2))

    collated = collate_data_and_cast(
        samples_list=samples,
        mask_ratio_tuple=(0.1, 0.5),
        mask_probability=0.5,
        dtype=torch.float32,
        n_tokens=4,
        mask_generator=mask_gen,
        random_circular_shift=False,
        local_batch_size=None,
        dino_version=2,
    )

    assert "collated_global_crops" in collated
    assert "collated_local_crops" in collated
    assert "collated_masks" in collated
    assert "mask_indices_list" in collated
    assert "n_masked_patches" in collated


# -------------------------------------------------------------
# DINOv2 SSL Technique Tests
# -------------------------------------------------------------
def test_dinov2_initialization(small_v2_backbone, small_dino_head):
    model = DINOv2(
        backbone=small_v2_backbone,
        learning_rate=1e-3,
        batch_size=2,
        epochs=5,
        iter_per_epoch=10,
        prediction_head=small_dino_head,
        crops={
            "global_crops_size": 32,
            "local_crops_size": 16,
            "local_crops_number": 2,
        },
    )

    assert model.dino_version == 2
    assert model.student is not None
    assert model.teacher is not None
    # Teacher should have requires_grad=False
    for p in model.teacher.parameters():
        assert not p.requires_grad


def test_dinov2_optimizer_and_transforms(small_v2_backbone, small_dino_head):
    model = DINOv2(
        backbone=small_v2_backbone,
        learning_rate=1e-3,
        batch_size=2,
        epochs=5,
        iter_per_epoch=10,
        prediction_head=small_dino_head,
        crops={
            "global_crops_size": 32,
            "local_crops_size": 16,
            "local_crops_number": 2,
        },
    )

    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)

    transforms = model.default_technique_transforms()
    assert callable(transforms)

    collate_fn = model.default_technique_collate_fn()
    assert callable(collate_fn)


def test_dinov2_forward(small_v2_backbone, small_dino_head):
    model = DINOv2(
        backbone=small_v2_backbone,
        learning_rate=1e-3,
        batch_size=2,
        epochs=5,
        iter_per_epoch=10,
        prediction_head=small_dino_head,
        crops={
            "global_crops_size": 32,
            "local_crops_size": 16,
            "local_crops_number": 2,
        },
        misc={"param_dtype": "fp32"},
    )
    model.init_weights()

    collate_fn = model.default_technique_collate_fn()
    sample = {
        "global_crops": [torch.randn(3, 32, 32), torch.randn(3, 32, 32)],
        "global_crops_teacher": [torch.randn(3, 32, 32), torch.randn(3, 32, 32)],
        "local_crops": [torch.randn(3, 16, 16), torch.randn(3, 16, 16)],
        "offsets": [],
    }
    batch = collate_fn([(sample, 0), (sample, 0)])

    loss, loss_dict = model.forward(batch, teacher_temp=0.07)
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)
    assert isinstance(loss_dict, dict)


# -------------------------------------------------------------
# DINOv3 SSL Technique Tests
# -------------------------------------------------------------
def test_dinov3_initialization(small_v3_backbone, small_dino_head_v3):
    ibot_head = DINOHead(
        in_dim=64,
        out_dim=32,
        nlayers=2,
        hidden_dim=64,
        bottleneck_dim=32,
        dino_version=3,
    )
    model = DINOv3(
        backbone=small_v3_backbone,
        learning_rate=1e-3,
        batch_size=2,
        epochs=5,
        iter_per_epoch=10,
        prediction_head=small_dino_head_v3,
        ibot_separate_head=True,
        ibot_head=ibot_head,
        crops={
            "global_crops_size": 32,
            "local_crops_size": 16,
            "local_crops_number": 2,
        },
    )

    assert model.dino_version == 3
    assert "ibot_head" in model.student
    assert "ibot_head" in model.teacher


def test_dinov3_forward(small_v3_backbone, small_dino_head_v3):
    ibot_head = DINOHead(
        in_dim=64,
        out_dim=32,
        nlayers=2,
        hidden_dim=64,
        bottleneck_dim=32,
        dino_version=3,
    )
    model = DINOv3(
        backbone=small_v3_backbone,
        learning_rate=1e-3,
        batch_size=2,
        epochs=5,
        iter_per_epoch=10,
        prediction_head=small_dino_head_v3,
        ibot_separate_head=True,
        ibot_head=ibot_head,
        crops={
            "global_crops_size": 32,
            "local_crops_size": 16,
            "local_crops_number": 2,
        },
        misc={"param_dtype": "fp32"},
    )
    model.init_weights()

    collate_fn = model.default_technique_collate_fn()
    sample = {
        "global_crops": [torch.randn(3, 32, 32), torch.randn(3, 32, 32)],
        "global_crops_teacher": [torch.randn(3, 32, 32), torch.randn(3, 32, 32)],
        "local_crops": [torch.randn(3, 16, 16), torch.randn(3, 16, 16)],
        "offsets": [],
    }
    batch = collate_fn([(sample, 0), (sample, 0)])

    loss, metrics = model.forward(batch, teacher_temp=0.07, iteration=0)
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)
    assert isinstance(metrics, dict)


def test_dino_wrapper_collate_forward_end_to_end(small_v2_backbone, small_dino_head):
    """Tuple dataset -> SSLDatasetWrapper(DINOTransform) -> collate -> forward."""
    from PIL import Image
    from torch.utils.data import Dataset

    from minerva.data.datasets.wrappers import SSLDatasetWrapper

    class Tuples(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            arr = np.random.randint(0, 256, size=(48, 48, 3), dtype=np.uint8)
            return Image.fromarray(arr), idx

    model = DINOv2(
        backbone=small_v2_backbone,
        learning_rate=1e-3,
        batch_size=2,
        epochs=5,
        iter_per_epoch=10,
        prediction_head=small_dino_head,
        crops={
            "global_crops_size": 32,
            "local_crops_size": 16,
            "local_crops_number": 2,
        },
        misc={"param_dtype": "fp32"},
    )
    model.init_weights()

    dataset = SSLDatasetWrapper(Tuples(), model.default_technique_transforms())
    sample = dataset[0]
    assert sample["image_global_crops"].shape[1:] == (3, 32, 32)
    assert sample["label"] == 0

    batch = model.default_technique_collate_fn()([dataset[0], dataset[1]])
    loss, _ = model.forward(batch, teacher_temp=0.07)
    assert torch.isfinite(loss)
