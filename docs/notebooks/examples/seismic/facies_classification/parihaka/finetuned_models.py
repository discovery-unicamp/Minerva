from pathlib import Path

import torch
from minerva.models.nets.image.deeplabv3 import DeepLabV3
from minerva.models.ssl.dinov2 import (
    DinoVisionTransformer,
    DPT,
    NestedTensorBlock,
    MemEffAttention,
    DinoV2,
    SETR_MLA,
    SETR_PUP,
)
from minerva.models.nets.image.sam import Sam
from minerva.models.nets.image.vit import SFM_BasePatch16_Downstream
from functools import partial
import lightning as L


default_ckpt_dir = Path.cwd() / "finetuned_parihaka_models"

# DeepLabV3 with 1x1006x590 input size + Logits


# C = channels, H = height, W = width, N = Classes, B = Batch size
###############################################################################
# SSL               Backbone            FORWARD INPUT      OUTPUT SIZE   
# -----------------------------------------------------------------------------
# BYOL              deeplabv3           1x3x1006x590       1x6x1006x590
# FASTSIAM          deeplabv3           1x3x1006x590       1x6x1006x590
# KENSHODENSE       deeplabv3           1x3x1006x590       1x6x1006x590
# SIMCLR            deeplabv3           1x3x1006x590       1x6x1006x590
# TRIBYOL           deeplabv3           1x3x1006x590       1x6x1006x590
# SAM               vit-b               1x1x1006x590       1x6x1006x590

# LFR               deeplabv3           1x3x1024x1024      1x6x1024x1024
# DINOV2_DPT        dino                1x3x1008x784       1x6x1008x784
# DINOV2_MLA        dino                1x3x1008x784       1x6x1008x784
# DINOV2_PUP        dino                1x3x1008x784       1x6x1008x784
# SFM_BASE_PATCH16  vit                 1x1x512x512        1x6x512x512

def byol():
    model = DeepLabV3()

    return {
        "name": "byol",
        "model": model,
        "ckpt_file": default_ckpt_dir
        / "byol"
        / "seam_ai"
        / "checkpoints"
        / "last.ckpt",
        "img_size": (1006, 590),
        "single_channel": False,
    }


def fastsiam():
    model = DeepLabV3()

    return {
        "name": "fastsiam",
        "model": model,
        "ckpt_file": default_ckpt_dir
        / "fastsiam"
        / "seam_ai"
        / "checkpoints"
        / "last.ckpt",
        "img_size": (1006, 590),
        "single_channel": False,
    }


def kenshodense():
    model = DeepLabV3()

    return {
        "name": "kenshodense",
        "model": model,
        "ckpt_file": default_ckpt_dir
        / "kenshodense"
        / "seam_ai"
        / "checkpoints"
        / "last.ckpt",
        "img_size": (1006, 590),
        "single_channel": False,
    }


def simclr():
    model = DeepLabV3()

    return {
        "name": "simclr",
        "model": model,
        "ckpt_file": default_ckpt_dir
        / "simclr"
        / "seam_ai"
        / "checkpoints"
        / "last.ckpt",
        "img_size": (1006, 590),
        "single_channel": False,
    }


def tribyol():
    model = DeepLabV3()

    return {
        "name": "tribyol",
        "model": model,
        "ckpt_file": default_ckpt_dir
        / "tribyol"
        / "seam_ai"
        / "checkpoints"
        / "last.ckpt",
        "img_size": (1006, 590),
        "single_channel": False,
    }


# SAM model with 1006x590 input size


def sam():
    img_size = (1006, 590)
    vit_model = "vit-b"
    num_classes = 6
    multimask_output = True
    ckpt_file = (
        default_ckpt_dir / "sam_vit_b" / "seam_ai" / "checkpoints" / "last.ckpt"
    )

    model = Sam(
        vit_type=vit_model,
        num_multimask_outputs=num_classes,
        iou_head_depth=num_classes,
        multimask_output=multimask_output,
        return_prediction_only=True,
        # checkpoint=str(ckpt_file),
        # apply_freeze=apply_freeze, # if you need to freeze some layer
        # apply_adapter=apply_adapter # if you need to apply an adapter to a layer
    )

    return {
        "name": "sam",
        "model": model,
        "ckpt_file": ckpt_file,
        "img_size": img_size,
        "single_channel": True,
    }


# DeepLabV3 with 1024x1024 input size


def lfr():
    model = DeepLabV3()

    return {
        "name": "lfr",
        "model": model,
        "ckpt_file": default_ckpt_dir
        / "lfr"
        / "seam_ai"
        / "checkpoints"
        / "last.ckpt",
        "img_size": (1024, 1024),
        "single_channel": False,
    }


# DinoV2 models (DPT, MLA, PUP) with 1008x784 input size


def dinov2_dpt():
    img_size = (1008, 784)

    backbone = DinoVisionTransformer(
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),  # type: ignore
        init_values=1e-5,
        block_chunks=0,
    )

    head = DPT(embedding_dim=384, num_classes=6)

    model = DinoV2(
        backbone=backbone,
        head=head,
        loss_fn=torch.nn.CrossEntropyLoss(),
        output_shape=img_size,
        middle=True,
    )

    return {
        "name": "dinov2_dpt",
        "model": model,
        "ckpt_file": default_ckpt_dir
        / "dinov2_dpt"
        / "seam_ai"
        / "checkpoints"
        / "last.ckpt",
        "img_size": img_size,
        "single_channel": False,
    }


def dinov2_mla():
    img_size = (1008, 784)

    backbone = DinoVisionTransformer(
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),  # type: ignore
        init_values=1e-5,
        block_chunks=0,
    )

    head = SETR_MLA(embedding_dim=384, num_classes=6)

    model = DinoV2(
        backbone=backbone,
        head=head,
        loss_fn=torch.nn.CrossEntropyLoss(),
        output_shape=img_size,
        middle=True,
    )

    return {
        "name": "dinov2_mla",
        "model": model,
        "ckpt_file": default_ckpt_dir
        / "dinov2_mla"
        / "seam_ai"
        / "checkpoints"
        / "last.ckpt",
        "img_size": img_size,
        "single_channel": False,
    }


def dinov2_pup():
    img_size = (1008, 784)

    backbone = DinoVisionTransformer(
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),  # type: ignore
        init_values=1e-5,
        block_chunks=0,
    )

    head = SETR_PUP(embedding_dim=384, num_classes=6)

    model = DinoV2(
        backbone=backbone,
        head=head,
        loss_fn=torch.nn.CrossEntropyLoss(),
        output_shape=(1008, 784),
        middle=False,
    )

    return {
        "name": "dinov2_pup",
        "model": model,
        "ckpt_file": default_ckpt_dir
        / "dinov2_pup"
        / "seam_ai"
        / "checkpoints"
        / "last.ckpt",
        "img_size": img_size,
        "single_channel": False,
    }


# ViT model with 512x512 input size


def sfm_base_patch16():
    img_size = (512, 512)
    model = SFM_BasePatch16_Downstream(
        img_size=img_size, num_classes=6, in_chans=1
    )

    return {
        "name": "sfm_base_patch16",
        "model": model,
        "ckpt_file": default_ckpt_dir
        / "sfm_base_patch16"
        / "seam_ai"
        / "checkpoints"
        / "last.ckpt",
        "img_size": img_size,
        "single_channel": True,
    }
