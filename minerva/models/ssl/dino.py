# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This file incorporates code derived from the DINOv2 and DINOv3 projects.
#
# Portions derived from DINOv2 are licensed under the Apache License,
# Version 2.0.
#
# Portions derived from DINOv3 are distributed under the terms of the
# DINOv3 License Agreement.
#
# See THIRD_PARTY_LICENSES.md for licensing details.
#
# References:
#   https://github.com/facebookresearch/dinov3/blob/main/dinov3/data/collate.py
#   https://github.com/facebookresearch/dinov3/blob/main/dinov3/fsdp/ac_compile_parallelize.py
#   https://github.com/facebookresearch/dinov3/blob/main/dinov3/train/cosine_lr_scheduler.py
#   https://github.com/facebookresearch/dinov3/blob/main/dinov3/train/param_groups.py
#   https://github.com/facebookresearch/dinov3/blob/main/dinov3/train/ssl_meta_arch.py
#   https://github.com/facebookresearch/dinov3/blob/main/dinov3/train/train.py
#   https://github.com/facebookresearch/dinov2/blob/main/dinov2/train/ssl_meta_arch.py
#   https://github.com/facebookresearch/dinov2/blob/main/dinov2/train/train.py


import random
import math
import copy
import time
from functools import partial
from dataclasses import dataclass, field
from typing import Literal, Union, Any, Tuple, Optional, Dict
from pathlib import Path
from collections import defaultdict


import numpy as np
import lightning as L
from lightning.pytorch.strategies import ModelParallelStrategy, DDPStrategy
from lightning.pytorch.strategies.parallel import ParallelStrategy
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
from torch import Tensor, nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh

from minerva.losses.dino import (
    KoLeoLoss,
    KoLeoLossDistributed,
    iBOTPatchLoss,
    GramLoss,
    DINOLoss,
)
from minerva.callback.specific_checkpoint_callback import (
    EvalCheckpointCallback,
    FilterWeights,
)
from minerva.transforms.dino import DINOTransform, MaskingGenerator
from minerva.models.nets.image.dino import DINOHead, wrap_compile_block
from minerva.models.ssl.base import _SSLTechnique
from minerva.utils.instantiators import instantiate_cls


@dataclass
class LossConfig:
    """
    Loss weights and hyperparameter configuration for DINO, KoLeo, and iBOT objectives.

    Parameters
    ----------
    dino_loss_weight : float, default 1.0
        Global multiplier for DINO cross-entropy loss.
    local_loss_weight_schedule : dict, optional
        Schedule configuration for local crops loss reweighting.
    reweight_dino_local_loss : bool, default False
        Whether to apply scheduled dynamic reweighting to local crop loss.
    koleo_loss_distributed : bool, default False
        Whether to compute KoLeo regularizer across distributed worker ranks.
    koleo_loss_weight : float, default 0.1
        Weight multiplier for KoLeo entropy regularization loss.
    koleo_loss_topk : int, default 1
        Top-k nearest neighbors for KoLeo distance calculation.
    koleo_distributed_loss_group_size : int, optional
        Process group size for distributed KoLeo communication.
    ibot_loss_weight : float, default 1.0
        Weight multiplier for iBOT masked patch reconstruction loss.
    ibot_mask_sample_probability : float, default 0.5
        Probability of applying masking to a global crop.
    ibot_mask_random_circular_shift : bool, default False
        Whether to apply random circular shifts to generated patch masks.
    ibot_mask_ratio_min_max : tuple of (float, float), default (0.1, 0.5)
        Minimum and maximum masking ratio boundaries.
    """

    # Dino loss
    dino_loss_weight: float = 1.0
    local_loss_weight_schedule: Optional[Dict[str, int]] = None
    reweight_dino_local_loss: bool = False
    # KoLeo loss
    koleo_loss_distributed: bool = False
    koleo_loss_weight: float = 0.1
    koleo_loss_topk: int = 1
    koleo_distributed_loss_group_size: Optional[int] = None
    # iBot loss
    ibot_loss_weight: float = 1.0
    ibot_mask_sample_probability: float = 0.5
    ibot_mask_random_circular_shift: bool = False
    ibot_mask_ratio_min_max: Tuple[float, float] = (0.1, 0.5)


@dataclass
class GramConfig:
    """
    Configuration for Gram matrix feature representation loss and teacher distillation.

    Parameters
    ----------
    backbone : nn.Module, optional
        Separate backbone architecture for Gram feature extraction.
    use_loss : bool, default False
        Whether to enable Gram representation loss.
    normalized : bool, default True
        Whether to L2-normalize patch representations before Gram computation.
    remove_neg : bool, default False
        Whether to clip negative similarity values in Gram matrices.
    remove_only_teacher_neg : bool, default False
        Whether to clip only negative similarity entries in teacher Gram matrices.
    loss_weight : float, optional
        Static loss multiplier for Gram loss.
    ema_teacher : bool, default False
        Whether to use the EMA teacher backbone as the Gram target teacher.
    ckpt : Path, optional
        Path to pretrained checkpoint for standalone Gram teacher.
    img_level : bool, default False
        Whether to compute Gram similarity across images rather than tokens.
    tokens_used : {"all", "masked", "unmasked"}, default "all"
        Which subset of patch tokens to include in Gram computation.
    rep_update : bool, default True
        Whether to periodically update Gram teacher representations.
    update_frequency : int, default 50000
        Iteration interval for updating Gram teacher weights.
    it_first_update : int, default 0
        First iteration at which Gram teacher updates start.
    max_updates : int, optional
        Maximum number of Gram teacher weight updates allowed.
    num_updates : int, default 0
        Counter tracking total completed Gram teacher updates.
    it_load_ema_teacher : int, default -1
        Iteration step at which to clone EMA teacher weights into Gram teacher.
    compute_stats : bool, default False
        Whether to record masked and unmasked Gram similarity statistics.
    global_teacher_resize_method : {"bicubic"}, default "bicubic"
        Interpolation mode for resizing Gram teacher feature grids.
    loss_weight_schedule : dict, optional
        Dynamic schedule parameters for Gram loss weight.
    global_teacher_resize_antialias : bool, default False
        Whether to apply antialiasing during Gram teacher feature map interpolation.
    teacher_crops_size : int, optional
        Pixel resolution for dedicated Gram teacher crops.
    teacher_no_distortions : bool, default False
        Whether to omit photometric distortions for Gram teacher crops.
    """

    backbone: Optional[nn.Module] = None
    use_loss: bool = False
    normalized: bool = True
    remove_neg: bool = False
    remove_only_teacher_neg: bool = False
    loss_weight: Optional[float] = None
    ema_teacher: bool = False
    ckpt: Optional[Path] = None
    img_level: bool = False
    tokens_used: Literal["all", "masked", "unmasked"] = "all"
    rep_update: bool = True
    update_frequency: int = 50000
    it_first_update: int = 0
    max_updates: Optional[int] = None
    num_updates: int = 0
    it_load_ema_teacher: int = -1
    compute_stats: bool = False
    global_teacher_resize_method: Literal["bicubic"] = "bicubic"
    loss_weight_schedule: Optional[Dict[str, float]] = None
    global_teacher_resize_antialias: bool = False
    teacher_crops_size: Optional[int] = None
    teacher_no_distortions: bool = False


@dataclass
class AugmentationConfig:
    """
    Multi-crop data augmentation parameters for DINO self-supervised training.

    Parameters
    ----------
    global_crops_scale : tuple of (float, float), default (0.32, 1.0)
        Scale range for global image crops.
    local_crops_scale : tuple of (float, float), default (0.05, 0.32)
        Scale range for local image crops.
    local_crops_number : int, default 8
        Number of small local crops generated per sample.
    global_crops_size : int, default 224
        Pixel resolution for square global crops.
    local_crops_size : int, default 96
        Pixel resolution for square local crops.
    localcrops_subset_of_globalcrops : bool, default False
        Whether to crop local views as sub-regions of global crops.
    share_color_jitter : bool, default False
        Whether to share identical color jitter transform parameters across crops.
    horizontal_flips : bool, default True
        Whether to apply random horizontal flips.
    rgb_mean : tuple of (float, float, float), default (0.485, 0.456, 0.406)
        Image channel normalization means.
    rgb_std : tuple of (float, float, float), default (0.229, 0.224, 0.225)
        Image channel normalization standard deviations.
    teacher_to_student_resolution_scale : float, default 1.0
        Ratio of teacher image input resolution to student image resolution.
    """

    global_crops_scale: Tuple[float, float] = (0.32, 1.0)
    local_crops_scale: Tuple[float, float] = (0.05, 0.32)
    local_crops_number: int = 8
    global_crops_size: int = 224
    local_crops_size: int = 96
    localcrops_subset_of_globalcrops: bool = False
    share_color_jitter: bool = False
    horizontal_flips: bool = True
    rgb_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    rgb_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    teacher_to_student_resolution_scale: float = 1.0


@dataclass
class OptimConfig:
    """
    Optimizer, learning rate scheduling, and weight decay hyperparameter configuration.

    Parameters
    ----------
    clip_grad : float, default 3.0
        Maximum gradient norm for gradient clipping.
    scaling_rule : str, default "sqrt_wrt_1024"
        Base learning rate scaling rule ("sqrt_wrt_1024", "linear_wrt_256").
    patch_embed_lr_mult : float, default 0.2
        Learning rate multiplier for patch embedding layer.
    dino_head_wd_multiplier : float, default 1.0
        Weight decay multiplier for DINO projection heads.
    layerwise_decay : float, default 0.9
        Layer-wise learning rate decay rate for transformer blocks.
    multi_tensor_optim : bool, default True
        Whether to fuse and vectorize AdamW parameter updates with multi-tensor ops.
    adamw_beta1 : float, default 0.9
        AdamW beta1 coefficient.
    adamw_beta2 : float, default 0.999
        AdamW beta2 coefficient.
    min_lr : float, default 1.0e-06
        Final minimum learning rate.
    warmup_epochs : int, default 10
        Number of epochs for linear learning rate warmup.
    weight_decay : float, default 0.04
        Initial weight decay value.
    weight_decay_end : float, default 0.4
        Final weight decay value.
    momentum_teacher : float, default 0.992
        Initial exponential moving average (EMA) momentum for teacher network.
    final_momentum_teacher : float, default 1.0
        Final EMA momentum for teacher network.
    teacher_temp : float, default 0.07
        Final softmax temperature for teacher outputs.
    warmup_teacher_temp_epochs : int, default 30
        Number of epochs for teacher temperature warmup.
    warmup_teacher_temp : float, default 0.04
        Initial teacher temperature value.
    schedule_trunc_extra : float, default 0.0
        Cosine decay truncation parameter.
    freeze_last_layer_epochs : int, default 1
        Number of epochs to freeze gradients of the final projection layer.
    freeze_backbone_epochs : int, default 0
        Number of initial epochs to keep the student backbone frozen.
    """

    clip_grad: float = 3.0
    scaling_rule: str = "sqrt_wrt_1024"
    patch_embed_lr_mult: float = 0.2
    dino_head_wd_multiplier: Optional[float] = 1.0
    layerwise_decay: float = 0.9
    multi_tensor_optim: bool = True
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    # Scheduler build
    min_lr: float = 1.0e-06
    warmup_epochs: int = 10
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    momentum_teacher: float = 0.992
    final_momentum_teacher: float = 1
    teacher_temp: float = 0.07
    warmup_teacher_temp_epochs: int = 30
    warmup_teacher_temp: float = 0.04
    schedule_trunc_extra: float = 0.0
    freeze_last_layer_epochs: int = 1
    freeze_backbone_epochs: int = 0


@dataclass
class MiscConfig:
    """
    Miscellaneous training optimizations, distributed execution, and precision settings.

    Parameters
    ----------
    global_ignore_diagonal : bool, default True
        Whether to exclude identical crop comparisons in global DINO loss.
    distillation_enabled : bool, default False
        Whether distillation from an external teacher is enabled.
    multidistillation_enabled : bool, default False
        Whether multi-teacher distillation is enabled.
    train_checkpointing : bool, default False
        Whether to enable activation checkpointing during training.
    train_compile : bool, default True
        Whether to enable torch.compile optimization.
    checkpointing_full : bool, default False
        Whether activation checkpointing wraps full modules or selective ops.
    use_cuda_graphs : bool, default False
        Whether to enable Triton CUDA graphs during compilation.
    param_dtype : {"fp16", "bf16", "fp32"}, default "bf16"
        Data type for model parameters in FSDP.
    reduce_dtype : {"fp16", "bf16", "fp32"}, default "fp32"
        Data type for distributed gradient reduction.
    """

    global_ignore_diagonal: bool = True
    distillation_enabled: bool = False
    multidistillation_enabled: bool = False
    train_checkpointing: bool = False
    train_compile: bool = True
    checkpointing_full: bool = False
    use_cuda_graphs: bool = False
    param_dtype: Literal["fp16", "bf16", "fp32"] = "bf16"
    reduce_dtype: Literal["fp16", "bf16", "fp32"] = "fp32"


@dataclass
class Schedules:
    """Container holding active step-level training schedules for LR, WD, EMA, temperature, and loss weights."""

    lr: Optional[Any] = None
    wd: Optional[Any] = None
    momentum: Optional[Any] = None
    teacher_temp: Optional[Any] = None
    last_layer_lr: Optional[Any] = None
    dino_local_loss: Optional[Any] = None
    gram_loss: Optional[Any] = None


Config = Union[LossConfig, GramConfig, OptimConfig, MiscConfig]


def init_config(config: Config, init: Optional[Union[Config, Dict[str, Any]]]):
    """
    Instantiate or return a typed dataclass configuration object from dictionary or existing instance.

    Parameters
    ----------
    config : type of Config
        Dataclass type to instantiate.
    init : Config or dict, optional
        Configuration instance or keyword dictionary.

    Returns
    -------
    Config
        Instantiated configuration dataclass.
    """
    if init is None:
        return config()
    if isinstance(init, dict):
        return config(**init)
    return init


class _DINO(_SSLTechnique):
    """
    Base LightningModule class implementing DINO self-supervised learning with teacher EMA, iBOT, and Gram objectives.

    Parameters
    ----------
    backbone : nn.Module
        Student feature extractor neural network.
    learning_rate : float
        Base learning rate.
    dino_version : {2, 3}
        Target DINO framework version formulation.
    batch_size : int
        Per-GPU batch size.
    epochs : int
        Total training epochs.
    iter_per_epoch : int
        Number of optimization iterations per epoch.
    prediction_head : DINOHead or nn.Module
        Student projection head.
    ibot_separate_head : bool, default False
        Whether to use a distinct projection head for iBOT patch loss.
    ibot_head : DINOHead or nn.Module, optional
        Separate projection head for iBOT patches.
    teacher_backbone : nn.Module, optional
        Teacher backbone architecture (cloned from student if omitted).
    teacher_prediction_head : DINOHead or nn.Module, optional
        Teacher projection head (cloned from student if omitted).
    centering : {"sinkhorn_knopp", "centering"}, default "sinkhorn_knopp"
        Probability normalization method for teacher output features.
    loss : LossConfig or dict, optional
        Loss configuration.
    gram : GramConfig or dict, optional
        Gram loss configuration.
    crops : AugmentationConfig or dict, optional
        Data augmentation configuration.
    optim : OptimConfig or dict, optional
        Optimizer configuration.
    misc : MiscConfig or dict, optional
        Miscellaneous optimization and precision configuration.
    """

    def __init__(
        self,
        # standard SSL API
        backbone: nn.Module,
        learning_rate: float,
        # Dino Specific API
        dino_version: Literal[2, 3],
        batch_size: int,
        epochs: int,
        iter_per_epoch: int,
        prediction_head: Union[DINOHead, nn.Module],
        ibot_separate_head: bool = False,
        ibot_head: Optional[Union[DINOHead, nn.Module]] = None,
        teacher_backbone: Optional[nn.Module] = None,
        teacher_prediction_head: Optional[Union[DINOHead, nn.Module]] = None,
        centering: Literal["sinkhorn_knopp", "centering"] = "sinkhorn_knopp",
        # DataClass configs
        loss: Optional[Union[LossConfig, Dict[str, Any]]] = None,
        gram: Optional[Union[GramConfig, Dict[str, Any]]] = None,
        crops: Optional[Union[AugmentationConfig, Dict[str, Any]]] = None,
        optim: Optional[Union[OptimConfig, Dict[str, Any]]] = None,
        misc: Optional[Union[MiscConfig, Dict[str, Any]]] = None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
        )
        self.automatic_optimization = False
        self.consecutive_nan_count = 0

        self.dino_version = dino_version
        self.batch_size = batch_size
        self.epochs = epochs
        self.iter_per_epoch = iter_per_epoch
        self.ibot_separate_head = ibot_separate_head
        self.centering = centering

        self.loss = init_config(LossConfig, loss)
        self.gram = init_config(GramConfig, gram)
        self.crops = init_config(AugmentationConfig, crops)
        self.optim = init_config(OptimConfig, optim)
        self.misc = init_config(MiscConfig, misc)
        self.schedules = Schedules()

        # Later init
        self.student = None
        self.teacher = None
        self.model_ema = None
        self.ema_params_lists = None

        self.gram_teacher = None
        self.has_gram_teacher = False
        self.gram_teacher_initialized = False
        self.gram_params_lists = None

        self.embed_dim = None
        self.dino_out_dim = None
        self.ibot_out_dim = None

        # Early validations
        assert self.misc.multidistillation_enabled is False
        assert self.crops.local_crops_number > 0

        assert (
            0
            <= self.loss.ibot_mask_ratio_min_max[0]
            < self.loss.ibot_mask_ratio_min_max[1]
            <= 1
        ), "provide a valid ibot_mask_ratio_min_max"
        assert (
            0 <= self.loss.ibot_mask_sample_probability <= 1
        ), "provide a positive mask probability for ibot"

        for key in ["start", "peak", "end", "warmup_epochs"]:
            if self.loss.local_loss_weight_schedule is not None:
                assert (
                    key in self.loss.local_loss_weight_schedule
                ), f"'local_loss_weight_schedule' must have '{key}' value"
            if self.gram.loss_weight_schedule is not None:
                assert (
                    key in self.gram.loss_weight_schedule
                ), f"'gram.loss_weight_schedule' must have '{key}' value"

        # Get components output shape and validate modules compatibility
        embed_dim = backbone.embed_dim

        dummy_input = torch.randn(1, embed_dim, device="meta")
        output = copy.deepcopy(prediction_head).to("meta")(dummy_input)
        dino_out_dim = output.shape[-1]

        if self.ibot_separate_head:
            assert (
                ibot_head is not None
            ), "ibot_head must be provided when ibot_separate_head is True"
            output = copy.deepcopy(ibot_head).to("meta")(dummy_input)
            ibot_out_dim = output.shape[-1]
        else:
            ibot_out_dim = dino_out_dim

        self.embed_dim = embed_dim  # D
        self.dino_out_dim = dino_out_dim  # K
        self.ibot_out_dim = ibot_out_dim

        # Compose models
        student_model_dict = dict()
        teacher_model_dict = dict()
        gram_model_dict = dict()

        student_model_dict["backbone"] = backbone
        teacher_model_dict["backbone"] = teacher_backbone or copy.deepcopy(backbone)
        gram_model_dict["backbone"] = self.gram.backbone or copy.deepcopy(backbone)

        student_model_dict["dino_head"] = prediction_head
        teacher_model_dict["dino_head"] = teacher_prediction_head or copy.deepcopy(
            prediction_head
        )
        if self.ibot_separate_head:
            student_model_dict["ibot_head"] = ibot_head
            teacher_model_dict["ibot_head"] = copy.deepcopy(ibot_head)

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        self.model_ema = self.teacher  # this may be overwritten for distillation

        # if self.misc.distillation_enabled: # TODO check if distillation is desirable
        #     self._setup_distillation()
        # No grad is needed for these two
        self.teacher.requires_grad_(False)
        self.model_ema.requires_grad_(False)

        # Losses
        self.dino_loss = DINOLoss(self.dino_out_dim, dino_version=self.dino_version)

        if self.loss.koleo_loss_distributed:
            self.koleo_loss = KoLeoLossDistributed(
                topk=self.loss.koleo_loss_topk,
                loss_group_size=self.loss.koleo_distributed_loss_group_size,
            )
        else:
            assert (
                self.loss.koleo_loss_topk == 1
            ), "Non-distributed KoLeo loss only supports `koleo_loss_topk=1`"
            self.koleo_loss = KoLeoLoss()

        self.ibot_loss = iBOTPatchLoss(ibot_out_dim)
        self.gram_loss = GramLoss(
            apply_norm=self.gram.normalized,
            remove_only_teacher_neg=self.gram.remove_only_teacher_neg,
            remove_neg=self.gram.remove_neg,
        )

        # Local loss reweighting
        if self.loss.local_loss_weight_schedule is not None:
            total_iterations = self.iter_per_epoch * self.epochs
            self.schedules.dino_local_loss = linear_warmup_cosine_decay(
                start=self.loss.local_loss_weight_schedule["start"],
                peak=self.loss.local_loss_weight_schedule["peak"],
                end=self.loss.local_loss_weight_schedule["end"],
                warmup_iterations=self.iter_per_epoch
                * self.loss.local_loss_weight_schedule["warmup_epochs"],
                total_iterations=total_iterations,
                cosine_iterations=(
                    self.iter_per_epoch
                    * self.loss.local_loss_weight_schedule["cosine_epochs"]
                    if "cosine_epochs" in self.loss.local_loss_weight_schedule
                    else None
                ),
            )

        if self.gram.use_loss:
            # Construct gram teacher
            self.has_gram_teacher = True if not self.gram.ema_teacher else False
            if self.has_gram_teacher:
                self.gram_teacher = nn.ModuleDict(gram_model_dict)
                self.gram_teacher.requires_grad_(False)
                # logger.info(f"Gram teacher parameter at init: {next(self.gram_teacher.named_parameters())}")

            if self.gram.loss_weight_schedule is not None:
                total_iterations = self.iter_per_epoch * self.epochs
                self.schedules.gram_loss = linear_warmup_cosine_decay(
                    start=self.gram.loss_weight_schedule["start"],
                    peak=self.gram.loss_weight_schedule["peak"],
                    end=self.gram.loss_weight_schedule["end"],
                    warmup_iterations=self.iter_per_epoch
                    * self.gram.loss_weight_schedule["warmup_epochs"],
                    total_iterations=total_iterations,
                    cosine_iterations=(
                        self.iter_per_epoch
                        * self.gram.loss_weight_schedule["cosine_epochs"]
                        if "cosine_epochs" in self.gram.loss_weight_schedule
                        else None
                    ),
                )

            if self.gram.ema_teacher and self.gram.ckpt is not None:
                raise ValueError(
                    "Cannot use both `gram.ema_teacher` and `gram.ckpt` at the same time. Please set one of them to False."
                )
            if self.gram.ckpt is None and self.gram.it_load_ema_teacher < 0:
                raise ValueError(
                    "If no gram checkpoint is provided, `gram.it_load_ema_teacher` must be set to a non-negative value."
                )

            assert not (self.gram.ema_teacher and self.gram.rep_update)
            assert self.gram.tokens_used in ["all", "masked", "unmasked"]
            # Currently using masked/unmasked not handle at the image-level
            if self.gram.tokens_used in ["masked", "unmasked"]:
                assert self.gram.img_level is False

            if self.gram.teacher_crops_size is None and self.has_gram_teacher:
                raise ValueError("gram.teacher_crops_size must be set to use gram loss")
            if self.gram.teacher_crops_size is not None and self.gram.ema_teacher:
                raise ValueError(
                    "gram.teacher_crops_size shoud be None when gram.ema_teacher=True"
                )

    def build_schedulers(self):
        """Build CosineScheduler schedules for learning rate, weight decay, teacher EMA momentum, and temperatures."""
        lr = dict(
            base_value=self.learning_rate,
            final_value=self.optim.min_lr,
            total_iters=self.epochs * self.iter_per_epoch,
            warmup_iters=self.optim.warmup_epochs * self.iter_per_epoch,
            start_warmup_value=0,
            trunc_extra=self.optim.schedule_trunc_extra,
        )
        wd = dict(
            base_value=self.optim.weight_decay,
            final_value=self.optim.weight_decay_end,
            total_iters=self.epochs * self.iter_per_epoch,
            trunc_extra=self.optim.schedule_trunc_extra,
        )
        momentum = dict(
            base_value=self.optim.momentum_teacher,
            final_value=self.optim.final_momentum_teacher,
            total_iters=self.epochs * self.iter_per_epoch,
            trunc_extra=self.optim.schedule_trunc_extra,
        )
        teacher_temp = dict(
            base_value=self.optim.teacher_temp,
            final_value=self.optim.teacher_temp,
            total_iters=self.optim.warmup_teacher_temp_epochs * self.iter_per_epoch,
            warmup_iters=self.optim.warmup_teacher_temp_epochs * self.iter_per_epoch,
            start_warmup_value=self.optim.warmup_teacher_temp,
        )

        lr_schedule = CosineScheduler(**lr)
        wd_schedule = CosineScheduler(**wd)
        momentum_schedule = CosineScheduler(**momentum)
        teacher_temp_schedule = CosineScheduler(**teacher_temp)
        last_layer_lr_schedule = CosineScheduler(**lr)

        last_layer_lr_schedule.schedule[
            : self.optim.freeze_last_layer_epochs * self.iter_per_epoch
        ] = 0  # mimicking the original schedules
        lr_schedule.schedule[
            : self.optim.freeze_backbone_epochs * self.iter_per_epoch
        ] = 0  # mimicking the original schedules

        self.schedules.lr = lr_schedule
        self.schedules.wd = wd_schedule
        self.schedules.momentum = momentum_schedule
        self.schedules.teacher_temp = teacher_temp_schedule
        self.schedules.last_layer_lr = last_layer_lr_schedule

    def configure_model(self):
        """Configure distributed sharding (FSDP/DDP), activation checkpointing, compilation, and LR scaling."""
        self.to_empty(device=self.device)
        self.init_weights()
        inference_only_models = [self.model_ema]
        if self.has_gram_teacher:
            inference_only_models.append(self.gram_teacher)
        if self.misc.distillation_enabled:
            inference_only_models.append(self.teacher)

        all_models = [self.student] + inference_only_models

        # Activation Checkpointing
        if self.misc.train_checkpointing:
            if hasattr(self.student["backbone"], "activation_checkpoint"):
                self.student["backbone"].activation_checkpoint(
                    self.misc.checkpointing_full
                )
            else:
                raise TypeError("Invalid Model Type")

        # Model Compilation
        if self.misc.train_compile:
            for model in all_models:
                for k in model.keys():
                    if k == "backbone":
                        if hasattr(model[k], "compile"):
                            model[k].compile(self.misc.use_cuda_graphs)
                        else:
                            model[k] = wrap_compile_block(
                                model[k],
                                use_cuda_graphs=self.misc.use_cuda_graphs,
                                is_backbone_block=False,
                            )
                    else:
                        model[k] = wrap_compile_block(
                            model[k], use_cuda_graphs=False, is_backbone_block=False
                        )

        strategy = self.trainer.strategy
        if isinstance(strategy, ModelParallelStrategy):
            self.fsdp_strategy()

        elif isinstance(strategy, DDPStrategy):
            self.ddp_strategy()

        if self.optim.scaling_rule == "linear_wrt_256":
            old_lr = self.learning_rate
            self.learning_rate *= self.batch_size * self.trainer.world_size
            print(
                f"linear scaling learning rate; old: {old_lr}, new: {self.learning_rate}"
            )
        elif self.optim.scaling_rule == "sqrt_wrt_1024":
            old_lr = self.learning_rate
            factor = 4 if self.dino_version == 3 else 1
            self.learning_rate *= factor * math.sqrt(
                self.batch_size * self.trainer.world_size / 1024.0
            )
            print(
                f"sqrt scaling learning rate; old: {old_lr}, new: {self.learning_rate}"
            )

        self.build_schedulers()

    def training_step(self, batch, batch_idx):
        """
        Execute single manual optimization step including schedule updates, backward pass, and EMA update.

        Parameters
        ----------
        batch : dict
            Collated training batch from `collate_data_and_cast`.
        batch_idx : int
            Index of current training batch.

        Returns
        -------
        torch.Tensor
            Scalar total training loss.
        """
        step = self.global_step
        opt = self.optimizers()

        # 1. Apply manual schedules for this iteration
        lr = self.schedules.lr[step]
        wd = self.schedules.wd[step]
        mom = self.schedules.momentum[step]
        teacher_temp = self.schedules.teacher_temp[step]
        last_layer_lr = self.schedules.last_layer_lr[step]

        apply_optim_scheduler(opt, lr, wd, last_layer_lr)

        # 2. Pre-forward Gram Teacher Logic
        if self.gram.use_loss and self.gram.it_load_ema_teacher == step:
            self.gram_load_ema_teacher()

        # 3. Forward pass
        opt.zero_grad(set_to_none=True)
        loss, loss_dict = self.forward(batch, teacher_temp=teacher_temp, iteration=step)

        # 4. Manual Backward
        self.manual_backward(loss)

        # 5. Gradient Clipping
        if self.optim.clip_grad:
            for k, v in self.student.items():
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    v.parameters(), max_norm=self.optim.clip_grad
                )
                self.log(f"grad_norm/{k}", grad_norm)

        # 6. NaN Check & Reduction
        if torch.isnan(loss):
            self.consecutive_nan_count += 1
            if (
                self.consecutive_nan_count > 2
                and not self.misc.multidistillation_enabled
            ):
                raise RuntimeError(
                    "Too many consecutive NaNs detected in loss, aborting..."
                )
        else:
            self.consecutive_nan_count = 0

        # 7. Optimizer Step & EMA Update
        opt.step()
        self.update_ema(mom)

        # 8. Post-step Gram Update
        if (
            self.gram.use_loss
            and self.gram.rep_update
            and (step + 1) >= self.gram.it_first_update
            and (step + 1) % self.gram.update_frequency == 0
            and (
                self.gram.max_updates is None
                or self.gram.num_updates < self.gram.max_updates
            )
        ):
            self.update_gram()
            self.gram.num_updates += 1
        timestamp = time.time()

        # 9. Logging
        log_kwargs = {"on_step": True, "on_epoch": False, "sync_dist": True}
        self.log("train/total_loss", loss, **log_kwargs)
        self.log("train/lr", lr, **log_kwargs)
        self.log("train/wd", wd, **log_kwargs)
        self.log("train/mom", mom, **log_kwargs)
        self.log("train/last_layer_lr", last_layer_lr, **log_kwargs)
        for name, value in loss_dict.items():
            self.log(f"train/{name}", value, **log_kwargs)
        self.log("timestamp", timestamp, **log_kwargs)

        return loss

    def ddp_strategy(self):
        """Configure DDP distributed strategy."""
        raise NotImplementedError(
            "DDP strategy is not implemented yet. Please use FSDP strategy."
        )

    def fsdp_strategy(self):
        """Configure PyTorch FSDP2 distributed sharding for student, EMA teacher, and Gram teacher networks."""
        print("Lightning is using FSDP2!")
        inference_only_models = [self.model_ema]
        if self.has_gram_teacher:
            inference_only_models.append(self.gram_teacher)
        if self.misc.distillation_enabled:
            inference_only_models.append(self.teacher)

        all_models = [self.student] + inference_only_models
        # Model Sharding
        DTYPE_MAP = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        mp_policy = MixedPrecisionPolicy(
            param_dtype=DTYPE_MAP[self.misc.param_dtype],
            reduce_dtype=DTYPE_MAP[self.misc.reduce_dtype],
        )

        # Let Lightning provide the world size for the FSDP2 DeviceMesh
        world_size = self.trainer.world_size
        world_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",)
        )
        fsdp_config = {"mesh": world_mesh, "mp_policy": mp_policy}

        for model in all_models:
            for k in model.keys():
                if k == "backbone":
                    if hasattr(model[k], "fsdp"):
                        model[k].fsdp(fsdp_config)
                    else:
                        model[k] = fully_shard(model[k], **fsdp_config)
                else:
                    model[k] = fully_shard(
                        model[k], **fsdp_config, reshard_after_forward=True
                    )

        # 3. Handle inference-only specific state fixes
        for model in inference_only_models:
            for k in model.keys():
                fsdp_state = model[k]._get_fsdp_state()
                if not fsdp_state._fsdp_param_group:
                    continue
                mi = fsdp_state._fsdp_param_group.post_forward_mesh_info
                fsdp_state._lazy_init()
                fsdp_state._fsdp_param_group.post_forward_mesh_info = mi

    def init_weights(self) -> None:
        """Initialize student parameters, projection heads, losses, and sync weights to teacher EMA."""
        # All weights are set to `nan` to ensure we initialize everything explicitly
        self.student.backbone.init_weights()
        self.student.dino_head.init_weights()
        if self.ibot_separate_head:
            self.student.ibot_head.init_weights()
        self.dino_loss.init_weights()
        self.ibot_loss.init_weights()
        self.model_ema.load_state_dict(self.student.state_dict())
        if self.has_gram_teacher:
            self.gram["backbone"].init_weights()
            self.gram_teacher_initialized = True

        # if self.misc.distillation_enabled:
        #     self.teacher.load_state_dict(self.student.state_dict())

    def update_ema(self, m: float):
        """Update teacher network parameters via Exponential Moving Average (EMA)."""
        if self.ema_params_lists is None:
            student_param_list = []
            teacher_param_list = []
            for k in self.student.keys():
                for ms, mt in zip(
                    self.student[k].parameters(), self.model_ema[k].parameters()
                ):
                    student_param_list += [ms]
                    teacher_param_list += [mt]
            self.ema_params_lists = (student_param_list, teacher_param_list)
        else:
            student_param_list, teacher_param_list = self.ema_params_lists
        with torch.no_grad():
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def update_gram(self, m: float = 0):
        """Update Gram teacher network parameters."""
        if not self.has_gram_teacher:
            return
        if self.gram_params_lists is None:
            teacher_param_list = []
            gramteacher_param_list = []
            for k in self.gram_teacher.keys():
                for mgt, mt in zip(
                    self.gram_teacher[k].parameters(), self.teacher[k].parameters()
                ):
                    gramteacher_param_list += [mgt]
                    teacher_param_list += [mt]
            self.gram_params_lists = (gramteacher_param_list, teacher_param_list)
        else:
            gramteacher_param_list, teacher_param_list = self.gram_params_lists

        with torch.no_grad():
            torch._foreach_mul_(gramteacher_param_list, m)
            torch._foreach_add_(gramteacher_param_list, teacher_param_list, alpha=1 - m)

    def get_maybe_fused_params_for_submodel(self, m: nn.Module):
        """Build parameter groups with layerwise decay and optional multi-tensor fusion."""
        params_groups = get_params_groups_with_decay_fsdp(
            model=m,
            lr_decay_rate=self.optim.layerwise_decay,
            patch_embed_lr_mult=self.optim.patch_embed_lr_mult,
            dino_head_wd_multiplier=self.optim.dino_head_wd_multiplier,
        )
        if self.optim.multi_tensor_optim:
            fused_params_groups = fuse_params_groups(params_groups)
            # logger.info("fusing param groups")

            for g in fused_params_groups:
                g["foreach"] = True
                g["fused"] = True
            return fused_params_groups
        else:
            return params_groups

    def get_params_groups(self):
        """Retrieve optimizer parameter groups across all student submodules."""
        all_params_groups = []
        for name, m in self.student.items():
            # logger.info(f"Getting paramer groups for {name}")
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def configure_optimizers(self):
        """Configure AdamW optimizer for the student network."""
        # Ensure only student params are passed
        return torch.optim.AdamW(
            self.get_params_groups(),
            betas=(self.optim.adamw_beta1, self.optim.adamw_beta2),
        )

    def default_technique_transforms(self):
        """Return the default multi-crop ``DINOTransform`` (dict in, dict out) applied to ``"image"``."""
        return DINOTransform(
            ["image"],
            self.crops.global_crops_scale,
            self.crops.local_crops_scale,
            self.crops.local_crops_number,
            global_crops_size=self.crops.global_crops_size,
            local_crops_size=self.crops.local_crops_size,
            gram_teacher_crops_size=self.gram.teacher_crops_size,
            gram_teacher_no_distortions=self.gram.teacher_no_distortions,
            local_crops_subset_of_global_crops=self.crops.localcrops_subset_of_globalcrops,
            share_color_jitter=self.crops.share_color_jitter,
            horizontal_flips=self.crops.horizontal_flips,
            mean=self.crops.rgb_mean,
            std=self.crops.rgb_std,
        )

    def technique_callbacks(self, logs_dir: Path):
        """
        Return technique-specific Lightning callbacks (evaluation checkpointing, weights filtering).

        Parameters
        ----------
        logs_dir : Path
            Logging output root directory.

        Returns
        -------
        list of Callback
            List of configured Lightning callbacks.
        """
        custom_callbacks = [
            EvalCheckpointCallback(
                period=self.iter_per_epoch * 10,
                name="teacher_checkpoint",
                model_attr="model_ema",
            ),
            ModelCheckpoint(
                dirpath=logs_dir / "ckpt",
                filename="{step}",
                every_n_train_steps=self.iter_per_epoch * 3,
                save_top_k=3,
                monitor="step",  # Use step to determine the "top K" (latest)
                mode="max",
                save_last="link",
            ),
        ]

        if self.misc.multidistillation_enabled:
            custom_callbacks.insert(1, FilterWeights(filter_values="teacher"))
        return custom_callbacks

    def default_technique_collate_fn(self):
        """Return default collate and batch-masking function for dataloader."""
        img_size = self.crops.global_crops_size
        patch_size = int(
            self.student.backbone.patch_size
            * self.crops.teacher_to_student_resolution_scale
        )
        n_tokens = (img_size // patch_size) ** 2
        mask_generator = MaskingGenerator(
            input_size=(img_size // patch_size, img_size // patch_size),
            max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
        )
        return partial(
            collate_data_and_cast,
            mask_ratio_tuple=self.loss.ibot_mask_ratio_min_max,
            mask_probability=self.loss.ibot_mask_sample_probability,
            dtype={
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }[self.misc.param_dtype],
            n_tokens=n_tokens,
            mask_generator=mask_generator,
            random_circular_shift=self.loss.ibot_mask_random_circular_shift,
            local_batch_size=None,
            dino_version=self.dino_version,
        )

    def train(self, mode: bool = True):
        """Set module training mode while maintaining teacher networks in eval mode."""
        super().train(mode)
        self.teacher.eval()
        if self.has_gram_teacher:
            self.gram_teacher.eval()
        return self

    def default_train_strategy(self, world_size: int) -> ParallelStrategy:
        """Return default ModelParallelStrategy for distributed training."""
        return ModelParallelStrategy(
            data_parallel_size=world_size,
            tensor_parallel_size=1,
        )


class DINOv2(_DINO):
    """
    LightningModule implementation of DINOv2 self-supervised learning with iBOT and KoLeo objectives.

    Parameters
    ----------
    backbone : nn.Module
        Student feature extractor neural network.
    learning_rate : float
        Base learning rate.
    batch_size : int
        Per-GPU batch size.
    epochs : int
        Total training epochs.
    iter_per_epoch : int
        Number of optimization iterations per epoch.
    prediction_head : DINOHead or nn.Module
        Student DINO projection head.
    ibot_separate_head : bool, default False
        Whether to use an independent projection head for iBOT loss.
    ibot_head : DINOHead or nn.Module, optional
        Separate projection head for iBOT patches.
    teacher_backbone : nn.Module, optional
        Teacher backbone architecture (cloned from student if omitted).
    teacher_prediction_head : DINOHead or nn.Module, optional
        Teacher projection head (cloned from student if omitted).
    centering : {"sinkhorn_knopp", "centering"}, default "sinkhorn_knopp"
        Probability normalization method for teacher output features.
    loss : LossConfig or dict, optional
        Loss configuration.
    crops : AugmentationConfig or dict, optional
        Data augmentation configuration.
    optim : OptimConfig or dict, optional
        Optimizer configuration.
    misc : MiscConfig or dict, optional
        Miscellaneous configuration.
    """

    def __init__(
        self,
        # standard SSL API
        backbone: nn.Module,
        learning_rate: float,
        # Dino Specific API
        batch_size: int,
        epochs: int,
        iter_per_epoch: int,
        prediction_head: Union[DINOHead, nn.Module],
        ibot_separate_head: bool = False,
        ibot_head: Optional[Union[DINOHead, nn.Module]] = None,
        teacher_backbone: Optional[nn.Module] = None,
        teacher_prediction_head: Optional[Union[DINOHead, nn.Module]] = None,
        centering: Literal["sinkhorn_knopp", "centering"] = "sinkhorn_knopp",
        # DataClass configs
        loss: Optional[Union[LossConfig, Dict[str, Any]]] = None,
        crops: Optional[Union[AugmentationConfig, Dict[str, Any]]] = None,
        optim: Optional[Union[OptimConfig, Dict[str, Any]]] = None,
        misc: Optional[Union[MiscConfig, Dict[str, Any]]] = None,
        **kwargs,
    ):

        loss = init_config(LossConfig, loss)
        crops = init_config(AugmentationConfig, crops)
        optim = init_config(OptimConfig, optim)
        misc = init_config(MiscConfig, misc)

        loss.koleo_loss_topk = 1
        loss.ibot_mask_random_circular_shift = False
        crops.teacher_to_student_resolution_scale = 1
        optim.schedule_trunc_extra = 0.0
        optim.multi_tensor_optim = True
        optim.dino_head_wd_multiplier = 0.0
        misc.train_checkpointing = False
        misc.train_compile = False
        local_vars = {
            k: v for k, v in locals().items() if k not in {"self", "__class__"}
        }
        super().__init__(
            **local_vars,
            dino_version=2,
        )

    @torch.no_grad()
    def get_teacher_output(
        self,
        images: Tensor,
        n_images: int,
        upperbound: int,
        mask_indices_list: Tensor,
        teacher_temp: float,
        n_masked_patches_tensor: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward teacher network on global views and compute centered softmax target probabilities.

        Parameters
        ----------
        images : Tensor
            Flattened global crop images of shape `(n_images * B, C, H, W)`.
        n_images : int
            Number of global crops (typically 2).
        upperbound : int
            Upper bound capacity for patch token buffer.
        mask_indices_list : Tensor
            Flat indices of masked patches across all global crops.
        teacher_temp : float
            Softmax temperature for teacher probabilities.
        n_masked_patches_tensor : Tensor
            Tensor recording the total number of masked patches.

        Returns
        -------
        teacher_dino_softmaxed_centered_list : Tensor
            Centered and sharpened teacher DINO probabilities.
        masked_teacher_ibot_softmaxed_centered : Tensor, optional
            Centered and sharpened teacher iBOT patch probabilities.
        """
        n_masked_patches = mask_indices_list.shape[0]
        x, n_global_crops_teacher = images, n_images
        teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
        teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
        teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
        # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
        teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
        ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
        _dim = ibot_teacher_patch_tokens.shape[-1]
        n_cls_tokens = teacher_cls_tokens.shape[0]

        if self.loss.ibot_loss_weight > 0 and not self.ibot_separate_head:
            buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(
                upperbound + n_cls_tokens, _dim
            )
            buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
            torch.index_select(
                ibot_teacher_patch_tokens.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
                out=buffer_tensor_teacher[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ],
            )
            tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
            teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
            masked_teacher_patch_tokens_after_head = tokens_after_head[
                n_cls_tokens : n_cls_tokens + n_masked_patches
            ]
        elif self.loss.ibot_loss_weight > 0 and self.ibot_separate_head:
            buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(
                upperbound, _dim
            )
            torch.index_select(
                ibot_teacher_patch_tokens.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
                out=buffer_tensor_teacher[:n_masked_patches],
            )
            teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
            masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(
                buffer_tensor_teacher
            )[:n_masked_patches]
        else:
            teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
            masked_teacher_ibot_softmaxed_centered = None

        if self.centering == "centering":
            teacher_dino_softmaxed_centered_list = (
                self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(
                    n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:]
                )
            )
            self.dino_loss.update_center(teacher_cls_tokens_after_head)
            if self.loss.ibot_loss_weight > 0:
                masked_teacher_patch_tokens_after_head = (
                    masked_teacher_patch_tokens_after_head.unsqueeze(0)
                )
                masked_teacher_ibot_softmaxed_centered = (
                    self.ibot_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches],
                        teacher_temp=teacher_temp,
                    )
                )
                masked_teacher_ibot_softmaxed_centered = (
                    masked_teacher_ibot_softmaxed_centered.squeeze(0)
                )
                self.ibot_loss.update_center(
                    masked_teacher_patch_tokens_after_head[:n_masked_patches]
                )

        elif self.centering == "sinkhorn_knopp":
            teacher_dino_softmaxed_centered_list = (
                self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(
                    n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:]
                )
            )

            if self.loss.ibot_loss_weight > 0:
                masked_teacher_ibot_softmaxed_centered = (
                    self.ibot_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )
                )

        else:
            raise NotImplementedError

        return (
            teacher_dino_softmaxed_centered_list,
            masked_teacher_ibot_softmaxed_centered,
        )

    def forward(
        self, data: Dict[str, Any], teacher_temp: float, **kwargs
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute DINOv2 forward pass, evaluating DINO, iBOT, and KoLeo losses across multi-crop views.

        Parameters
        ----------
        data : dict
            Batch dictionary with collated crops, masks, and indexing metadata.
        teacher_temp : float
            Current softmax temperature for teacher distribution sharpening.

        Returns
        -------
        loss_accumulator : Tensor
            Scalar weighted loss for backpropagation.
        loss_dict : dict of str to Tensor
            Dictionary of individual loss term values for logging.
        """
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.crops.local_crops_number

        global_crops = data["collated_global_crops"]
        local_crops = data["collated_local_crops"]

        masks = data["collated_masks"]
        mask_indices_list = data["mask_indices_list"]
        n_masked_patches_tensor = data["n_masked_patches"]
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = data["upperbound"]
        masks_weight = data["masks_weight"]

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = (
            self.get_teacher_output(
                images=global_crops,
                n_images=n_global_crops,
                upperbound=upperbound,
                mask_indices_list=mask_indices_list,
                teacher_temp=teacher_temp,
                n_masked_patches_tensor=n_masked_patches_tensor,
            )
        )

        loss_dict = {}

        loss_accumulator = 0  # for backprop
        student_global_backbone_output_dict, student_local_backbone_output_dict = (
            self.student.backbone(
                [global_crops, local_crops], masks=[masks, None], is_training=True
            )
        )

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict[
            "x_norm_clstoken"
        ]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if self.loss.ibot_loss_weight > 0:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict[
                "x_norm_patchtokens"
            ]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(
                upperbound, _dim
            )
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(
                    ibot_student_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                )
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(
                    buffer_tensor_patch_tokens.unsqueeze(0)
                )
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(
                    buffer_tensor_patch_tokens
                )[:n_masked_patches]

        # 2: run
        # old implementation that uses a fmha hack replaced with a PyTorch 2.x implementation
        # _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        # outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        sizes = [x.shape[1] for x in inputs_for_student_head_list]
        cat_inputs = torch.cat(inputs_for_student_head_list, dim=1)
        head_outputs = self.student.dino_head(cat_inputs)
        outputs_list = list(torch.split(head_outputs, sizes, dim=1))

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if self.loss.ibot_loss_weight > 0 and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(
                0
            )[:n_masked_patches]

        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(
                    n_local_crops
                ),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.loss.dino_loss_weight * dino_local_crops_loss

        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        if self.loss.dino_loss_weight > 0:
            # compute loss
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

            # accumulate loss
            loss_accumulator += self.loss.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.loss.koleo_loss_weight > 0:
                koleo_loss = self.loss.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually

        if self.loss.ibot_loss_weight > 0:
            # compute loss
            ibot_patch_loss = (
                self.ibot_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2

            # accumulate loss
            loss_accumulator += self.loss.ibot_loss_weight * ibot_patch_loss

        return loss_accumulator, loss_dict


class DINOv3(_DINO):
    """
    LightningModule implementation of DINOv3 self-supervised learning with Gram matrix feature distillation.

    Parameters
    ----------
    backbone : nn.Module
        Student feature extractor neural network.
    learning_rate : float
        Base learning rate.
    batch_size : int
        Per-GPU batch size.
    epochs : int
        Total training epochs.
    iter_per_epoch : int
        Number of optimization iterations per epoch.
    prediction_head : DINOHead or nn.Module
        Student DINO projection head.
    ibot_separate_head : bool, default True
        Whether to use an independent projection head for iBOT loss (must be True for DINOv3).
    ibot_head : DINOHead or nn.Module, optional
        Separate projection head for iBOT patches.
    teacher_backbone : nn.Module, optional
        Teacher backbone architecture (cloned from student if omitted).
    teacher_prediction_head : DINOHead or nn.Module, optional
        Teacher projection head (cloned from student if omitted).
    centering : {"sinkhorn_knopp"}, default "sinkhorn_knopp"
        Probability normalization method for teacher output features.
    loss : LossConfig or dict, optional
        Loss configuration.
    gram : GramConfig or dict, optional
        Gram loss configuration.
    crops : AugmentationConfig or dict, optional
        Data augmentation configuration.
    optim : OptimConfig or dict, optional
        Optimizer configuration.
    misc : MiscConfig or dict, optional
        Miscellaneous configuration.
    """

    def __init__(
        self,
        # standard SSL API
        backbone: nn.Module,
        learning_rate: float,
        # Dino Specific API
        batch_size: int,
        epochs: int,
        iter_per_epoch: int,
        prediction_head: Union[DINOHead, nn.Module],
        ibot_separate_head: bool = False,
        ibot_head: Optional[Union[DINOHead, nn.Module]] = None,
        teacher_backbone: Optional[nn.Module] = None,
        teacher_prediction_head: Optional[Union[DINOHead, nn.Module]] = None,
        centering: Literal["sinkhorn_knopp", "centering"] = "sinkhorn_knopp",
        # DataClass configs
        loss: Optional[Union[LossConfig, Dict[str, Any]]] = None,
        gram: Optional[Union[GramConfig, Dict[str, Any]]] = None,
        crops: Optional[Union[AugmentationConfig, Dict[str, Any]]] = None,
        optim: Optional[Union[OptimConfig, Dict[str, Any]]] = None,
        misc: Optional[Union[MiscConfig, Dict[str, Any]]] = None,
        **kwargs,
    ):
        assert ibot_separate_head is True
        assert centering == "sinkhorn_knopp"
        local_vars = locals()
        local_vars = {
            key: value
            for key, value in local_vars.items()
            if (key != "self" and key != "__class__")
        }
        super().__init__(
            **local_vars,
            dino_version=3,
        )

    def forward(
        self,
        data: Dict[str, Any],
        teacher_temp: float,
        iteration: int = 0,
    ) -> Tuple[Tensor, Dict[str, Union[float, Tensor]]]:
        """
        Compute DINOv3 forward pass across global, local, and optional Gram teacher views.

        Parameters
        ----------
        data : dict
            Collated batch dictionary with views, masks, and index metadata.
        teacher_temp : float
            Softmax temperature for teacher probability centering.
        iteration : int, default 0
            Current optimization iteration step.

        Returns
        -------
        loss : Tensor
            Scalar weighted loss for backpropagation.
        loss_dict : dict of str to (float or Tensor)
            Individual loss components and tracking statistics.
        """
        # del ignored_kwargs
        # metrics_dict = {}
        # print(type(data))

        # Shapes
        n_global_crops = 2
        n_local_crops = self.crops.local_crops_number
        B = data["collated_local_crops"].shape[0] // n_local_crops

        # first_image_sum = data["collated_local_crops"][0].sum().item()
        # print(f"Rank {self.trainer.global_rank} - First Image Sum: {first_image_sum}")

        assert data["collated_global_crops"].shape[0] == n_global_crops * B
        # metrics_dict["local_batch_size"] = B
        # metrics_dict["global_batch_size"] = data["global_batch_size"]

        global_crops = data["collated_global_crops"]
        local_crops = data["collated_local_crops"]
        masks = data["collated_masks"]
        mask_indices_list = data["mask_indices_list"]
        masks_weight = data["masks_weight"]
        n_masked_patches_tensor = data["n_masked_patches"]

        if self.has_gram_teacher:
            assert (
                "collated_gram_teacher_crops" in data
            ), "no gram teacher crops in the data, have you set gram_teacher_crops_size?"
            gram_teacher_crops = data["collated_gram_teacher_crops"]
        else:
            gram_teacher_crops = None

        # Teacher output (will trigger an all-gather to unshard)
        teacher_global = self.get_teacher_output(
            global_crops.unflatten(0, (n_global_crops, B)),
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
            mask_indices_list=mask_indices_list,
            upperbound=data["upperbound"],
        )

        # Student output (will trigger an all-gather to unshard)
        student_global, student_local = self.get_student_output(
            global_crops=global_crops.unflatten(0, (n_global_crops, B)),
            local_crops=local_crops.unflatten(0, (n_local_crops, B)),
            upperbound=data["upperbound"],
            masks=masks,
            mask_indices_list=mask_indices_list,
        )
        with torch.no_grad():
            student_variance = torch.var(student_global["cls_after_head"], dim=0).mean()
            if self._trainer is not None:
                self.log(
                    "student_feature_variance",
                    student_variance.item(),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                )

        # Gram output
        if self.gram.use_loss:
            gram_global = self.get_gram_teacher_output(
                (
                    gram_teacher_crops.unflatten(0, (n_global_crops, B))
                    if gram_teacher_crops is not None
                    else None
                ),
                masks=masks,
                teacher_global=teacher_global,
                student_global=student_global,
                student_global_crops_size=global_crops.shape[-1],
            )
        else:
            gram_global = {}

        # Compute losses and backprop
        loss, loss_dict = self.compute_losses(
            teacher_global=teacher_global,
            student_global=student_global,
            student_local=student_local,
            gram_global=gram_global,
            masks=masks,
            mask_indices_list=mask_indices_list,
            masks_weight=masks_weight,
            iteration=iteration,
        )

        # Return total weighted loss
        return loss, loss_dict

    @torch.no_grad()
    def get_teacher_output(
        self,
        images: Tensor,
        upperbound: int,
        mask_indices_list: Tensor,
        teacher_temp: float,
        n_masked_patches_tensor: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Forward teacher network on unflattened global crops and compute Sinkhorn-Knopp targets.

        Parameters
        ----------
        images : Tensor
            Teacher crop images of shape `(n_crops, B, C, H, W)`.
        upperbound : int
            Upper bound capacity for token buffering.
        mask_indices_list : Tensor
            Indices of student-masked patches.
        teacher_temp : float
            Temperature for Sinkhorn-Knopp normalization.
        n_masked_patches_tensor : Tensor
            Tensor of masked patch count.

        Returns
        -------
        dict of str to Tensor
            Dictionary containing CLS, register, and patch embeddings before and after projection and centering.
        """
        n_crops, B, rgb, H, W = images.shape
        images = images.flatten(0, 1)

        backbone_out = self.teacher.backbone(images, is_training=True)
        cls = backbone_out["x_norm_clstoken"]  # [n_crops * B, D]
        reg = backbone_out["x_storage_tokens"]  # [n_crops * B, R, D]
        ibot_patch = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P, D]

        # IBOT head only on patches that are masked for the student
        buffer = torch.index_select(
            ibot_patch.flatten(0, 1), dim=0, index=mask_indices_list
        )
        masked_patch_after_head = self.teacher.ibot_head(buffer)

        # DINO head on CLS tokens
        cls_after_head = self.teacher.dino_head(cls)  # [n_crops * B, K]

        # Center with sinkhorn-knopp
        cls_centered = self.dino_loss.sinkhorn_knopp_teacher(
            cls_after_head,
            teacher_temp=teacher_temp,
        )  # [n_crops * B, K]
        cls_centered = cls_centered.unflatten(0, (n_crops, B))  # [n_crops, B, K]
        masked_patch_centered = self.ibot_loss.sinkhorn_knopp_teacher(
            masked_patch_after_head,
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
        )  # [n_masked_patches, K]

        return {
            "cls_pre_head": cls.unflatten(0, [n_crops, B]),  # [n_crops, B, D]
            "reg_pre_head": reg.unflatten(0, [n_crops, B]),  # [n_crops, B, R, D]
            "patch_pre_head": ibot_patch.unflatten(
                0, [n_crops, B]
            ),  # [n_crops, B, P, D]
            "cls_after_head": cls_after_head.unflatten(
                0, [n_crops, B]
            ),  # [n_crops, B, K]
            "cls_centered": cls_centered,  # [n_crops, B, K]
            "masked_patch_centered": masked_patch_centered,  # [n_masked_patches, K]
        }

    def get_gram_teacher_output(
        self,
        images: Optional[Tensor],
        masks: Tensor,
        teacher_global: Dict[str, Tensor],
        student_global: Dict[str, Tensor],
        student_global_crops_size: int,
    ) -> Dict[str, Tensor]:
        """
        Extract Gram representation targets from standalone Gram teacher or EMA teacher backbone.

        Parameters
        ----------
        images : Tensor, optional
            Dedicated Gram teacher crops of shape `(n_crops, B, C, H, W)`.
        masks : Tensor
            Boolean mask tensor indicating student-masked patch locations.
        teacher_global : dict of str to Tensor
            Outputs from the standard EMA teacher network.
        student_global : dict of str to Tensor
            Outputs from the student network on global crops.
        student_global_crops_size : int
            Pixel resolution of student global crops.

        Returns
        -------
        dict of str to Tensor
            Dictionary containing student and teacher patch feature tensors for Gram loss computation.
        """
        # Get student patch features
        student_patches = student_global["patch_pre_head"].flatten(
            0, 1
        )  # [n_crops * B, P, D]

        # Get gram targets
        if self.gram.ema_teacher:
            teacher_patches = teacher_global["patch_pre_head"].flatten(
                0, 1
            )  # [n_crops * B, P, D]
        else:
            if not self.gram_teacher_initialized:
                raise ValueError(
                    "Gram teacher has not been initialized. Load a checkpoint or from the EMA teacher."
                )
            n_crops, B, rgb, H, W = images.shape
            images = images.flatten(0, 1)  # [n_crops * B, rgb, H, W]

            with torch.no_grad():
                backbone_out = self.gram_teacher.backbone(images, is_training=True)
            teacher_patches = backbone_out[
                "x_norm_patchtokens"
            ]  # [n_crops * B, P_T, D]

            # Downsample Gram teacher features if needed
            if teacher_patches.shape[1] != student_patches.shape[1]:
                N = H // self.student.backbone.patch_size
                assert teacher_patches.shape[1] == N**2
                N_student = (
                    student_global_crops_size // self.student.backbone.patch_size
                )
                assert student_patches.shape[1] == N_student**2
                patches_hw = teacher_patches.transpose(-2, -1).unflatten(
                    -1, (N, N)
                )  # [n_crops * B, D, N, N]
                patches_hw = torch.nn.functional.interpolate(
                    patches_hw,
                    size=(N_student, N_student),
                    mode=self.gram.global_teacher_resize_method,
                    align_corners=False,
                    antialias=self.gram.global_teacher_resize_antialias,
                )
                teacher_patches = patches_hw.flatten(-2, -1).transpose(
                    -2, -1
                )  # [n_crops * B, N_student * N_student, D]
                assert teacher_patches.shape == student_patches.shape

        # Select the patches to be considered in the loss
        orig_student_patches = student_patches
        orig_teacher_patches = teacher_patches
        if self.gram.tokens_used == "masked":
            student_patches = student_patches[masks]
            teacher_patches = teacher_patches[masks]
        elif self.gram.tokens_used == "unmasked":
            student_patches = student_patches[~masks]
            teacher_patches = teacher_patches[~masks]

        return {
            "student_patches": student_patches,  # [n_crops * B, P, D] or [n_selected_patches, D]
            "teacher_patches": teacher_patches,  # [n_crops * B, P, D] or [n_selected_patches, D]
            # Unmasked patches, for computing statistics
            "orig_student_patches": orig_student_patches,  # [n_crops * B, P, D]
            "orig_teacher_patches": orig_teacher_patches,  # [n_crops * B, P, D]
        }

    def get_student_output(
        self,
        *,
        global_crops: Tensor,
        local_crops: Tensor,
        upperbound: int,
        masks: Tensor,
        mask_indices_list: Tensor,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Forward global and local crops jointly through student backbone and projection heads.

        Parameters
        ----------
        global_crops : Tensor
            Global crop images of shape `(n_global_crops, B, C, H, W)`.
        local_crops : Tensor
            Local crop images of shape `(n_local_crops, B, C, H, W)`.
        upperbound : int
            Upper bound capacity for patch token buffer.
        masks : Tensor
            Boolean mask tensor for student global crops.
        mask_indices_list : Tensor
            Indices of masked patches.

        Returns
        -------
        global_out : dict of str to Tensor
            Student outputs on global crops (CLS, storage tokens, masked patch tokens).
        local_out : dict of str to Tensor
            Student outputs on local crops.
        """
        n_global_crops, B, rgb, H, W = global_crops.shape
        n_local_crops, B, rgb, H, W = local_crops.shape

        global_crops = global_crops.flatten(0, 1)

        # Forward global and local crops through the student backbone jointly
        global_out, local_out = self.student.backbone(
            [global_crops, local_crops.flatten(0, 1)],
            masks=[masks if not self.misc.distillation_enabled else None, None],
            is_training=True,
        )
        g_cls, g_reg, g_patch = (
            global_out["x_norm_clstoken"],
            global_out["x_storage_tokens"],
            global_out["x_norm_patchtokens"],
        )
        l_cls, l_reg, l_patch = (
            local_out["x_norm_clstoken"],
            local_out["x_storage_tokens"],
            local_out["x_norm_patchtokens"],
        )

        # IBOT head only on masked patches
        masked_patches_pre_head = torch.index_select(
            g_patch.flatten(0, 1), dim=0, index=mask_indices_list
        )
        global_masked_patch_after_head = self.student.ibot_head(masked_patches_pre_head)

        # DINO head on CLS tokens (all in one pass)
        buffer = [
            g_cls,  # [n_global_crops * B, D]
            l_cls,  # [n_local_crops * B, D]
        ]
        sizes = [x.shape[0] for x in buffer]
        buffer = torch.cat(buffer, dim=0)  # [n_global_crops * B + n_local_crops * B, D]
        buffer = self.student.dino_head(
            buffer
        )  # [n_global_crops * B + n_local_crops * B, K]
        buffer = torch.split_with_sizes(buffer, sizes, dim=0)

        global_out = {
            "cls_pre_head": g_cls.unflatten(
                0, [n_global_crops, B]
            ),  # [n_global_crops, B, D]
            "reg_pre_head": g_reg.unflatten(
                0, [n_global_crops, B]
            ),  # [n_global_crops, B, R, D]
            "patch_pre_head": g_patch.unflatten(
                0, [n_global_crops, B]
            ),  # [n_global_crops, B, P, D]
            "cls_after_head": buffer[0].unflatten(
                0, [n_global_crops, B]
            ),  # [n_global_crops, B, K],
            "masked_patch_after_head": global_masked_patch_after_head,  # [n_masked_patches, K]
            "masked_patch_pre_head": masked_patches_pre_head,  # [n_masked_patches, D]
        }
        local_out = {
            "cls_pre_head": l_cls.unflatten(
                0, [n_local_crops, B]
            ),  # [n_local_crops, B, D]
            "reg_pre_head": l_reg.unflatten(
                0, [n_local_crops, B]
            ),  # [n_local_crops, B, R, D]
            "patch_pre_head": l_patch.unflatten(
                0, [n_local_crops, B]
            ),  # [n_local_crops, B, P, D]
            "cls_after_head": buffer[1].unflatten(
                0, [n_local_crops, B]
            ),  # [n_local_crops, B, K],
        }

        return global_out, local_out

    def compute_losses(
        self,
        teacher_global: Dict[str, Tensor],
        student_global: Dict[str, Tensor],
        student_local: Dict[str, Tensor],
        gram_global: Dict[str, Tensor],
        masks: Tensor,
        mask_indices_list: Tensor,
        masks_weight: Tensor,
        iteration: int,
    ) -> Tuple[Tensor, Dict[str, Union[float, Tensor]]]:
        """
        Calculate total weighted self-supervised loss combining DINO, KoLeo, iBOT, and Gram objectives.

        Parameters
        ----------
        teacher_global : dict of str to Tensor
            Teacher network features and centered target distributions.
        student_global : dict of str to Tensor
            Student network representations on global crops.
        student_local : dict of str to Tensor
            Student network representations on local crops.
        gram_global : dict of str to Tensor
            Extracted patch features for Gram representation alignment.
        masks : Tensor
            Flat student mask tensor across global crops.
        mask_indices_list : Tensor
            Indices of masked patches.
        masks_weight : Tensor
            Inverse frequency weights per mask.
        iteration : int
            Current global optimization step.

        Returns
        -------
        loss_accumulator : Tensor
            Total combined scalar loss for backpropagation.
        loss_dict : dict of str to (float or Tensor)
            Dictionary of named individual loss components and statistics.
        """
        # print(student_local["cls_after_head"].dtype)
        # print(teacher_global["cls_centered"].dtype)
        n_global_crops = student_global["cls_after_head"].shape[0]
        n_local_crops = student_local["cls_after_head"].shape[0]
        loss_dict = {}
        loss_accumulator = 0.0

        # Loss scales like in DINOv2, these are multiplied with the loss weights from the config
        dino_global_terms = (
            n_global_crops * (n_global_crops - 1)
            if self.misc.global_ignore_diagonal
            else n_global_crops**2
        )
        dino_local_terms = n_global_crops * n_local_crops
        dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
        dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)
        koleo_scale = n_global_crops

        # DINO local loss: compare post-head CLS tokens: student(local crops) vs. teacher(global crops)
        dino_local_crops_loss = self.dino_loss(
            student_logits=student_local["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
        )
        loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

        # Reweighting of DINO loss
        if self.loss.reweight_dino_local_loss:
            local_weight = self.schedules.dino_local_loss[iteration]
        else:
            local_weight = 1.0

        loss_dict["dino_local_loss_weight"] = local_weight
        loss_accumulator += (
            self.loss.dino_loss_weight
            * dino_local_scale
            * local_weight
            * dino_local_crops_loss
        )

        # DINO global loss: compare post-head CLS tokens: student(global crops) vs. teacher(global crops)
        dino_global_crops_loss = self.dino_loss(
            student_logits=student_global["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
            ignore_diagonal=self.misc.global_ignore_diagonal,
        )

        loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
        loss_accumulator += (
            self.loss.dino_loss_weight * dino_global_scale * dino_global_crops_loss
        )

        # Koleo: regularize pre-head CLS tokens of student(global crops)
        koleo_loss = (
            sum(self.koleo_loss(x) for x in student_global["cls_pre_head"])
            / n_global_crops
        )
        loss_dict["koleo_loss"] = koleo_loss
        loss_accumulator += self.loss.koleo_loss_weight * koleo_scale * koleo_loss

        # IBOT loss
        ibot_patch_loss = self.ibot_loss.forward_masked(
            student_global["masked_patch_after_head"],
            teacher_global["masked_patch_centered"],
            student_masks_flat=masks,
            n_masked_patches=mask_indices_list.shape[0],
            masks_weight=masks_weight,
        )
        loss_dict["ibot_loss"] = ibot_patch_loss
        loss_accumulator += self.loss.ibot_loss_weight * ibot_patch_loss

        # Gram loss
        if self.gram.use_loss:
            gram_loss = self.gram_loss(
                gram_global["student_patches"],
                gram_global["teacher_patches"],
                img_level=self.gram.img_level,
            )

            if self.schedules.gram_loss is not None:
                gram_loss_weight = self.schedules.gram_loss[iteration]
            else:
                gram_loss_weight = self.gram.loss_weight

            loss_dict["gram_loss_weight"] = gram_loss_weight
            loss_accumulator += gram_loss * gram_loss_weight
            loss_dict["gram_loss"] = gram_loss

            if self.gram.compute_stats:
                with torch.no_grad():
                    # Save stats over masked / unmasked tokens
                    gram_loss_masked = self.gram_loss(
                        gram_global["orig_student_patches"][masks].detach(),
                        gram_global["orig_teacher_patches"][masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/masked_gram_loss"] = gram_loss_masked
                    gram_loss_unmasked = self.gram_loss(
                        gram_global["orig_student_patches"][~masks].detach(),
                        gram_global["orig_teacher_patches"][~masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/unmasked_gram_loss"] = gram_loss_unmasked

        return loss_accumulator, loss_dict

    @torch.no_grad()
    def gram_load_ema_teacher(self):
        """Clone EMA teacher backbone weights into the standalone Gram teacher network."""
        if self.has_gram_teacher:
            skip_load_prefixes = ["dino_head.", "ibot_head."]
            self.gram_teacher.load_state_dict(
                {
                    k: v
                    for k, v in self.model_ema.state_dict().items()
                    if not any(k.startswith(prefix) for prefix in skip_load_prefixes)
                }
            )
            self.gram_teacher.requires_grad_(False)
            self.gram_teacher.eval()
            self.gram_teacher_initialized = True


def _sample_crops(sample, image_key: str) -> Dict[str, Any]:
    """Return a sample's crops as ``{"global_crops": ..., "local_crops": ...}``.

    Accepts a flat dict from ``DINOTransform`` or a legacy ``(crops, label)``
    tuple.
    """
    if isinstance(sample, dict):
        names = ("global_crops", "local_crops", "gram_teacher_crops")
        return {
            n: sample[f"{image_key}_{n}"]
            for n in names
            if f"{image_key}_{n}" in sample
        }
    return sample[0]


def collate_data_and_cast(
    samples_list,
    mask_ratio_tuple: Tuple[float, float],
    mask_probability: float,
    dtype: torch.dtype,
    n_tokens: Optional[int] = None,
    mask_generator: Optional[MaskingGenerator] = None,
    random_circular_shift: bool = False,
    local_batch_size: Optional[int] = None,
    dino_version: int = 3,
    image_key: str = "image",
) -> Dict[str, Any]:
    """
    Collate multi-crop samples, generate random patch masks for iBOT, and cast tensors to precision.

    Parameters
    ----------
    samples_list : list
        Samples loaded by dataset workers. Each is either a flat dict produced
        by ``DINOTransform`` (crops under ``{image_key}_global_crops`` etc.) or
        a legacy ``(crops_dict, label)`` tuple.
    mask_ratio_tuple : tuple of (float, float)
        Range (min, max) of masking ratios.
    mask_probability : float
        Probability of applying masking to a given crop.
    dtype : torch.dtype
        Target floating point tensor data type.
    n_tokens : int, optional
        Total number of patch tokens per crop image.
    mask_generator : MaskingGenerator, optional
        Mask generator instance creating block/patch boolean masks.
    random_circular_shift : bool, default False
        Whether to apply random circular spatial rolls to masks.
    local_batch_size : int, optional
        Batch size per worker rank.
    dino_version : int, default 3
        Target DINO framework version (2 or 3).
    image_key : str, default "image"
        Key ``DINOTransform`` was applied to. Samples that are flat dicts hold
        their crops under ``{image_key}_global_crops``, ``..._local_crops``,
        etc. Legacy ``(crops_dict, label)`` tuple samples are still accepted.

    Returns
    -------
    dict
        Collated batch dictionary containing stacked global/local/Gram crops, masks, and indices.
    """
    samples_list = [_sample_crops(s, image_key) for s in samples_list]
    n_global_crops = len(samples_list[0]["global_crops"])
    n_local_crops = len(samples_list[0]["local_crops"])

    collated_global_crops = torch.stack(
        [s["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
    )  # [n_global_crops, B, ...]
    collated_local_crops = torch.stack(
        [s["local_crops"][i] for i in range(n_local_crops) for s in samples_list]
    )
    if "gram_teacher_crops" in samples_list[0]:
        collated_gram_teacher_crops = torch.stack(
            [
                s["gram_teacher_crops"][i]
                for i in range(n_global_crops)
                for s in samples_list
            ]
        )  # [n_global_crops, B, ...]
    else:
        collated_gram_teacher_crops = None

    if local_batch_size is not None:
        # multi-distillation case, number of masks is different because the number of samples masked
        # is different of the number of samples passed into the teacher initially
        B = n_global_crops * local_batch_size
    else:
        B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        if dino_version == 3:
            mask = torch.BoolTensor(mask_generator(int(N * prob_max)))
        elif dino_version == 2:
            mask = torch.BoolTensor(
                mask_generator(int(N * random.uniform(prob_min, prob_max)))
            )
        if random_circular_shift:  # apply le random circular shift to
            shift_x, shift_y = (
                random.randint(0, mask.shape[0] - 1),
                random.randint(0, mask.shape[1] - 1),
            )
            mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
        masks_list.append(mask)
        upperbound += int(N * prob_max)
    for _ in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (
        (1 / collated_masks.sum(-1).clamp(min=1.0))
        .unsqueeze(-1)
        .expand_as(collated_masks)[collated_masks]
    )

    out = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full(
            (1,), fill_value=mask_indices_list.shape[0], dtype=torch.long
        ),
    }
    if collated_gram_teacher_crops is not None:
        out["collated_gram_teacher_crops"] = collated_gram_teacher_crops.to(dtype)
    return out


# ------------ Auxiliary functions for learning rate schedules and parameter groups ------------ #
def apply_optim_scheduler(
    optimizer: torch.optim.Optimizer, lr: float, wd: float, last_layer_lr: float
) -> None:
    """
    Apply current step learning rate and weight decay values across optimizer parameter groups.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        PyTorch optimizer.
    lr : float
        Base learning rate for current step.
    wd : float
        Weight decay for current step.
    last_layer_lr : float
        Learning rate applied to the last prediction projection layer.
    """
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        if is_last_layer:
            param_group["lr"] = last_layer_lr * lr_multiplier
        else:
            param_group["lr"] = lr * lr_multiplier


def linear_warmup_cosine_decay(
    start: float,
    peak: float,
    end: float,
    warmup_iterations: int,
    total_iterations: int,
    cosine_iterations: int | None = None,
) -> np.ndarray:
    """
    Create a schedule with linear warmup, cosine decay, and an optional constant end part.

    Parameters
    ----------
    start : float
        Initial schedule value.
    peak : float
        Peak value achieved at the end of linear warmup.
    end : float
        Final value after cosine decay.
    warmup_iterations : int
        Number of steps for linear warmup.
    total_iterations : int
        Total schedule step count.
    cosine_iterations : int, optional
        Number of steps for cosine decay (defaults to total - warmup).

    Returns
    -------
    np.ndarray
        Array containing step-wise schedule values.
    """
    linear = np.linspace(start, peak, warmup_iterations, endpoint=False)
    if cosine_iterations is None:
        cosine_iterations = total_iterations - warmup_iterations
    cosine = np.cos(np.linspace(0, np.pi, cosine_iterations))
    cosine = (cosine + 1) / 2
    cosine = (peak - end) * cosine + end
    remaining_iterations = total_iterations - cosine_iterations - warmup_iterations
    assert remaining_iterations >= 0
    constant = np.full((remaining_iterations,), fill_value=end)
    return np.concatenate([linear, cosine, constant])


def get_vit_lr_decay_rate(
    name: str,
    lr_decay_rate: float = 1.0,
    num_layers: int = 12,
    force_is_backbone: bool = False,
    chunked_blocks: bool = False,
) -> float:
    """
    Calculate layer-wise learning rate decay multiplier for Vision Transformer parameters.

    Parameters
    ----------
    name : str
        Parameter name string.
    lr_decay_rate : float, default 1.0
        Base decay rate per layer.
    num_layers : int, default 12
        Total number of transformer blocks.
    force_is_backbone : bool, default False
        Whether to treat parameter as belonging to backbone.
    chunked_blocks : bool, default False
        Whether blocks are organized into chunked modules.

    Returns
    -------
    float
        Learning rate multiplier for the specified parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone") or force_is_backbone:
        if (
            ".pos_embed" in name
            or ".patch_embed" in name
            or ".mask_token" in name
            or ".cls_token" in name
            or ".storage_tokens" in name
        ):
            layer_id = 0
        elif force_is_backbone and (
            "pos_embed" in name
            or "patch_embed" in name
            or "mask_token" in name
            or "cls_token" in name
            or "storage_tokens" in name
        ):
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
        elif chunked_blocks and "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[2]) + 1
        elif "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[1]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_params_groups_with_decay(
    model: nn.Module,
    lr_decay_rate: float = 1.0,
    patch_embed_lr_mult: float = 1.0,
    dino_head_wd_multiplier: float = 1.0,
) -> list:
    """
    Build parameter group dictionaries with layer-wise LR decay and weight decay filtering.

    Parameters
    ----------
    model : nn.Module
        Neural network module.
    lr_decay_rate : float, default 1.0
        Layer-wise learning rate decay rate.
    patch_embed_lr_mult : float, default 1.0
        Learning rate scaling factor for patch embedding layer.
    dino_head_wd_multiplier : float, default 1.0
        Weight decay multiplier for DINO projection head.

    Returns
    -------
    list of dict
        List of parameter group dictionaries.
    """
    chunked_blocks = False
    if hasattr(model, "n_blocks"):
        # logger.info("chunked fsdp")
        n_blocks = model.n_blocks
        chunked_blocks = model.chunked_blocks
    elif hasattr(model, "blocks"):
        # logger.info("first code branch")
        n_blocks = len(model.blocks)
    elif hasattr(model, "backbone"):
        # logger.info("second code branch")
        n_blocks = len(model.backbone.blocks)
    else:
        # logger.info("else code branch")
        n_blocks = 0
    all_param_groups = []

    for name, param in model.named_parameters():
        name = remove_fsdp_compile_names(name)
        if not param.requires_grad:
            continue
        decay_rate = get_vit_lr_decay_rate(
            name,
            lr_decay_rate,
            num_layers=n_blocks,
            force_is_backbone=n_blocks > 0,
            chunked_blocks=chunked_blocks,
        )
        d = {
            "name": name,
            "params": param,
            "is_last_layer": False,
            "lr_multiplier": decay_rate,
            "wd_multiplier": 1.0,
        }

        if "dino_head" in name:
            d["wd_multiplier"] = dino_head_wd_multiplier

        if "last_layer" in name:
            d["is_last_layer"] = True

        # No weight-decay on biases, norm parameters, layer scale gamma, learned tokens and embeddings
        if (
            name.endswith("bias")
            or "norm" in name
            or "gamma" in name
            or "fourier_w" in name
        ):
            d["wd_multiplier"] = 0.0

        if "patch_embed" in name:
            d["lr_multiplier"] *= patch_embed_lr_mult

        all_param_groups.append(d)
        # logger.info(f"{name}: lr_multiplier: {d['lr_multiplier']}, wd_multiplier: {d['wd_multiplier']}")

    return all_param_groups


def fuse_params_groups(
    all_params_groups: list,
    keys: Tuple[str, ...] = ("lr_multiplier", "wd_multiplier", "is_last_layer"),
):
    """
    Merge parameter group dictionaries sharing identical hyperparameters for multi-tensor optimization.

    Parameters
    ----------
    all_params_groups : list of dict
        Input list of individual parameter group dictionaries.
    keys : tuple of str, default ("lr_multiplier", "wd_multiplier", "is_last_layer")
        Attribute keys to match for group fusion.

    Returns
    -------
    dict_values
        Fused parameter groups.
    """
    fused_params_groups = defaultdict(lambda: {"params": []})
    for d in all_params_groups:
        identifier = ""
        for k in keys:
            identifier += k + str(d[k]) + "_"

        for k in keys:
            fused_params_groups[identifier][k] = d[k]
        fused_params_groups[identifier]["params"].append(d["params"])

    return fused_params_groups.values()


def get_params_groups_with_decay_fsdp(
    model: nn.Module,
    lr_decay_rate: float = 1.0,
    patch_embed_lr_mult: float = 1.0,
    dino_head_wd_multiplier: float = 1.0,
) -> list:
    """
    Construct optimizer parameter groups with layer-wise decay for FSDP-wrapped modules.

    Parameters
    ----------
    model : nn.Module
        FSDP-wrapped module.
    lr_decay_rate : float, default 1.0
        Layer-wise learning rate decay factor.
    patch_embed_lr_mult : float, default 1.0
        Patch embedding learning rate multiplier.
    dino_head_wd_multiplier : float, default 1.0
        DINO head weight decay multiplier.

    Returns
    -------
    list of dict
        Parameter groups with decay metadata.
    """
    if hasattr(model, "module"):  # SimpleFSDP
        is_backbone = hasattr(model.module, "blocks")
        n_blocks = len(model.module.blocks) if is_backbone else 0
    else:  # FSDP2
        is_backbone = hasattr(model, "blocks")
        n_blocks = len(model.blocks) if is_backbone else 0

    all_param_groups = []

    for name, param in model.named_parameters():
        name = remove_fsdp_compile_names(name)
        if not param.requires_grad:
            continue
        decay_rate = get_vit_lr_decay_rate(
            name,
            lr_decay_rate,
            num_layers=n_blocks,
            force_is_backbone=n_blocks > 0,
            chunked_blocks=False,
        )
        d = {
            "name": name,
            "params": param,
            "is_last_layer": False,
            "lr_multiplier": decay_rate,
            "wd_multiplier": 1.0,
        }

        if "dino_head" in name and dino_head_wd_multiplier is not None:
            d["wd_multiplier"] = dino_head_wd_multiplier

        if "last_layer" in name:
            d["is_last_layer"] = True

        # No weight-decay on biases, norm parameters, layer scale gamma, learned tokens and embeddings
        if (
            name.endswith("bias")
            or "norm" in name
            or "gamma" in name
            or "fourier_w" in name
        ):
            d["wd_multiplier"] = 0.0

        if "patch_embed" in name:
            d["lr_multiplier"] *= patch_embed_lr_mult

        all_param_groups.append(d)
        # print(f"{name}: lr_multiplier: {d['lr_multiplier']}, wd_multiplier: {d['wd_multiplier']}")

    return all_param_groups


def remove_fsdp_compile_names(name: str) -> str:
    """
    Strip wrapper prefixes introduced by FSDP, activation checkpointing, and torch.compile.

    Parameters
    ----------
    name : str
        Parameter name string.

    Returns
    -------
    str
        Cleaned parameter name.
    """
    name = name.replace("_fsdp_wrapped_module.", "")  # Added by FSDP
    name = name.replace(
        "_checkpoint_wrapped_module.", ""
    )  # Added by activation checkpointing for xFSDP
    name = name.replace("parametrizations.", "")  # Added by xFSDP
    name = name.removesuffix(".original")  # Added by xFSDP
    name = name.replace("module.", "")  # Added by xFSDP
    name = name.replace("_orig_mod.", "")  # Added by torch.compile
    return name


class CosineScheduler:
    """
    Cosine learning rate and hyperparameter decay schedule with optional warmup and truncation.

    Parameters
    ----------
    base_value : float
        Initial value after warmup.
    final_value : float
        Target value at the end of the cosine schedule.
    total_iters : int
        Total number of iterations in the schedule.
    warmup_iters : int, default 0
        Number of linear warmup iterations.
    start_warmup_value : float, default 0.0
        Starting value at iteration 0 before warmup.
    freeze_iters : int, default 0
        Number of iterations to keep the value frozen at 0.
    trunc_extra : float, default 0.0
        Truncation factor for modifying cosine decay profile.
    """

    def __init__(
        self,
        base_value: float,
        final_value: float,
        total_iters: int,
        warmup_iters: int = 0,
        start_warmup_value: float = 0,
        freeze_iters: int = 0,
        trunc_extra: float = 0.0,
    ):
        super().__init__()
        self.final_value = np.float64(final_value)
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        if trunc_extra == 0.0:
            iters = np.arange(total_iters - warmup_iters - freeze_iters)
            schedule = final_value + 0.5 * (base_value - final_value) * (
                1 + np.cos(np.pi * iters / len(iters))
            )
        else:
            cosine_steps = total_iters - warmup_iters - freeze_iters
            iters = np.linspace(0, np.pi, int((1 + trunc_extra) * cosine_steps))[
                :cosine_steps
            ]
            schedule = np.cos(iters)
            schedule = (schedule + 1) / 2
            schedule = (schedule - schedule[-1]) / (1 - schedule[-1])
            schedule = schedule * (base_value - final_value) + final_value

        self.schedule = np.concatenate(
            (freeze_schedule, warmup_schedule, schedule), dtype=np.float64
        )

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it: int) -> float:
        """
        Retrieve scheduled value at iteration `it`.

        Parameters
        ----------
        it : int
            Iteration step index.

        Returns
        -------
        float
            Scheduled value at the given step.
        """
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]
