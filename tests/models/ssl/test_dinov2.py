import pytest
import torch

from minerva.models.ssl.dinov2 import (
    DinoVisionTransformer,
    SETR_PUP,
    SETR_MLA,
    DPT,
    NestedTensorBlock,
    MemEffAttention,
    DinoV2,
)
from functools import partial
from minerva.data.data_module_tools import RandomDataModule
import lightning as L


heads_and_middle = [(SETR_PUP, False), (SETR_MLA, True), (DPT, True)]


@pytest.mark.parametrize("head_cls,middle", heads_and_middle)
def test_dinov2_fit(head_cls, middle):
    input_shape = (14**2, 14**2)  # Must be divisible by 14
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

    head = head_cls(embedding_dim=384, num_classes=6)

    model = DinoV2(
        backbone=backbone,
        head=head,
        loss_fn=torch.nn.CrossEntropyLoss(),
        output_shape=input_shape,
        middle=middle,
    )

    data_module = RandomDataModule(
        data_shape=(3, *input_shape),
        label_shape=(1, *input_shape),
        num_classes=6,
        num_train_samples=2,
        batch_size=2,
    )

    trainer = L.Trainer(
        fast_dev_run=True, devices=1, accelerator="cpu", max_epochs=1
    )
    trainer.fit(model, data_module)
