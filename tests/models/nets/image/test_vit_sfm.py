import pytest
import torch

from minerva.models.nets.image.vit import (
    mae_vit_base_patch16,
    mae_vit_base_patch16D4d256,
    mae_vit_huge_patch14,
    mae_vit_large_patch16,
    mae_vit_large_patch16D4d256,
    mae_vit_small_patch16,
    SFM_BasePatch16_Downstream
)

test_models = [
    (mae_vit_small_patch16, 224),
    (mae_vit_base_patch16, 224),
    (mae_vit_large_patch16, 224),
    (mae_vit_huge_patch14, 224),
    (mae_vit_large_patch16D4d256, 224),
    (mae_vit_base_patch16D4d256, 224),
]

from minerva.data.data_module_tools import RandomDataModule
import lightning as L

@pytest.mark.parametrize("model_cls,img_size", test_models)
def test_sfm_pretrain_forward(model_cls, img_size):
    # Test the class instantiation
    model = model_cls(img_size=img_size, in_chans=1, norm_pix_loss=False)
    assert model is not None

    # Generate a random input tensor (B, C, H, W) and the random mask of the
    # same shape
    input_shape = (1, 1, img_size, img_size)
    x = torch.rand(*input_shape)
    mask = torch.rand(*input_shape)

    # Test the forward method
    loss, pred, mask = model(x)

    assert loss is not None
    assert loss >= 0, f"Expected non-negative loss, but got {loss}"
    assert pred is not None
    assert mask is not None


def test_sfm_vit_patch16_downstream():
    model = SFM_BasePatch16_Downstream(
        img_size=(512, 512),
        num_classes=6,
        in_chans=1
    )
    
    data_module = RandomDataModule(
        data_shape=(1, 512, 512),
        label_shape=(1, 512, 512),
        num_classes=6,
        num_train_samples=2,
        batch_size=2,
    )

    trainer = L.Trainer(accelerator="cpu", max_epochs=1, devices=1, fast_dev_run=True)
    trainer.fit(model, data_module)