import pytest
import torch
import torch.nn as nn
import lightning as L
from functools import partial
from minerva.losses.dice import MultiClassDiceCELoss
from minerva.models.nets.image.unetplusplus_resnet50 import (
    DeepLabV3ResNet50Backbone,
    ConvBlock,
    UNetPlusPlusDeepLabV3,
    LitUNetPlusPlusDeepLabV3,
)
from minerva.data.data_module_tools import RandomDataModule

# Parameterize test configurations
model_configs = [
    {"deep_supervision": True, "pretrained": False, "num_classes": 6},
    {"deep_supervision": False, "pretrained": False, "num_classes": 6},
    {"deep_supervision": True, "pretrained": False, "num_classes": 2},
]


@pytest.mark.parametrize("config", model_configs)
def test_unetplusplus_deeplabv3_fit(config):
    """Test the full training pipeline for LitUNetPlusPlusDeepLabV3.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys: deep_supervision, pretrained, num_classes.
    """
    input_shape = (256, 256)  # Must be divisible by 8 due to backbone downsampling
    model = LitUNetPlusPlusDeepLabV3(
        in_channels=3,
        num_classes=config["num_classes"],
        deep_supervision=config["deep_supervision"],
        lr=1e-3,
        pretrained=config["pretrained"],
    )

    data_module = RandomDataModule(
        data_shape=(3, *input_shape),
        label_shape=(1, *input_shape),
        num_classes=config["num_classes"],
        num_train_samples=2,
        batch_size=2,
    )

    trainer = L.Trainer(fast_dev_run=True, devices=1, accelerator="cpu", max_epochs=1)
    trainer.fit(model, data_module)


@pytest.mark.parametrize("pretrained", [True, False])
def test_deeplabv3_resnet50_backbone_forward(pretrained):
    """Test the forward pass of DeepLabV3ResNet50Backbone.

    Parameters
    ----------
    pretrained : bool
        Whether to use pretrained weights for the backbone.
    """
    backbone = DeepLabV3ResNet50Backbone(pretrained=pretrained)
    input_shape = (256, 256)
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, *input_shape)

    c1, c2, c3, c4, c5 = backbone(input_tensor)

    # Check output shapes
    assert c1.shape == (batch_size, 64, input_shape[0] // 4, input_shape[1] // 4)
    assert c2.shape == (batch_size, 256, input_shape[0] // 4, input_shape[1] // 4)
    assert c3.shape == (batch_size, 512, input_shape[0] // 8, input_shape[1] // 8)
    assert c4.shape == (batch_size, 1024, input_shape[0] // 8, input_shape[1] // 8)
    assert c5.shape == (batch_size, 2048, input_shape[0] // 8, input_shape[1] // 8)


@pytest.mark.parametrize("in_channels,out_channels", [(64, 128), (128, 256)])
def test_conv_block_forward(in_channels, out_channels):
    """Test the forward pass of ConvBlock.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    conv_block = ConvBlock(in_channels=in_channels, out_channels=out_channels)
    batch_size = 2
    input_shape = (64, 64)
    input_tensor = torch.randn(batch_size, in_channels, *input_shape)

    output = conv_block(input_tensor)

    assert output.shape == (batch_size, out_channels, *input_shape)
    assert (output >= 0).all()  # ReLU should ensure non-negative outputs


@pytest.mark.parametrize("weight_ce,weight_dice", [(1.0, 1.0), (0.5, 1.5), (1.0, 0.0)])
def test_multiclass_dice_ce_loss(weight_ce, weight_dice):
    """Test the MultiClassDiceCELoss computation.

    Parameters
    ----------
    weight_ce : float
        Weight for Cross-Entropy loss component.
    weight_dice : float
        Weight for Dice loss component.
    """
    loss_fn = MultiClassDiceCELoss(weight_ce=weight_ce, weight_dice=weight_dice)
    batch_size = 2
    num_classes = 6
    input_shape = (64, 64)

    # Test single output
    predictions = torch.randn(batch_size, num_classes, *input_shape)
    targets = torch.randint(0, num_classes, (batch_size, *input_shape))
    loss = loss_fn(predictions, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)
    assert loss.item() >= 0

    # Test deep supervision outputs
    predictions_list = [
        torch.randn(batch_size, num_classes, *input_shape) for _ in range(3)
    ]
    loss = loss_fn(predictions_list, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)
    assert loss.item() >= 0


@pytest.mark.parametrize("config", model_configs)
def test_unetplusplus_deeplabv3_forward(config):
    """Test the forward pass of UNetPlusPlusDeepLabV3.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys: deep_supervision, pretrained, num_classes.
    """
    model = UNetPlusPlusDeepLabV3(
        in_channels=3,
        num_classes=config["num_classes"],
        deep_supervision=config["deep_supervision"],
        pretrained=config["pretrained"],
    )
    batch_size = 2
    input_shape = (256, 256)
    input_tensor = torch.randn(batch_size, 3, *input_shape)

    output = model(input_tensor)

    if config["deep_supervision"]:
        assert isinstance(output, list)
        assert len(output) == 3
        for out in output:
            assert out.shape == (batch_size, config["num_classes"], *input_shape)
    else:
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, config["num_classes"], *input_shape)


@pytest.mark.parametrize("input_shape", [(128, 128), (320, 320)])
def test_model_different_input_sizes(input_shape):
    """Test UNetPlusPlusDeepLabV3 with different input sizes.

    Parameters
    ----------
    input_shape : tuple
        Height and width of the input tensor.
    """
    model = UNetPlusPlusDeepLabV3(
        in_channels=3, num_classes=6, deep_supervision=False, pretrained=False
    )
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, *input_shape)

    output = model(input_tensor)

    assert output.shape == (batch_size, 6, *input_shape)
