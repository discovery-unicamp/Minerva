import torch

from minerva.models.nets.deeplabv3 import DeepLabV3, DeepLabV3Backbone


def test_deeplabv3_model():

    # Test the class instantiation
    model = DeepLabV3()
    assert model is not None

    # Test the forward method
    input_shape = (2, 3, 701, 255)
    expected_output_size = torch.Size([2, 6, 701, 255])
    x = torch.rand(*input_shape)
    output = model(x)
    assert (
        output.shape == expected_output_size
    ), f"Expected output shape {input_shape}, but got {output.shape}"

    # Test the _loss_func method
    label_shape = (2, 1, 701, 255)
    mask = torch.rand(*label_shape)
    loss = model._loss_func(x, mask)
    assert loss is not None
    # TODO: assert the loss result

    # Test the configure_optimizers method
    optimizer = model.configure_optimizers()
    assert optimizer is not None


def test_deeplabv3_backbone():

    # Test the class instantiation
    backbone = DeepLabV3Backbone()
    assert backbone is not None

    # Test the forward method
    input_shape = (2, 3, 701, 255)
    expected_output_size = torch.Size([2, 2048, 88, 32])
    x = torch.rand(*input_shape)
    output = backbone(x)
    assert (
        output.shape == expected_output_size
    ), f"Expected output shape {input_shape}, but got {output.shape}"

    # Test the freeze_weights method
    backbone.freeze_weights()

    # Test the unfreeze_weights method
    backbone.unfreeze_weights()


def test_deeplabv3_save_restore():

    # Test the class instantiation
    model = DeepLabV3()
    assert model is not None
