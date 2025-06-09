import torch
import tempfile
import pytest
from minerva.models.nets.image.deeplabv3 import DeepLabV3, DeepLabV3Backbone


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
    ), f"Expected output shape {expected_output_size}, but got {output.shape}"
    # Test the *loss*func method
    label_shape = (2, 1, 701, 255)
    mask = torch.rand(*label_shape)
    loss = model._loss_func(output, mask)
    assert loss is not None
    # TODO: assert the loss result
    # Test the configure_optimizers method (inherited from SimpleSupervisedModel)
    optimizer = model.configure_optimizers()
    assert optimizer is not None


def test_deeplabv3_model_with_custom_num_classes():
    """Test DeepLabV3 with custom number of classes"""
    num_classes = 10
    model = DeepLabV3(num_classes=num_classes, pretrained=False)
    assert model is not None

    # Test forward pass
    input_shape = (2, 3, 224, 224)
    expected_output_size = torch.Size([2, num_classes, 224, 224])
    x = torch.rand(*input_shape)
    output = model(x)
    assert output.shape == expected_output_size


def test_deeplabv3_model_with_local_weights():
    """Test DeepLabV3 with local pretrained weights"""
    # Create a temporary weight file
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=True) as tmp_file:
        # Create a mock state dict similar to ResNet50
        mock_state_dict = {
            "conv1.weight": torch.randn(64, 3, 7, 7),
            "bn1.weight": torch.randn(64),
            "bn1.bias": torch.randn(64),
            "layer1.0.conv1.weight": torch.randn(64, 64, 1, 1),
            "layer1.0.bn1.weight": torch.randn(64),
            "layer1.0.bn1.bias": torch.randn(64),
            # Add fc layer that should be filtered out
            "fc.weight": torch.randn(1000, 2048),
            "fc.bias": torch.randn(1000),
        }
        torch.save(mock_state_dict, tmp_file.name)

        # Test with local weights
        model = DeepLabV3(pretrained=True, weights_path=tmp_file.name)
        assert model is not None

        # Test forward pass
        input_shape = (2, 3, 224, 224)
        expected_output_size = torch.Size([2, 6, 224, 224])
        x = torch.rand(*input_shape)
        output = model(x)
        assert output.shape == expected_output_size


def test_deeplabv3_model_with_invalid_weights_path():
    """Test DeepLabV3 with invalid weights path (should raise FileNotFoundError)"""
    with pytest.raises(FileNotFoundError):
        DeepLabV3(pretrained=True, weights_path="/invalid/path/weights.pth")


def test_deeplabv3_model_with_output_shape():
    """Test DeepLabV3 with custom output shape (B2 specific)"""
    custom_output_shape = (512, 512)
    model = DeepLabV3(pretrained=False, output_shape=custom_output_shape)
    assert model is not None

    # Test forward pass
    input_shape = (2, 3, 224, 224)
    expected_output_size = torch.Size([2, 6, 512, 512])
    x = torch.rand(*input_shape)
    output = model(x)
    assert output.shape == expected_output_size


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
    ), f"Expected output shape {expected_output_size}, but got {output.shape}"


def test_deeplabv3_backbone_with_pretrained_false():
    """Test DeepLabV3Backbone with explicit pretrained=False"""
    backbone = DeepLabV3Backbone(pretrained=False)
    assert backbone is not None

    # Test forward pass
    input_shape = (2, 3, 224, 224)
    expected_output_size = torch.Size([2, 2048, 28, 28])
    x = torch.rand(*input_shape)
    output = backbone(x)
    assert output.shape == expected_output_size


def test_deeplabv3_backbone_with_local_weights():
    """Test DeepLabV3Backbone with local pretrained weights"""
    # Create a temporary weight file with nested state_dict structure
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=True) as tmp_file:
        # Create a mock state dict with nested structure (common in checkpoints)
        mock_state_dict = {
            "state_dict": {
                "conv1.weight": torch.randn(64, 3, 7, 7),
                "bn1.weight": torch.randn(64),
                "bn1.bias": torch.randn(64),
                "layer1.0.conv1.weight": torch.randn(64, 64, 1, 1),
                "layer1.0.bn1.weight": torch.randn(64),
                "layer1.0.bn1.bias": torch.randn(64),
                # Add fc layer that should be filtered out
                "fc.weight": torch.randn(1000, 2048),
                "fc.bias": torch.randn(1000),
            }
        }
        torch.save(mock_state_dict, tmp_file.name)

        # Test with local weights
        backbone = DeepLabV3Backbone(pretrained=True, weights_path=tmp_file.name)
        assert backbone is not None

        # Test forward pass
        input_shape = (2, 3, 224, 224)
        expected_output_size = torch.Size([2, 2048, 28, 28])
        x = torch.rand(*input_shape)
        output = backbone(x)
        assert output.shape == expected_output_size


def test_deeplabv3_backbone_with_invalid_weights_path():
    """Test DeepLabV3Backbone with invalid weights path (should raise FileNotFoundError)"""
    with pytest.raises(FileNotFoundError):
        DeepLabV3Backbone(pretrained=True, weights_path="/invalid/path/weights.pth")


def test_deeplabv3_backbone_freeze_unfreeze():
    """Test freeze and unfreeze functionality"""
    backbone = DeepLabV3Backbone(pretrained=False)

    # Test initial state (should be trainable)
    assert all(param.requires_grad for param in backbone.parameters())

    # Test freeze
    backbone.freeze_weights()
    assert all(not param.requires_grad for param in backbone.parameters())

    # Test unfreeze
    backbone.unfreeze_weights()
    assert all(param.requires_grad for param in backbone.parameters())


def test_deeplabv3_save_restore():
    # Test the class instantiation
    model = DeepLabV3()
    assert model is not None
