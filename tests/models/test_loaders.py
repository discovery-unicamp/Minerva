import torch
import lightning as L
import pytest

from minerva.models.nets.base import SimpleSupervisedModel
from minerva.models.loaders import (
    ExtractedModel,
    IntermediateLayerGetter,
    FromModel,
    FromPretrained,
)

from minerva.models.nets.cpc_networks import (
    CNN,
    HARCPCAutoregressive,
    PredictionNetwork,
    HARPredictionHead,
)
from minerva.models.ssl.cpc import CPC
from minerva.utils.data import RandomDataModule
from minerva.models.nets.base import SimpleSupervisedModel
import lightning as L
import tempfile


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 32, bias=False)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, 16, bias=False)
        self.fc3 = torch.nn.Linear(16, 8, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class ComposableModel(L.LightningModule):
    def __init__(self, backbone, head, flatten: bool = False):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.flatten = flatten

    def forward(self, x):
        x = self.backbone(x)
        if self.flatten:
            x = x.flatten(start_dim=1)
        x = self.head(x)
        return x


class LSTMBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=128, hidden_size=256, batch_first=True
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class Conv1DBackbone(torch.nn.Module):
    def __init__(self, input_dim=64, output_dim=128):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            input_dim, output_dim, kernel_size=3, padding="same", bias=False
        )

    def forward(self, x):
        return self.conv(x)


class MLPBackbone(torch.nn.Module):
    def __init__(self, input_dim=64, output_dim=128):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.fc(x)


class MLPBackbone2Layers(torch.nn.Module):
    def __init__(self, input_dim=64, output_dim=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64, bias=False)
        self.fc2 = torch.nn.Linear(64, output_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class MLPBackbone3Layers(torch.nn.Module):
    def __init__(self, input_dim=64, output_dim=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 32, bias=False)
        self.fc2 = torch.nn.Linear(32, 16, bias=False)
        self.fc3 = torch.nn.Linear(16, output_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


class Head(torch.nn.Module):
    def __init__(self, input_dim=128, output_dim=6):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.fc(x)


################################################################################


def mlp_3layer_model():
    backbone = MLPBackbone3Layers(input_dim=128, output_dim=64)
    head = Head(input_dim=64, output_dim=2)
    model = ComposableModel(backbone, head, flatten=True)
    return model


def test_load_from_pretrained(tmp_path):
    # Create a simple model
    checkpoint_file = tmp_path / "model.ckpt"

    model = mlp_3layer_model()
    model.eval()

    # Save the model
    torch.save(model.state_dict(), checkpoint_file)

    # Load the model from the checkpoint file
    new_model = mlp_3layer_model()
    new_model = FromPretrained(model, checkpoint_file)
    new_model.eval()

    # Check if the weights are the same
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2), "Weights are different!"


def test_load_from_pretrained_backbone_with_extractor(tmp_path):
    # Create a simple model
    checkpoint_file = tmp_path / "model.ckpt"

    model = mlp_3layer_model()
    model.eval()

    # Save the model
    torch.save(model.state_dict(), checkpoint_file)

    # *---------------------------------------------------------------*
    backbone = mlp_3layer_model()
    backbone = FromPretrained(
        backbone,
        checkpoint_file,
        extractor=IntermediateLayerGetter(layers=["backbone"]),
        strict=False,
    )
    head = Head(input_dim=64, output_dim=6)
    new_model = ComposableModel(backbone, head, flatten=True)
    new_model.eval()

    # Check if backbone weights are the same
    for p1, p2 in zip(
        model.backbone.parameters(), new_model.backbone.parameters()
    ):
        assert torch.allclose(p1, p2), "Weights are different!"

    # Check if backbone's forward method has same output
    x = torch.rand(1, 128)
    y1 = model.backbone(x)
    y2 = new_model.backbone(x)
    assert torch.allclose(y1, y2), "Output is different!"

    # Check if forwards have the correct shape
    expected_model_shape = (1, 2)
    expected_new_model_shape = (1, 6)

    y1 = model(x)
    y2 = new_model(x)
    assert (
        y1.shape == expected_model_shape
    ), f"Expected {expected_model_shape}, got {y1.shape}"
    assert (
        y2.shape == expected_new_model_shape
    ), f"Expected {expected_new_model_shape}, got {y2.shape}"


    # Save and load the new model
    torch.save(new_model.state_dict(), checkpoint_file)
    
    # Load 
    newest_model = new_model
    newest_model = FromPretrained(
        newest_model,
        checkpoint_file,
        extractor=IntermediateLayerGetter(layers=["backbone"]),
        strict=False,
    )
    
    # Check if backbone weights are the same
    for p1, p2 in zip(
        model.backbone.parameters(), newest_model.backbone.parameters()
    ):
        assert torch.allclose(p1, p2), "Weights are different!"
        
    # Check if backbone's forward method has same output
    x = torch.rand(1, 128)
    y1 = model.backbone(x)
    y2 = newest_model.backbone(x)
    assert torch.allclose(y1, y2), "Output is different!"
    

def test_load_from_pretrained_backbone_without_extractor(tmp_path):
    # Create a simple model
    checkpoint_file = tmp_path / "model.ckpt"

    model = mlp_3layer_model()
    model.eval()

    # Save the model
    torch.save(model.state_dict(), checkpoint_file)

    # *---------------------------------------------------------------*
    backbone = MLPBackbone3Layers(input_dim=128, output_dim=64)
    backbone = FromPretrained(
        backbone,
        checkpoint_file,
        # extractor=IntermediateLayerGetter(layers=["backbone"]),
        filter_keys=["backbone."],
        keys_to_rename={"backbone.": ""},
        strict=True,
        error_on_missing_keys=True
    )
    head = Head(input_dim=64, output_dim=6)
    new_model = ComposableModel(backbone, head, flatten=True)
    new_model.eval()

    # Check if backbone weights are the same
    for p1, p2 in zip(
        model.backbone.parameters(), new_model.backbone.parameters()
    ):
        assert torch.allclose(p1, p2), "Weights are different!"

