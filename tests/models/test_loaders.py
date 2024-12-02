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
from minerva.data.data_module_tools import RandomDataModule
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
        filter_keys=["backbone"],
        strict=True,
    )
    head = Head(input_dim=64, output_dim=6)
    new_model = ComposableModel(backbone, head, flatten=True)
    new_model.eval()

    # Check if backbone weights are the same
    for p1, p2 in zip(
        model.backbone.parameters(), new_model.backbone.parameters()
    ):
        assert torch.allclose(p1, p2), "Weights are different!"


    # Heads should have different weights
    # for p1, p2 in zip(model.head.parameters(), new_model.head.parameters()):
    #     assert not torch.allclose(p1, p2), "Weights are the same!"


# class Backbone_MLP_3L(torch.nn.Module):
#     def __init__(self, input_dim=64, output_dim=8):
#         super().__init__()
#         self.fc1 = torch.nn.Linear(input_dim, 32, bias=False)
#         self.fc2 = torch.nn.Linear(32, 16, bias=False)
#         self.fc3 = torch.nn.Linear(16, output_dim, bias=False)
#         self.relu = torch.nn.ReLU()

#     def forward(self, x):
#         x = self.fc1(x)
#         print("fc1:", x.shape)
#         x = self.relu(x)
#         print("relu:", x.shape)
#         x = self.fc2(x)
#         print("fc2:", x.shape)
#         x = self.relu(x)
#         print("relu:", x.shape)
#         x = self.fc3(x)
#         print("fc3:", x.shape)
#         return x


# class Backbone_MLP_3L_Sequential(torch.nn.Module):
#     def __init__(self, input_dim=64, output_dim=8):
#         super().__init__()
#         self.fc = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, 32),
#             torch.nn.ReLU(),
#             torch.nn.Linear(32, 16),
#             torch.nn.ReLU(),
#             torch.nn.Linear(16, output_dim),
#         )

#     def forward(self, x):
#         x = self.fc(x)
#         return x


# class Head_1L(torch.nn.Module):
#     def __init__(self, input_dim=8, output_dim=2):
#         super().__init__()
#         self.fc1 = torch.nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         x = self.fc1(x)
#         return x


# class Head_2L(torch.nn.Module):
#     def __init__(self, input_dim=8, output_dim=2):
#         super().__init__()
#         self.fc1 = torch.nn.Linear(input_dim, 4)
#         self.fc2 = torch.nn.Linear(4, output_dim)
#         self.relu = torch.nn.ReLU()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x


# class FusedGencGar(torch.nn.Module):
#     def __init__(self, g_enc, g_ar):
#         super().__init__()
#         self.g_enc = g_enc
#         self.g_ar = g_ar

#     def forward(self, x):
#         x = self.g_enc(x)
#         return self.g_ar(x, None)


# def rnn_to_mlp(x: tuple):
#     # unpack
#     x, _ = x
#     x = x[:, -1, :]
#     return x


# g_enc = CNN()
# g_ar = HARCPCAutoregressive()
# prediction_head = PredictionNetwork()

# # Test the class instantiation
# model = CPC(g_enc=g_enc, g_ar=g_ar, prediction_head=prediction_head)

# # Generate a random input tensor (B, C, T) and the expected output shape
# input_shape = (1, 6, 60)
# expected_output_shape_z = (1, 60, 128)

# # Create random input data
# x = torch.rand(*input_shape)

# # Test the forward method
# # Output forward method: return z,y
# output = model.forward(x)
# if isinstance(output, tuple):
#     output = output[0]

# print(output.shape)

# dm = RandomDataModule(
#     data_shape=(6, 60), num_train_samples=1, batch_size=1, num_classes=6
# )
# trainer = L.Trainer(
#     fast_dev_run=True,
#     enable_model_summary=True,
#     accelerator="cpu",
#     enable_progress_bar=False,
# )
# trainer.fit(model, dm)


# print(model)

# with tempfile.TemporaryDirectory() as temp_dir:
#     temp_path = f"{temp_dir}/model.ckpt"
#     torch.save(model.state_dict(), temp_path)
#     print(f"** Model saved to {temp_path}!")

#     pretrained = FromPretrained(
#         model,
#         temp_path,
#         extractor=IntermediateLayerGetter(
#             layers=[
#                 ("g_enc", "g_enc"),
#                 ("g_ar", "g_ar"),
#             ]
#         ),
#     )

#     r = pretrained(x)
#     print(r[0].shape, r[1].shape)

#     new_model = SimpleSupervisedModel(
#         backbone=pretrained,
#         adapter=rnn_to_mlp,
#         flatten=False,
#         fc=Head_1L(256, 6),
#         loss_fn=torch.nn.CrossEntropyLoss(),
#     )

#     new_model.eval()
#     print(new_model)

#     r = new_model(x)
#     print(f"Result: {r.shape}")

#     print(
#         f"Modules: {[k for k, v in new_model.named_modules(remove_duplicate=False)]}"
#     )

# print("-" * 80)

# with tempfile.TemporaryDirectory() as temp_dir:
#     temp_path = f"{temp_dir}/model.ckpt"
#     torch.save(model.state_dict(), temp_path)
#     print(f"** Model saved to {temp_path}!")

#     pretrained = FromPretrained(
#         FusedGencGar(g_enc, g_ar),
#         temp_path,
#         extractor=None,
#         filter_keys=["g_enc", "g_ar"],
#         strict=True,
#     )

#     r = pretrained(x)
#     print(r[0].shape, r[1].shape)

#     new_model = SimpleSupervisedModel(
#         backbone=pretrained,
#         adapter=rnn_to_mlp,
#         flatten=False,
#         fc=Head_1L(256, 6),
#         loss_fn=torch.nn.CrossEntropyLoss(),
#     )

#     new_model.eval()
#     print(new_model)

#     r = new_model(x)
#     print(f"Result: {r.shape}")

#     print(
#         f"Modules: {[k for k, v in new_model.named_modules(remove_duplicate=False)]}"
#     )


# print("-" * 80)

# # new_model = SimpleSupervisedModel(
# #     backbone=
# #     fc=Head_1L(128, 2),
# # )


# data = torch.rand(1, 64)

# backbone_1 = Backbone_MLP_3L()
# backbone_1.eval()
# print(backbone_1)
# result = backbone_1(data)
# result_sum = result.sum().item()
# print("SUM=", result_sum)

# getter = IntermediateLayerGetter(
#     layers=[
#         ("fc1", "fc1"),
#         ("relu", "relu1"),
#         ("fc2", "fc2"),
#         ("relu", "relu2"),
#         ("fc3", "fc3"),
#         # ("relu", "relu3"),
#     ]
# )(backbone_1)
# getter.eval()
# print(getter)
# result_2 = getter(data)
# result_2_sum = result_2.sum().item()
# print("SUM=", result_2_sum)

# print(
#     f"Result1: {result_sum}, Result2: {result_2_sum}. Equal? {result_sum == result_2_sum}"
# )
