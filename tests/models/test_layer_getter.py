import torch
import pytest

from minerva.models.loaders import IntermediateLayerGetter

import lightning as L


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


def test_intermediate_layer_getter_using_list():
    # Create a simple model
    model = SimpleModel()
    model.eval()

    named_modules = dict(model.named_modules(remove_duplicate=False))
    assert len(named_modules) == 5
    assert "fc1" in named_modules
    assert "relu" in named_modules
    assert "fc2" in named_modules
    assert "fc3" in named_modules

    getter = IntermediateLayerGetter(layers=["fc1", "relu"])
    new_model = getter(model)
    new_model.eval()

    # Create a new model with the same architecture
    new_named_modules = dict(new_model.named_modules(remove_duplicate=False))
    new_named_modules.pop("")
    assert len(new_named_modules) == 2, f"Expected 2, got {len(new_named_modules)}"
    assert "fc1" in new_named_modules, f"Expected fc1, got {new_named_modules.keys()}"
    assert "relu" in new_named_modules, f"Expected relu, got {new_named_modules.keys()}"
    assert torch.allclose(named_modules["fc1"].weight, new_named_modules["fc1"].weight)

    # Check if order is correct
    assert list(new_named_modules.keys()) == [
        "fc1",
        "relu",
    ], f"Expected ['fc1', 'relu'], in order"

    # Test with repeated layers
    getter = IntermediateLayerGetter(layers=["fc1", "fc1", "relu", "relu", "fc1"])
    new_model = getter(model)
    new_model.eval()
    new_named_modules = dict(new_model.named_modules(remove_duplicate=False))
    new_named_modules.pop("")
    assert (
        len(dict(new_named_modules)) == 2
    ), f"Expected 2, got {len(new_named_modules)}"
    assert list(new_named_modules.keys()) == [
        "fc1",
        "relu",
    ], f"Expected ['fc1', 'relu'], in order"

    # Get in other order
    getter = IntermediateLayerGetter(layers=["relu", "fc1"])
    new_model = getter(model)
    new_model.eval()
    new_named_modules = dict(new_model.named_modules(remove_duplicate=False))
    new_named_modules.pop("")
    assert (
        len(dict(new_named_modules)) == 2
    ), f"Expected 2, got {len(new_named_modules)}"
    assert list(new_named_modules.keys()) == [
        "relu",
        "fc1",
    ], f"Expected ['relu', 'fc1'], in order"

    # Test with invalid layer
    with pytest.raises(ValueError):
        getter = IntermediateLayerGetter(layers=["fc1", "dummy"])
        new_model = getter(model)


def test_intermediate_layer_getter_using_dict():
    # Create a simple model
    model = SimpleModel()
    model.eval()

    named_modules = dict(model.named_modules(remove_duplicate=False))
    assert len(named_modules) == 5
    assert "fc1" in named_modules
    assert "relu" in named_modules
    assert "fc2" in named_modules
    assert "fc3" in named_modules

    getter = IntermediateLayerGetter(layers={"fc1": "new_fc1", "relu": "new_relu"})
    new_model = getter(model)
    new_model.eval()

    # Create a new model with the same architecture
    new_named_modules = dict(new_model.named_modules(remove_duplicate=False))
    new_named_modules.pop("")
    assert len(new_named_modules) == 2, f"Expected 2, got {len(new_named_modules)}"
    assert (
        "new_fc1" in new_named_modules
    ), f"Expected new_fc1, got {new_named_modules.keys()}"
    assert (
        "new_relu" in new_named_modules
    ), f"Expected new_relu, got {new_named_modules.keys()}"
    assert torch.allclose(
        named_modules["fc1"].weight, new_named_modules["new_fc1"].weight
    )

    getter = IntermediateLayerGetter(
        layers={"fc1": "new_fc1", "relu": "new_relu", "fc1": "new_fc2"}
    )
    new_model = getter(model)
    new_model.eval()

    # Create a new model with the same architecture
    new_named_modules = dict(new_model.named_modules(remove_duplicate=False))
    new_named_modules.pop("")
    assert len(new_named_modules) == 2, f"Expected 2, got {len(new_named_modules)}"
    assert (
        "new_fc2" in new_named_modules
    ), f"Expected new_fc2, got {new_named_modules.keys()}"
    assert (
        "new_relu" in new_named_modules
    ), f"Expected new_relu, got {new_named_modules.keys()}"
    assert torch.allclose(
        named_modules["fc1"].weight, new_named_modules["new_fc2"].weight
    )

    getter = IntermediateLayerGetter(
        layers={"fc1": "new_fc1", "relu": "new_relu", "fc2": "new_fc1"}
    )
    new_model = getter(model)
    new_model.eval()

    # Create a new model with the same architecture
    new_named_modules = dict(new_model.named_modules(remove_duplicate=False))
    new_named_modules.pop("")
    assert len(new_named_modules) == 2, f"Expected 2, got {len(new_named_modules)}"
    assert (
        "new_fc1" in new_named_modules
    ), f"Expected new_fc1, got {new_named_modules.keys()}"
    assert (
        "new_relu" in new_named_modules
    ), f"Expected new_relu, got {new_named_modules.keys()}"
    assert torch.allclose(
        named_modules["fc2"].weight, new_named_modules["new_fc1"].weight
    )

    # Test with invalid layer
    with pytest.raises(ValueError):
        getter = IntermediateLayerGetter(
            layers={"fc1": "new_fc1", "dummy": "new_dummy"}
        )
        new_model = getter(model)


def test_intermediate_layer_getter_using_list_of_tuples():
    # Create a simple model
    model = SimpleModel()
    model.eval()

    named_modules = dict(model.named_modules(remove_duplicate=False))
    assert len(named_modules) == 5
    assert "fc1" in named_modules
    assert "relu" in named_modules
    assert "fc2" in named_modules
    assert "fc3" in named_modules

    getter = IntermediateLayerGetter(layers=[("fc1", "new_fc1"), ("relu", "new_relu")])
    new_model = getter(model)
    new_model.eval()

    # Create a new model with the same architecture
    new_named_modules = dict(new_model.named_modules(remove_duplicate=False))
    new_named_modules.pop("")
    assert len(new_named_modules) == 2, f"Expected 2, got {len(new_named_modules)}"
    assert (
        "new_fc1" in new_named_modules
    ), f"Expected new_fc1, got {new_named_modules.keys()}"
    assert (
        "new_relu" in new_named_modules
    ), f"Expected new_relu, got {new_named_modules.keys()}"
    assert torch.allclose(
        named_modules["fc1"].weight, new_named_modules["new_fc1"].weight
    )

    # Check if order is correct
    assert list(new_named_modules.keys()) == [
        "new_fc1",
        "new_relu",
    ], f"Expected ['new_fc1', 'new_relu'], in order"

    # Test with repeated layers
    getter = IntermediateLayerGetter(
        layers=[
            ("fc1", "new_fc1"),
            ("fc1", "new_fc2"),
            ("relu", "new_relu_1"),
            ("relu", "new_relu_2"),
            ("fc1", "new_fc1_3"),
        ]
    )
    new_model = getter(model)
    new_model.eval()
    new_named_modules = dict(new_model.named_modules(remove_duplicate=False))
    new_named_modules.pop("")
    assert (
        len(new_named_modules) == 5
    ), f"Expected 5, got {len(dict(new_named_modules))}"
    assert list(new_named_modules.keys()) == [
        "new_fc1",
        "new_fc2",
        "new_relu_1",
        "new_relu_2",
        "new_fc1_3",
    ], f"Expected ['new_fc1', 'new_fc2', 'new_relu_1', 'new_relu_2', 'new_fc1_3'], in order"


def test_intermediate_layer_getter_corner_cases():
    model = SimpleModel()
    model.eval()

    with pytest.raises(ValueError):
        getter = IntermediateLayerGetter(layers=[])
        new_model = getter(model)

    with pytest.raises(ValueError):
        getter = IntermediateLayerGetter(layers="dummy")
        new_model = getter(model)

    with pytest.raises(ValueError):
        getter = IntermediateLayerGetter(layers=1)
        new_model = getter(model)

    with pytest.raises(ValueError):
        getter = IntermediateLayerGetter(layers=[1, 2])
        new_model = getter(model)
