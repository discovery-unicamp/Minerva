import pytest
import torch
from minerva.utils.instantiators import instantiate_cls, ParserException
from minerva.models.nets.base import SimpleSupervisedModel
from minerva.models.nets.mlp import MLP

import json
import yaml


@pytest.fixture
def model_config():
    return {
        "backbone": {
            "class_path": "minerva.models.nets.mlp.MLP",
            "init_args": {"layer_sizes": [8, 4, 2]},
        },
        "fc": {
            "class_path": "torch.nn.Linear",
            "init_args": {"in_features": 2, "out_features": 1, "bias": True},
        },
        "loss_fn": {"class_path": "torch.nn.CrossEntropyLoss"},
        "flatten": True,
    }


def test_instantiate_cls_dict(model_config):
    model = instantiate_cls(
        cls=SimpleSupervisedModel,
        config=model_config,
    )

    assert isinstance(model, SimpleSupervisedModel)
    assert isinstance(model.backbone, MLP)
    assert isinstance(model.fc, torch.nn.Linear)
    assert model.backbone[0].in_features == 8
    assert model.backbone[0].out_features == 4
    assert model.backbone[2].in_features == 4
    assert model.backbone[2].out_features == 2
    assert model.fc.in_features == 2
    assert model.fc.out_features == 1
    assert model.flatten is True


def test_instantiate_cls_dict_with_additional_kwargs(model_config):
    additional_kwargs = {"fc.out_features": 10}

    model = instantiate_cls(
        cls=SimpleSupervisedModel,
        config=model_config,
        additional_kwargs=additional_kwargs,
    )

    assert isinstance(model, SimpleSupervisedModel)
    assert isinstance(model.backbone, MLP)
    assert isinstance(model.fc, torch.nn.Linear)
    assert model.backbone[0].in_features == 8
    assert model.backbone[0].out_features == 4
    assert model.backbone[2].in_features == 4
    assert model.backbone[2].out_features == 2
    assert model.fc.in_features == 2
    assert model.fc.out_features == 10
    assert model.flatten is True


def test_instantiate_cls_dict_with_additional_kwargs_2(model_config):
    additional_kwargs = {"backbone.layer_sizes": [10, 5, 2]}

    model = instantiate_cls(
        cls=SimpleSupervisedModel,
        config=model_config,
        additional_kwargs=additional_kwargs,
    )

    assert isinstance(model, SimpleSupervisedModel)
    assert isinstance(model.backbone, MLP)
    assert isinstance(model.fc, torch.nn.Linear)
    assert model.backbone[0].in_features == 10
    assert model.backbone[0].out_features == 5
    assert model.backbone[2].in_features == 5
    assert model.backbone[2].out_features == 2
    assert model.fc.in_features == 2
    assert model.fc.out_features == 1
    assert model.flatten is True


def test_instantiate_cls_dict_error(model_config):
    # Test with an invalid class path in the config
    model_config_invalid = model_config.copy()
    model_config_invalid["backbone"]["class_path"] = "non.existent.Class"

    with pytest.raises(ParserException):
        instantiate_cls(
            cls=SimpleSupervisedModel,
            config=model_config_invalid,
        )


def test_instantiate_cls_json(tmp_path, model_config):
    config_path = tmp_path / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f)

    model = instantiate_cls(
        cls=SimpleSupervisedModel,
        config=config_path,
    )

    assert isinstance(model, SimpleSupervisedModel)
    assert isinstance(model.backbone, MLP)
    assert model.fc.in_features == 2
    assert model.fc.out_features == 1
    assert model.flatten is True


def test_instantiate_cls_yaml(tmp_path, model_config):
    config_path = tmp_path / "model_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(model_config, f)

    model = instantiate_cls(
        cls=SimpleSupervisedModel,
        config=config_path,
    )

    assert isinstance(model, SimpleSupervisedModel)
    assert isinstance(model.backbone, MLP)
    assert model.fc.in_features == 2
    assert model.fc.out_features == 1
    assert model.flatten is True


def test_instantiate_cls_invalid_type():
    with pytest.raises(ValueError):
        instantiate_cls(
            cls=SimpleSupervisedModel,
            config=42,  # Invalid type
        )


def test_instantiate_cls_file_not_found(tmp_path):
    config_path = tmp_path / "non_existent_config.json"

    with pytest.raises(FileNotFoundError):
        instantiate_cls(
            cls=SimpleSupervisedModel,
            config=config_path,  # Non-existent file
        )


def test_instantiate_cls_unsupported_format(tmp_path):
    config_path = tmp_path / "model_config.txt"
    with open(config_path, "w") as f:
        f.write("This is not a valid config file.")

    with pytest.raises(ValueError):
        instantiate_cls(
            cls=SimpleSupervisedModel,
            config=config_path,  # Unsupported format
        )
