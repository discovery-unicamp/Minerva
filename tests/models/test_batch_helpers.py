import pytest
import torch

from minerva.models.batch import (
    get_input,
    get_input_target,
    get_many,
    get_target,
    get_views,
)
from minerva.models.nets.base import SimpleSupervisedModel
from minerva.models.ssl.simclr import SimCLR
from minerva.models.ssl.simsiam import SimSiam
from minerva.models.ssl.barlowtwins import BarlowTwins  # noqa: F401


def test_dict_and_tuple_batches_give_same_values():
    x, y = torch.rand(4, 3), torch.arange(4)
    for batch in ({"image": x, "label": y, "extra": 1}, (x, y)):
        gx, gy = get_input_target(batch)
        assert gx is x and gy is y
        assert get_input(batch) is x and get_target(batch) is y


def test_custom_keys():
    x = torch.rand(2)
    assert get_input({"signal": x}, key="signal") is x


def test_views_and_many():
    a, b = torch.rand(2, 3), torch.rand(2, 3)
    d = {"view1_image": a, "view2_image": b, "label": 0}
    assert get_views(d) == (a, b)
    assert get_views(((a, b), 0)) == (a, b)
    assert get_many({"anchor": a, "neighbor": b, "non_neighbor": a}, ("anchor", "neighbor", "non_neighbor")) == (a, b, a)
    assert get_many((a, b, a), ("anchor", "neighbor", "non_neighbor")) == (a, b, a)


def test_supervised_model_trains_on_dict_batch():
    backbone = torch.nn.Linear(5, 3)
    model = SimpleSupervisedModel(
        backbone=backbone, fc=torch.nn.Identity(), loss_fn=torch.nn.CrossEntropyLoss()
    )
    batch = {"image": torch.rand(4, 5), "label": torch.randint(0, 3, (4,))}
    loss = model._single_step(batch, 0, "train")
    assert torch.isfinite(loss)
    assert model.predict_step(batch, 0).shape == (4, 3)


def test_get_views_bare_tuple():
    a, b = torch.rand(2, 3), torch.rand(2, 3)
    assert get_views((a, b)) == (a, b)
