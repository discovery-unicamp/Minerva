from collections import OrderedDict

import lightning as L
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from minerva.data.datasets import (
    FromTupleDataset,
    MinervaDataset,
    SSLDatasetWrapper,
    ToTupleDataset,
)
from minerva.data.readers.reader import _Reader
from minerva.models.nets.base import SimpleSupervisedModel
from minerva.transforms import (
    CastToTransform,
    ExternalTransform,
    MultiViewTransform,
    RandomCropTransform,
    _TransformsPipeline,
)


class _ArrayReader(_Reader):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def _dataset():
    images = np.random.rand(8, 4).astype(np.float64)
    labels = np.arange(8) % 3
    return MinervaDataset(
        OrderedDict(image=_ArrayReader(images), label=_ArrayReader(labels)),
        _TransformsPipeline([CastToTransform(["image"], np.float32)]),
    )


class _TupleDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return np.full(3, idx), idx


def test_to_tuple_dataset():
    ds = _dataset()
    x, y = ToTupleDataset(ds)[2]
    assert x.dtype == np.float32 and y == 2
    assert ToTupleDataset(ds, target_key=None)[2].shape == (4,)
    assert len(ToTupleDataset(ds)) == len(ds)


def test_from_tuple_dataset():
    ds = FromTupleDataset(
        _TupleDataset(),
        transforms=_TransformsPipeline([ExternalTransform(["image"], lambda a: a + 1)]),
    )
    sample = ds[3]
    assert set(sample) == {"image", "label"}
    np.testing.assert_array_equal(sample["image"], np.full(3, 4))
    with pytest.raises(ValueError):
        FromTupleDataset(_TupleDataset(), keys=("a", "b", "c"))[0]


def test_ssl_wrapper_bridges_tuple_dataset_to_dict():
    wrapped = SSLDatasetWrapper(
        _TupleDataset(), _TransformsPipeline([ExternalTransform(["image"], lambda a: a * 2)])
    )
    sample = wrapped[2]
    np.testing.assert_array_equal(sample["image"], np.full(3, 4))
    assert sample["label"] == 2


def test_dataset_to_trainer_end_to_end():
    ds = _dataset()
    model = SimpleSupervisedModel(
        backbone=torch.nn.Linear(4, 3),
        fc=torch.nn.Identity(),
        loss_fn=torch.nn.CrossEntropyLoss(),
    )
    loader = DataLoader(ds, batch_size=4)
    L.Trainer(
        fast_dev_run=1, accelerator="cpu", logger=False, enable_progress_bar=False
    ).fit(model, loader, loader)


def test_dataset_with_multiview_pipeline_gives_view_keys():
    view = _TransformsPipeline([RandomCropTransform(["image"], (2, 2))])
    ds = MinervaDataset(
        OrderedDict(image=_ArrayReader(np.random.rand(4, 1, 6, 6))),
        _TransformsPipeline([MultiViewTransform(["image"], {"view1": view, "view2": view})]),
    )
    assert set(ds[0]) == {"view1_image", "view2_image"}
