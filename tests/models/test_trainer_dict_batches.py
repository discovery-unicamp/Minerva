"""Real Lightning runs on dict batches coming out of a DataLoader."""

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from minerva.models.nets.base import SimpleSupervisedModel
from minerva.transforms import (
    MultiViewTransform,
    RandomCropTransform,
    TransposeTransform,
    _TransformsPipeline,
)


class _Supervised(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return {
            "image": np.random.rand(5).astype(np.float32),
            "label": np.int64(idx % 3),
        }


class _Views(Dataset):
    def __init__(self):
        # arrays are (H, W, C); the model wants (C, H, W), so transpose last
        view = _TransformsPipeline(
            [
                RandomCropTransform(["image"], (8, 8)),
                TransposeTransform(["image"], (2, 0, 1)),
            ]
        )
        self.transform = MultiViewTransform(["image"], {"view1": view, "view2": view})

    def __len__(self):
        return 8

    def __getitem__(self, idx):
        data = {"image": np.random.rand(16, 16, 3).astype(np.float32), "label": idx}
        return self.transform(data)


def test_supervised_model_fit_on_dict_batches():
    model = SimpleSupervisedModel(
        backbone=torch.nn.Linear(5, 3),
        fc=torch.nn.Identity(),
        loss_fn=torch.nn.CrossEntropyLoss(),
    )
    loader = DataLoader(_Supervised(), batch_size=4)
    trainer = L.Trainer(
        fast_dev_run=1, accelerator="cpu", logger=False, enable_progress_bar=False
    )
    trainer.fit(model, loader, loader)


def test_simclr_fit_on_multiview_dict_batches():
    from minerva.models.ssl.simclr import SimCLR

    backbone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 16))
    model = SimCLR(backbone=backbone, projection_head=torch.nn.Linear(16, 8))
    loader = DataLoader(_Views(), batch_size=4, num_workers=2)
    trainer = L.Trainer(
        fast_dev_run=1, accelerator="cpu", logger=False, enable_progress_bar=False
    )
    trainer.fit(model, loader)
