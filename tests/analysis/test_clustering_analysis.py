import pytest
from unittest import mock
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from minerva.analysis.clustering_analysis import ClusteringAnalysis
from minerva.data.data_modules import MinervaDataModule
from torch.utils.data import Dataset
import lightning as L
from minerva.models.nets.base import SimpleSupervisedModel


class DummyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class DummyModel(SimpleSupervisedModel):
    def __init__(self):
        super().__init__(
            backbone=torch.nn.Identity(),
            fc=torch.nn.Identity(),
            loss_fn=torch.nn.CrossEntropyLoss(),
        )


@pytest.fixture
def dummy_data_module():
    data = np.arange(1000).reshape(100, 10)
    labels = np.concatenate([np.arange(50), np.arange(50)])
    return MinervaDataModule(
        test_dataset=DummyDataset(data, labels),
        predict_split="test",
        batch_size=3,
    )


@pytest.fixture
def dummy_model():
    return DummyModel()


def test_compute_normal(dummy_data_module, dummy_model):
    analysis = ClusteringAnalysis()
    result = analysis.compute(dummy_model, dummy_data_module)

    assert "silhouette-score" in result
    assert "davies-bouldin-score" in result
    assert isinstance(result["silhouette-score"], float)
    assert isinstance(result["davies-bouldin-score"], float)
