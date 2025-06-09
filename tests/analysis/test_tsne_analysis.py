import pytest
from unittest import mock
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from minerva.analysis.model_analysis import TSNEAnalysis
from minerva.data.data_modules import MinervaDataModule
from torch.utils.data import Dataset
import lightning as L
from minerva.models.nets.base import SimpleSupervisedModel
from sklearn.manifold import TSNE


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
    labels = np.arange(100)
    return MinervaDataModule(
        test_dataset=DummyDataset(data, labels),
        predict_split="test",
        batch_size=3,
    )


@pytest.fixture
def dummy_model():
    return DummyModel()


def test_compute_saves_png_and_html(tmp_path, dummy_data_module, dummy_model):
    analysis = TSNEAnalysis()
    analysis.set_path(tmp_path)
    result = analysis.compute(dummy_model, dummy_data_module)
    assert (tmp_path / "tsne.png").exists()
    assert (tmp_path / "tsne.html").exists()
    assert "png_path" in result
    assert "html_path" in result
    assert "tnse_df" in result
    assert isinstance(result["tnse_df"], pd.DataFrame)
    assert set(result["tnse_df"].columns) == {"x", "y", "label"}


def test_compute_raises_if_path_not_set(dummy_data_module, dummy_model):
    analysis = TSNEAnalysis()
    with pytest.raises(ValueError, match="Path is not set"):
        analysis.compute(dummy_model, dummy_data_module)


def test_compute_with_different_n_components_raises(dummy_data_module, dummy_model):
    with pytest.raises(AssertionError, match="n_components must be set to 2"):
        TSNEAnalysis(n_components=3)


def test_compute_dataframe_shape(tmp_path, dummy_data_module, dummy_model):
    analysis = TSNEAnalysis()
    analysis.set_path(tmp_path)
    result = analysis.compute(dummy_model, dummy_data_module)
    df = result["tnse_df"]
    assert df.shape[0] == dummy_data_module.test_dataset.data.shape[0]
    assert df.shape[1] == 3


def test_compute_html_not_written_if_disabled(tmp_path, dummy_data_module, dummy_model):
    analysis = TSNEAnalysis(write_html=False)
    analysis.set_path(tmp_path)
    result = analysis.compute(dummy_model, dummy_data_module)
    assert (tmp_path / "tsne.png").exists()
    assert not (tmp_path / "tsne.html").exists()
    assert result["html_path"] is None
