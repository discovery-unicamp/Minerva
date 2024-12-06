import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from minerva.data.data_modules.parihaka import ParihakaDataModule

@pytest.fixture
def mock_dataset():
    return MagicMock()

@pytest.fixture
def mock_dataloader():
    return MagicMock()

@pytest.fixture
def data_module():
    return ParihakaDataModule(
        root_data_dir="mock_data_dir",
        root_annotation_dir="mock_annotation_dir",
        batch_size=2,
        num_workers=1,
        drop_last=False,
    )

def test_init(data_module):
    assert data_module.root_data_dir == Path("mock_data_dir")
    assert data_module.root_annotation_dir == Path("mock_annotation_dir")
    assert data_module.batch_size == 2
    assert data_module.num_workers == 1
    assert not data_module.drop_last

@patch("minerva.data.data_modules.parihaka.TiffReaderWithNumericSort")
@patch("minerva.data.data_modules.parihaka.PNGReaderWithNumericSort")
@patch("minerva.data.data_modules.parihaka.SupervisedReconstructionDataset")
def test_setup_fit(mock_dataset, mock_png_reader, mock_tiff_reader, data_module):
    data_module.setup(stage="fit")
    assert "train" in data_module.datasets
    assert "val" in data_module.datasets

@patch("minerva.data.data_modules.parihaka.TiffReaderWithNumericSort")
@patch("minerva.data.data_modules.parihaka.PNGReaderWithNumericSort")
@patch("minerva.data.data_modules.parihaka.SupervisedReconstructionDataset")
def test_setup_test(mock_dataset, mock_png_reader, mock_tiff_reader, data_module):
    data_module.setup(stage="test")
    assert "test" in data_module.datasets
    assert "predict" in data_module.datasets

def test_get_dataloader(data_module, mock_dataset, mock_dataloader):
    data_module.datasets["train"] = mock_dataset
    with patch("minerva.data.data_modules.parihaka.DataLoader", return_value=mock_dataloader) as mock_loader:
        dataloader = data_module.train_dataloader()
        mock_loader.assert_called_once_with(
            mock_dataset,
            batch_size=2,
            num_workers=1,
            shuffle=True,
            drop_last=False,
        )
        assert dataloader == mock_dataloader
        

