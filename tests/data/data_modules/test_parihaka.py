import pytest
import torch
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
def fake_data_dir(tmp_path):
    """Creates a temporary directory with the following structure:

    root_data_dir/
        train/
            il_1.tif
            il_2.tif
        val/
            il_1.tif
            il_2.tif
        test/
            il_1.tif
            il_2.tif
    root_annotation_dir/
        train/
            il_1.png
            il_2.png
        val/
            il_1.png
            il_2.png
        test/
            il_1.png
            il_2.png
    """

    # Define the root directories
    root_data_dir = tmp_path / "root_data_dir"
    root_annotation_dir = tmp_path / "root_annotation_dir"

    # Create subdirectories
    sub_dirs = ["train", "val", "test"]
    for sub in sub_dirs:
        (root_data_dir / sub).mkdir(parents=True, exist_ok=True)
        (root_annotation_dir / sub).mkdir(parents=True, exist_ok=True)

    # Create fake .tif image files in data directories
    for sub in sub_dirs:
        for i in range(1, 3):  # Create 2 sample files per category
            (root_data_dir / sub / f"il_{i}.tif").write_text("Fake data")
            (root_annotation_dir / sub / f"il_{i}.png").write_text("Fake data")

    return tmp_path


@pytest.fixture
def data_module(fake_data_dir):
    return ParihakaDataModule(
        root_data_dir=fake_data_dir / "root_data_dir",
        root_annotation_dir=fake_data_dir / "root_annotation_dir",
        batch_size=2,
        num_workers=1,
        drop_last=False,
    )


def test_init(data_module, fake_data_dir):
    assert data_module.root_data_dir == (fake_data_dir / "root_data_dir")
    assert data_module.root_annotation_dir == (
        fake_data_dir / "root_annotation_dir"
    )
    assert data_module.batch_size == 2
    assert data_module.num_workers == 1
    assert not data_module.drop_last


def test_setup_fit(data_module):
    data_module.setup(stage="fit")
    assert "train" in data_module.datasets
    assert "val" in data_module.datasets


def test_setup_test(data_module):
    data_module.setup(stage="test")
    assert "test" in data_module.datasets
    assert "predict" in data_module.datasets


def test_get_dataloader(data_module, mock_dataset, mock_dataloader):
    data_module.datasets["train"] = mock_dataset
    with patch(
        "minerva.data.data_modules.parihaka.DataLoader",
        return_value=mock_dataloader,
    ) as mock_loader:
        dataloader = data_module.train_dataloader()
        mock_loader.assert_called_once_with(
            mock_dataset,
            batch_size=2,
            num_workers=1,
            shuffle=True,
            drop_last=False,
        )
        assert dataloader == mock_dataloader


def test_data_loader_kwargs(fake_data_dir):
    data_loader_kwargs = {
        "generator": torch.Generator(),
        "worker_init_fn": MagicMock(),
    }
    data_module = ParihakaDataModule(
        root_data_dir=fake_data_dir / "root_data_dir",
        root_annotation_dir=fake_data_dir / "root_annotation_dir",
        batch_size=2,
        num_workers=1,
        drop_last=False,
        data_loader_kwargs=data_loader_kwargs,
    )

    data_module.setup(stage="fit")
    train_dataloader = data_module.train_dataloader()
    assert train_dataloader.generator == data_loader_kwargs["generator"]
    assert (
        train_dataloader.worker_init_fn == data_loader_kwargs["worker_init_fn"]
    )


def test_data_loader_kwargs_ignoring_parameters(fake_data_dir):
    data_loader_kwargs = {
        "generator": torch.Generator(),
        "worker_init_fn": MagicMock(),
        # Ignored parameters, as they are set in the data module __init__
        "batch_size": 10,
        "num_workers": 10,
        "shuffle": False,
        "drop_last": False,
    }

    data_module = ParihakaDataModule(
        root_data_dir=fake_data_dir / "root_data_dir",
        root_annotation_dir=fake_data_dir / "root_annotation_dir",
        batch_size=2,
        num_workers=1,
        drop_last=True,
        data_loader_kwargs=data_loader_kwargs,
    )

    data_module.setup(stage="fit")
    train_dataloader = data_module.train_dataloader()
    assert train_dataloader.generator == data_loader_kwargs["generator"]
    assert (
        train_dataloader.worker_init_fn == data_loader_kwargs["worker_init_fn"]
    )

    assert train_dataloader.batch_size == 2
    assert train_dataloader.num_workers == 1
    # For train dataloader, shuffle should be True (thus, RandomSampler is used,
    # instead of SequentialSampler)
    assert isinstance(train_dataloader.sampler, torch.utils.data.RandomSampler)
    assert train_dataloader.drop_last == True
