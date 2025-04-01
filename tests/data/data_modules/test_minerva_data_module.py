import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from minerva.data.data_modules.base import MinervaDataModule


# Dummy dataset for testing
class DummyDataset(Dataset):
    def __init__(self, size=10):
        self.data = torch.arange(size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Fixtures for train, val, and test datasets
@pytest.fixture
def train_dataset():
    return DummyDataset(size=50)


@pytest.fixture
def val_dataset():
    return DummyDataset(size=20)


@pytest.fixture
def test_dataset():
    return DummyDataset(size=30)


@pytest.fixture
def data_module(train_dataset, val_dataset, test_dataset):
    """Fixture for a fully initialized MinervaDataModule."""
    return MinervaDataModule(
        name="TestModule",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=4,
        num_workers=2,
        shuffle_train=True,
    )


# Test module creation and dataset assignment
def test_data_module_creation(
    data_module, train_dataset, val_dataset, test_dataset
):
    """Test that the MinervaDataModule is initialized correctly."""
    assert data_module.dataset_name == "TestModule"
    assert data_module.train_dataset == train_dataset
    assert data_module.val_dataset == val_dataset
    assert data_module.test_dataset == test_dataset


# Test train dataloader existence
def test_train_dataloader(data_module):
    """Check if train dataloader is created properly."""
    train_loader = data_module.train_dataloader()
    assert isinstance(train_loader, DataLoader)
    assert train_loader.batch_size == 4
    assert train_loader.num_workers == 2
    assert train_loader.dataset == data_module.train_dataset
    assert train_loader.dataset is not None


# Test val dataloader existence
def test_val_dataloader(data_module):
    """Check if validation dataloader is created properly."""
    val_loader = data_module.val_dataloader()
    assert isinstance(val_loader, DataLoader)
    assert val_loader.batch_size == 4
    assert val_loader.num_workers == 2
    assert val_loader.dataset == data_module.val_dataset
    assert val_loader.dataset is not None


# Test test dataloader existence
def test_test_dataloader(data_module):
    """Check if test dataloader is created properly."""
    test_loader = data_module.test_dataloader()
    assert isinstance(test_loader, DataLoader)
    assert test_loader.batch_size == 4
    assert test_loader.num_workers == 2
    assert test_loader.dataset == data_module.test_dataset
    assert test_loader.dataset is not None


# Test predict dataloader with different splits
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_predict_dataloader(train_dataset, val_dataset, test_dataset, split):
    """Check if predict dataloader selects the correct dataset based on split."""
    data_module = MinervaDataModule(
        name="TestModule",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        predict_split=split,
        batch_size=4,
        num_workers=2,
    )

    predict_loader = data_module.predict_dataloader()

    if split == "train":
        assert predict_loader.dataset == data_module.train_dataset
    elif split == "val":
        assert predict_loader.dataset == data_module.val_dataset
    elif split == "test":
        assert predict_loader.dataset == data_module.test_dataset

    assert isinstance(predict_loader, DataLoader)
    assert predict_loader.batch_size == 4
    assert predict_loader.num_workers == 2


# Test predict dataloader when predict_split is None
def test_predict_dataloader_none(train_dataset):
    """Check if predict dataloader raises an error when predict_split is None."""
    data_module = MinervaDataModule(
        name="TestModule",
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        predict_split="test",
    )

    assert data_module.train_dataloader is None
    assert data_module.val_dataloader is None
    assert data_module.test_dataloader is None
    assert data_module.predict_dataloader is None

    data_module = MinervaDataModule(
        name="TestModule",
        train_dataset=train_dataset,
    )
    assert data_module.train_dataloader is not None
    assert data_module.val_dataloader is None
    assert data_module.test_dataloader is None
    assert data_module.predict_dataloader is None


# Test updating additional dataloader kwargs
def test_additional_dataloader_kwargs(train_dataset):
    """Check if additional dataloader kwargs override defaults."""
    additional_kwargs = {"pin_memory": True, "drop_last": True}
    module = MinervaDataModule(
        train_dataset=train_dataset,
        additional_train_dataloader_kwargs=additional_kwargs,
        batch_size=8,
    )
    train_loader = module.train_dataloader()

    assert train_loader.pin_memory is True
    assert train_loader.drop_last is True
    assert train_loader.batch_size == 8  # Ensure batch size is not overridden


# Test __str__ method
def test_module_str_representation(data_module):
    """Ensure the string representation of the module contains expected info."""
    module_str = str(data_module)
    assert "TestModule" in module_str
    assert "Train Dataset" in module_str
    assert "Val Dataset" in module_str
    assert "Test Dataset" in module_str
    assert "Dataloader Configurations" in module_str
    assert "No data." not in module_str
    
    module_repr = repr(data_module)
    assert "TestModule" in module_repr
    assert "Train Dataset" in module_repr
    assert "Val Dataset" in module_repr
    assert "Test Dataset" in module_repr
    assert "Dataloader Configurations" in module_repr
    assert "No data." not in module_repr
    
def test_no_predict_split(train_dataset, val_dataset, test_dataset):
    """Ensure that predict dataloader is None when predict dataset is not provided."""
    data_module = MinervaDataModule(
        name="TestModule",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        predict_split=None,
    )

    assert data_module.predict_dataloader is None
    assert data_module.predict_dataset is None
    assert data_module._predict_split is None
    assert data_module._predict_dataloader_kwargs == {}
    
def test_error_predict_split(train_dataset, val_dataset, test_dataset):
    """Ensure that an error is raised when predict_split is not one of 'train', 'val', 'test'."""
    with pytest.raises(ValueError):
        MinervaDataModule(
            name="TestModule",
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            predict_split="invalid",
        )