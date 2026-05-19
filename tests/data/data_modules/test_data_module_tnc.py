from pathlib import Path

import numpy as np
import pytest

from minerva.data.data_modules.har_xu_23 import HarDataModule
from minerva.data.datasets.har_xu_23 import HarDataset, TNCDataset


@pytest.mark.parametrize("use_val_with_train", [True, False])
def test_har_data_module(tmp_path, use_val_with_train):
    """
    Test function to verify the behavior of HarDataModule when `use_val_with_train` is True or False.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory unique to the test
    use_val_with_train : bool
        If True, the validation data will be concatenated with the training data.
        If False, the validation data will remain separate.
    """
    # Generate dummy data
    n_samples_train = 10
    n_samples_val = 5
    n_samples_test = 5
    n_timesteps = 100
    n_channels = 6

    # Create dummy training, validation, and test data
    train_data = np.random.rand(n_samples_train, n_timesteps, n_channels)
    val_data = np.random.rand(n_samples_val, n_timesteps, n_channels)
    test_data = np.random.rand(n_samples_test, n_timesteps, n_channels)

    # Save dummy data to temporary directory
    np.save(tmp_path / "train_data.npy", train_data)
    np.save(tmp_path / "val_data.npy", val_data)
    np.save(tmp_path / "test_data.npy", test_data)

    # Initialize HarDataModule with dummy data
    data_module = HarDataModule(
        processed_data_dir=tmp_path,
        window_size=60,
        batch_size=64,
        adf=False,
        use_val_with_train=use_val_with_train,
    )

    # Verify concatenation behavior
    if use_val_with_train:
        expected_train_shape = (
            n_samples_train + n_samples_val,
            n_timesteps,
            n_channels,
        )
        assert data_module.har_train.shape == expected_train_shape
    else:
        expected_train_shape = (n_samples_train, n_timesteps, n_channels)
        expected_val_shape = (n_samples_val, n_timesteps, n_channels)
        assert data_module.har_train.shape == expected_train_shape
        assert data_module.har_val.shape == expected_val_shape
