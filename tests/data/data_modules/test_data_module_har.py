import pytest
import numpy as np
import pandas as pd
from minerva.data.data_modules.har import MultiModalHARSeriesDataModule


@pytest.fixture
def sample_csv_dir(tmp_path):
    # Create a DataFrame with 10 samples and 6 features (4 time steps per feature)
    df = pd.DataFrame(
        {
            "accel-x-0": np.arange(100),
            "accel-x-1": np.arange(100) + 10,
            "accel-x-2": np.arange(100) + 100,
            "accel-x-3": np.arange(100) + 1000,
            "accel-y-0": np.arange(100),
            "accel-y-1": np.arange(100) * 2,
            "accel-y-2": np.arange(100) * 3,
            "accel-y-3": np.arange(100) * 4,
            "accel-z-0": np.arange(100) + 5,
            "accel-z-1": np.arange(100) + 15,
            "accel-z-2": np.arange(100) + 25,
            "accel-z-3": np.arange(100) + 35,
            "gyro-x-0": np.arange(100) - 10,
            "gyro-x-1": np.arange(100) - 20,
            "gyro-x-2": np.arange(100) - 30,
            "gyro-x-3": np.arange(100) - 40,
            "gyro-y-0": np.arange(100) + 1,
            "gyro-y-1": np.arange(100) + 2,
            "gyro-y-2": np.arange(100) + 3,
            "gyro-y-3": np.arange(100) + 4,
            "gyro-z-0": np.arange(100) + 6,
            "gyro-z-1": np.arange(100) + 7,
            "gyro-z-2": np.arange(100) + 8,
            "gyro-z-3": np.arange(100) + 9,
            "standard activity code": [i % 4 for i in range(100)],
        }
    )
    # Save as train.csv, validation.csv, and test.csv
    for split in ["train", "validation", "test"]:
        csv_path = tmp_path / f"{split}.csv"
        df.to_csv(csv_path, index=False)
    return tmp_path


def test_multimdodal_defaults(sample_csv_dir):
    """Test that MultiModalHARSeriesDataModule initializes with default parameters."""
    data_module = MultiModalHARSeriesDataModule(
        data_path=sample_csv_dir,
        feature_prefixes=[
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ],
        features_as_channels=True,
        label="standard activity code",
        batch_size=4,
        cast_to="float32",
        shuffle_train=True,
    )

    data_module.setup("fit")
    data_module.setup("test")

    assert len(data_module.datasets["train"][0]) == 100
    assert len(data_module.datasets["validation"][0]) == 100
    assert len(data_module.datasets["test"][0]) == 100

    assert len(data_module.datasets["train"][1]) == 100
    assert len(data_module.datasets["validation"][1]) == 100
    assert len(data_module.datasets["test"][1]) == 100

    # Single-domain
    assert all(i == 0 for i in data_module.datasets["train"][1])
    assert all(i == 0 for i in data_module.datasets["validation"][1])
    assert all(i == 0 for i in data_module.datasets["test"][1])

    train_dataset = data_module.datasets["train"][0]
    val_dataset = data_module.datasets["validation"][0]
    test_dataset = data_module.datasets["test"][0]

    train_x, train_y = train_dataset[0]
    val_x, val_y = val_dataset[0]
    test_x, test_y = test_dataset[0]

    assert train_x.shape == (6, 4)
    assert val_x.shape == (6, 4)
    assert test_x.shape == (6, 4)

    assert val_y == 0
    assert test_y == 0
    expected_val_y = np.array(
        [
            [0, 10, 100, 1000],  # accel-x
            [0, 0, 0, 0],  # accel-y
            [5, 15, 25, 35],  # accel-z
            [-10, -20, -30, -40],  # gyro-x
            [1, 2, 3, 4],  # gyro-y
            [6, 7, 8, 9],  # gyro-z
        ],
        dtype=np.float32,
    )

    np.testing.assert_equal(val_x, expected_val_y)

    with open(sample_csv_dir / "validation.csv", "r") as f:
        val_df = pd.read_csv(f)
        for r in range(10):
            sample = val_df.iloc[r]
            sample_values = (
                [sample[f"accel-x-{i}"] for i in range(4)]
                + [sample[f"accel-y-{i}"] for i in range(4)]
                + [sample[f"accel-z-{i}"] for i in range(4)]
                + [sample[f"gyro-x-{i}"] for i in range(4)]
                + [sample[f"gyro-y-{i}"] for i in range(4)]
                + [sample[f"gyro-z-{i}"] for i in range(4)]
            )

            val_x = val_dataset[r][0]
            assert np.array_equal(val_x.flatten(), np.array(sample_values))


@pytest.mark.parametrize("data_percentage", [0.1, 0.5, 0.75, 1.0])
def test_data_percentage(sample_csv_dir, data_percentage):
    data_module = MultiModalHARSeriesDataModule(
        data_path=sample_csv_dir,
        feature_prefixes=[
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ],
        features_as_channels=True,
        label="standard activity code",
        batch_size=4,
        cast_to="float32",
        shuffle_train=True,
        data_percentage=data_percentage,
    )

    data_module.setup("fit")
    data_module.setup("test")

    assert len(data_module.datasets["train"][0]) == int(100 * data_percentage)
    assert len(data_module.datasets["validation"][0]) == 100
    assert len(data_module.datasets["test"][0]) == 100


@pytest.mark.parametrize("samples_per_class", [1, 2, 7, 10, 25])
def test_samples_per_class(sample_csv_dir, samples_per_class):
    """Test that samples_per_class gives exactly the requested number per class."""
    data_module = MultiModalHARSeriesDataModule(
        data_path=sample_csv_dir,
        feature_prefixes=[
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ],
        features_as_channels=True,
        label="standard activity code",
        batch_size=4,
        cast_to="float32",
        shuffle_train=True,
        samples_per_class=samples_per_class,
        seed=42,
    )

    data_module.setup("fit")
    data_module.setup("test")

    train_dataset = data_module.datasets["train"][0]

    class_counts = {}
    for i in range(len(train_dataset)):
        _, y = train_dataset[i]
        class_counts[y] = class_counts.get(y, 0) + 1

    for count in class_counts.values():
        assert (
            count == samples_per_class
        ), f"Should have exactly {samples_per_class} samples per class"


def test_samples_per_class_cumulative(sample_csv_dir):
    data_module_1 = MultiModalHARSeriesDataModule(
        data_path=sample_csv_dir,
        feature_prefixes=[
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ],
        features_as_channels=True,
        label="standard activity code",
        batch_size=4,
        cast_to="float32",
        shuffle_train=True,
        samples_per_class=10,
        seed=42,
    )

    data_module_2 = MultiModalHARSeriesDataModule(
        data_path=sample_csv_dir,
        feature_prefixes=[
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ],
        features_as_channels=True,
        label="standard activity code",
        batch_size=4,
        cast_to="float32",
        shuffle_train=True,
        samples_per_class=20,
        seed=42,
    )

    data_module_1.setup("fit")
    data_module_2.setup("fit")

    train_dataset_1 = data_module_1.datasets["train"][0]
    train_dataset_2 = data_module_2.datasets["train"][0]

    xs_1 = [train_dataset_1[i][0] for i in range(len(train_dataset_1))]
    xs_2 = [train_dataset_2[i][0] for i in range(len(train_dataset_2))]

    # Convert each sample to a hashable tuple for easier comparison
    xs_1_set = set([tuple(sample.flatten()) for sample in xs_1])
    xs_2_set = set([tuple(sample.flatten()) for sample in xs_2])

    # Assert that all samples in xs_1 are contained in xs_2
    assert xs_1_set.issubset(
        xs_2_set
    ), "Not all samples from the smaller subset are present in the larger subset"


# Should not be subsets if seeds are different
def test_samples_per_class_cumulative_different_seeds(sample_csv_dir):
    data_module_1 = MultiModalHARSeriesDataModule(
        data_path=sample_csv_dir,
        feature_prefixes=[
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ],
        features_as_channels=True,
        label="standard activity code",
        batch_size=4,
        cast_to="float32",
        shuffle_train=True,
        samples_per_class=10,
        seed=42,
    )

    data_module_2 = MultiModalHARSeriesDataModule(
        data_path=sample_csv_dir,
        feature_prefixes=[
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ],
        features_as_channels=True,
        label="standard activity code",
        batch_size=4,
        cast_to="float32",
        shuffle_train=True,
        samples_per_class=10,
        seed=43,
    )

    data_module_1.setup("fit")
    data_module_2.setup("fit")

    train_dataset_1 = data_module_1.datasets["train"][0]
    train_dataset_2 = data_module_2.datasets["train"][0]

    xs_1 = [train_dataset_1[i][0] for i in range(len(train_dataset_1))]
    xs_2 = [train_dataset_2[i][0] for i in range(len(train_dataset_2))]

    # Convert each sample to a hashable tuple for easier comparison
    xs_1_set = set([tuple(sample.flatten()) for sample in xs_1])
    xs_2_set = set([tuple(sample.flatten()) for sample in xs_2])

    # Assert that all samples in xs_1 are contained in xs_2
    assert not xs_1_set.issubset(
        xs_2_set
    ), "Samples from different seeds should not be subsets of each other"


def test_error_data_percentage_and_samples_per_class(sample_csv_dir):
    """Test that an error is raised if both data_percentage and samples_per_class are set."""
    with pytest.raises(
        ValueError, match="Cannot use both data_percentage and samples_per_class"
    ):
        MultiModalHARSeriesDataModule(
            data_path=sample_csv_dir,
            feature_prefixes=[
                "accel-x",
                "accel-y",
                "accel-z",
                "gyro-x",
                "gyro-y",
                "gyro-z",
            ],
            features_as_channels=True,
            label="standard activity code",
            batch_size=4,
            cast_to="float32",
            shuffle_train=True,
            data_percentage=0.5,
            samples_per_class=10,
        )


def test_error_data_percentage(sample_csv_dir):
    """Test that an error is raised if data_percentage is not between 0 and 1."""
    with pytest.raises(ValueError, match="data_percentage must be between 0 and 1"):
        MultiModalHARSeriesDataModule(
            data_path=sample_csv_dir,
            feature_prefixes=[
                "accel-x",
                "accel-y",
                "accel-z",
                "gyro-x",
                "gyro-y",
                "gyro-z",
            ],
            features_as_channels=True,
            label="standard activity code",
            batch_size=4,
            cast_to="float32",
            shuffle_train=True,
            data_percentage=-0.1,
        )

    with pytest.raises(ValueError, match="data_percentage must be between 0 and 1"):
        MultiModalHARSeriesDataModule(
            data_path=sample_csv_dir,
            feature_prefixes=[
                "accel-x",
                "accel-y",
                "accel-z",
                "gyro-x",
                "gyro-y",
                "gyro-z",
            ],
            features_as_channels=True,
            label="standard activity code",
            batch_size=4,
            cast_to="float32",
            shuffle_train=True,
            data_percentage=1.1,
        )


def test_error_samples_per_class(sample_csv_dir):
    """Test that an error is raised if samples_per_class is not a positive integer."""
    with pytest.raises(
        ValueError, match="samples_per_class must be a positive integer"
    ):
        data_module = MultiModalHARSeriesDataModule(
            data_path=sample_csv_dir,
            feature_prefixes=[
                "accel-x",
                "accel-y",
                "accel-z",
                "gyro-x",
                "gyro-y",
                "gyro-z",
            ],
            features_as_channels=True,
            label="standard activity code",
            batch_size=4,
            cast_to="float32",
            shuffle_train=True,
            samples_per_class=-1,
        )
        data_module.setup("fit")

    with pytest.raises(
        ValueError, match="samples_per_class must be a positive integer"
    ):
        data_module = MultiModalHARSeriesDataModule(
            data_path=sample_csv_dir,
            feature_prefixes=[
                "accel-x",
                "accel-y",
                "accel-z",
                "gyro-x",
                "gyro-y",
                "gyro-z",
            ],
            features_as_channels=True,
            label="standard activity code",
            batch_size=4,
            cast_to="float32",
            shuffle_train=True,
            samples_per_class=0,
        )
        data_module.setup("fit")

    with pytest.raises(ValueError):
        data_module = MultiModalHARSeriesDataModule(
            data_path=sample_csv_dir,
            feature_prefixes=[
                "accel-x",
                "accel-y",
                "accel-z",
                "gyro-x",
                "gyro-y",
                "gyro-z",
            ],
            features_as_channels=True,
            label="standard activity code",
            batch_size=4,
            cast_to="float32",
            shuffle_train=True,
            samples_per_class=100000,
        )
        data_module.setup("fit")
