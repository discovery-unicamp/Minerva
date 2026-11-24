from collections import OrderedDict

import numpy as np
import pytest

from minerva.data.readers.reader import _Reader
from minerva.transforms.transform import _Transform
from minerva.data.datasets.base import (
    MinervaDataset,
    Subset,
    FractionalSubset,
    FractionalRandomSubset,
    ConcatDataset,
)


class _SimpleReader(_Reader):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class _SumTransform(_Transform):
    """Add a constant to every listed key."""

    def __init__(self, keys_to_transform, constant):
        self.keys_to_transform = keys_to_transform
        self.constant = constant

    def __call__(self, data):
        for key in self.keys_to_transform:
            data[key] = data[key] + self.constant
        return data


# @pytest.fixture(
#     params=[(10), (10, 10), (10, 10, 10)],
#     ids=["1d", "2d", "3d"]
# )
@pytest.fixture
def random_data_reader() -> _SimpleReader:
    return _SimpleReader(np.random.random((10, 10)))


# Not the best way to do this, but it works for now
@pytest.fixture
def random_data_reader2() -> _SimpleReader:
    return _SimpleReader(np.random.random((10, 10)))


@pytest.fixture
def random_sum_transform() -> _SumTransform:
    rand_int = np.random.randint(1, 10)
    return _SumTransform(["image"], rand_int)


@pytest.fixture
def random_sum_transform2() -> _SumTransform:
    rand_int = np.random.randint(1, 10)
    return _SumTransform(["image"], rand_int)


@pytest.fixture
def simple_dataset(random_data_reader):
    return MinervaDataset(OrderedDict(image=random_data_reader))


@pytest.fixture
def simple_dataset2(random_data_reader2):
    return MinervaDataset(OrderedDict(image=random_data_reader2))


def test_minerva_dataset_single_reader(random_data_reader):
    dataset = MinervaDataset(OrderedDict(image=random_data_reader))
    assert len(dataset) == len(random_data_reader)
    sample = dataset[0]
    assert list(sample.keys()) == ["image"]
    np.testing.assert_array_equal(sample["image"], random_data_reader[0])


def test_minerva_dataset_multiple_readers_are_keyed(
    random_data_reader, random_data_reader2
):
    dataset = MinervaDataset(
        OrderedDict(image=random_data_reader, label=random_data_reader2)
    )
    sample = dataset[3]
    assert list(sample.keys()) == ["image", "label"]
    np.testing.assert_array_equal(sample["image"], random_data_reader[3])
    np.testing.assert_array_equal(sample["label"], random_data_reader2[3])


def test_minerva_dataset_applies_one_pipeline_to_whole_sample(
    random_data_reader, random_data_reader2, random_sum_transform
):
    dataset = MinervaDataset(
        OrderedDict(image=random_data_reader, label=random_data_reader2),
        random_sum_transform,
    )
    sample = dataset[0]
    np.testing.assert_array_equal(
        sample["image"], random_data_reader[0] + random_sum_transform.constant
    )
    # label is not listed, so it passes through unchanged
    np.testing.assert_array_equal(sample["label"], random_data_reader2[0])


def test_minerva_dataset_with_pipeline(random_data_reader, random_sum_transform):
    from minerva.transforms import _TransformsPipeline

    t2 = _SumTransform(["image"], 100)
    dataset = MinervaDataset(
        OrderedDict(image=random_data_reader),
        _TransformsPipeline([random_sum_transform, t2]),
    )
    np.testing.assert_array_equal(
        dataset[1]["image"],
        random_data_reader[1] + random_sum_transform.constant + 100,
    )


def test_minerva_dataset_invalid_readers(random_data_reader):
    with pytest.raises(TypeError):
        MinervaDataset([random_data_reader])  # type: ignore
    with pytest.raises(ValueError):
        MinervaDataset(OrderedDict())


def test_minerva_dataset_str(simple_dataset):
    assert "MinervaDataset" in str(simple_dataset)
    assert "'image'" in str(simple_dataset)


def test_subset_indices(simple_dataset):
    indices = [0, 2, 4]
    subset = Subset(simple_dataset, indices)
    assert len(subset) == len(indices)
    for i, idx in enumerate(indices):
        assert all(subset[i]["image"] == simple_dataset[idx]["image"])


def test_subset_str(simple_dataset):
    indices = [1, 3]
    subset = Subset(simple_dataset, indices)
    s = str(subset)
    assert "Subset with 2 samples" in s


def test_fractional_subset_valid(simple_dataset):
    frac = 0.5
    subset = FractionalSubset(simple_dataset, frac)
    expected_len = int(len(simple_dataset) * frac)
    assert len(subset) == expected_len
    assert all(isinstance(i, int) for i in range(len(subset)))


def test_fractional_subset_invalid(simple_dataset):
    with pytest.raises(ValueError):
        FractionalSubset(simple_dataset, 0)
    with pytest.raises(ValueError):
        FractionalSubset(simple_dataset, 1.1)


def test_fractional_subset_str(simple_dataset):
    frac = 0.3
    subset = FractionalSubset(simple_dataset, frac)
    s = str(subset)
    assert "Fractional Subset" in s
    assert f"{frac * 100:.2f}%" in s


def test_fractional_random_subset_valid(simple_dataset):
    frac = 0.4
    subset = FractionalRandomSubset(simple_dataset, frac, seed=42)
    expected_len = int(len(simple_dataset) * frac)
    assert len(subset) == expected_len


def test_fractional_random_subset_reproducibility(simple_dataset):
    frac = 0.6
    subset1 = FractionalRandomSubset(simple_dataset, frac, seed=123)
    subset2 = FractionalRandomSubset(simple_dataset, frac, seed=123)
    assert subset1.indices == subset2.indices


def test_fractional_random_subset_invalid(simple_dataset):
    # 0 is always invalid
    with pytest.raises(ValueError):
        FractionalRandomSubset(simple_dataset, 0)
    # Invalid float
    with pytest.raises(ValueError):
        FractionalRandomSubset(simple_dataset, 1.2)
    # Invalid int
    with pytest.raises(ValueError):
        FractionalRandomSubset(simple_dataset, -1)
    # Value higher than dataset length
    with pytest.raises(ValueError):
        FractionalRandomSubset(simple_dataset, len(simple_dataset) + 1)
    # Invalid type
    with pytest.raises(TypeError):
        FractionalRandomSubset(simple_dataset, "0.5")  # type: ignore


def test_fractional_random_subset_str(simple_dataset):
    frac = 0.7
    subset = FractionalRandomSubset(simple_dataset, frac, seed=99)
    s = str(subset)
    assert "Random Fractional Subset" in s
    assert "seed: 99" in s


def test_concat_dataset(simple_dataset, simple_dataset2):
    concat = ConcatDataset([simple_dataset, simple_dataset2])
    assert len(concat) == len(simple_dataset) + len(simple_dataset2)
    s = str(concat)
    assert "Concatenated" in s
    assert str(len(concat)) in s


def test_fractional_subset_with_integer(simple_dataset):
    count = 3
    subset = FractionalSubset(simple_dataset, count)
    assert len(subset) == count
    assert all(isinstance(i, int) for i in range(len(subset)))


def test_fractional_random_subset_with_integer(simple_dataset):
    count = 4
    subset = FractionalRandomSubset(simple_dataset, count, seed=7)
    assert len(subset) == count
    assert all(isinstance(i, int) for i in range(len(subset)))


def test_fractional_subset_invalid_integer(simple_dataset):
    with pytest.raises(ValueError):
        FractionalSubset(simple_dataset, 0)
    with pytest.raises(ValueError):
        FractionalSubset(simple_dataset, len(simple_dataset) + 1)


def test_fractional_random_subset_invalid_integer(simple_dataset):
    with pytest.raises(ValueError):
        FractionalRandomSubset(simple_dataset, -1)
    with pytest.raises(ValueError):
        FractionalRandomSubset(simple_dataset, len(simple_dataset) + 5)


def test_fractional_subset_invalid_type(simple_dataset):
    with pytest.raises(TypeError):
        FractionalSubset(simple_dataset, "0.5")


def test_fractional_random_subset_invalid_type(simple_dataset):
    with pytest.raises(TypeError):
        FractionalRandomSubset(simple_dataset, [0.2])
