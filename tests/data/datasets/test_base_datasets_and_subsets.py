import numpy as np
import pytest

from minerva.data.readers.reader import _Reader
from minerva.transforms.transform import _Transform
from minerva.data.datasets.base import (
    SimpleDataset,
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
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, data):
        return data + self.constant


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
    return _SumTransform(rand_int)


@pytest.fixture
def random_sum_transform2() -> _SumTransform:
    rand_int = np.random.randint(1, 10)
    return _SumTransform(rand_int)


@pytest.fixture
def simple_dataset(random_data_reader):
    return SimpleDataset(random_data_reader)


@pytest.fixture
def simple_dataset2(random_data_reader2):
    return SimpleDataset(random_data_reader2)


def test_simple_dataset(random_data_reader: _SimpleReader):
    """Test the SimpleDataset class with a single reader passed as a list and
    as a single object.
    """

    # Single reader (list)
    dataset = SimpleDataset([random_data_reader])
    assert len(dataset) == len(random_data_reader)
    assert all(dataset[0][0] == random_data_reader[0])

    # Single reader (single)
    dataset = SimpleDataset(random_data_reader)
    assert len(dataset) == len(random_data_reader)
    assert all(dataset[0][0] == random_data_reader[0])


def test_simple_dataset_reader_return_single(random_data_reader: _SimpleReader):
    """Test return_single=True with a single reader."""
    dataset = SimpleDataset(random_data_reader, return_single=True)
    assert len(dataset) == len(random_data_reader)
    assert all(dataset[0] == random_data_reader[0])


def test_simple_dataset_multiple_readers(
    random_data_reader: _SimpleReader, random_data_reader2: _SimpleReader
):
    """Test multiple readers passed as a list"""
    dataset = SimpleDataset([random_data_reader, random_data_reader2])
    assert len(dataset) == len(random_data_reader)
    assert all(dataset[0][0] == random_data_reader[0])
    assert all(dataset[0][1] == random_data_reader2[0])


def test_simple_dataset_multiple_readers_return_single(
    random_data_reader: _SimpleReader, random_data_reader2: _SimpleReader
):
    """Test multiple readers returning a single object, instead of a list.
    This should raise an AssertionError, as it is not possible to return a
    single object (and not a tuple) when there are multiple readers.
    """
    with pytest.raises(AssertionError):
        dataset = SimpleDataset(
            [random_data_reader, random_data_reader2], return_single=True
        )


def test_simple_dataset_transform(
    random_data_reader: _SimpleReader, random_sum_transform: _SumTransform
):
    """Test the SimpleDataset class with a single reader and a single transform
    passed as a list and as a single object.
    """
    # List of transforms
    dataset = SimpleDataset([random_data_reader], [random_sum_transform])
    assert len(dataset) == len(random_data_reader)
    assert all(dataset[0][0] == random_data_reader[0] + random_sum_transform.constant)

    # Single transform
    dataset = SimpleDataset([random_data_reader], random_sum_transform)
    assert len(dataset) == len(random_data_reader)
    assert all(dataset[0][0] == random_data_reader[0] + random_sum_transform.constant)


def test_simple_dataset_transform_multiple_readers(
    random_data_reader: _SimpleReader,
    random_data_reader2: _SimpleReader,
    random_sum_transform: _SumTransform,
):
    """Test the SimpleDataset class with multiple readers and a single
    transform. When a single transform object is passed, it is applied to all
    readers. However, passing a 1-element list of transforms to a 2-element list
    of readers should raise an AssertionError.
    """
    # Single transform (single oject)
    dataset = SimpleDataset(
        [random_data_reader, random_data_reader2], random_sum_transform
    )
    assert len(dataset) == len(random_data_reader)
    assert all(dataset[0][0] == random_data_reader[0] + random_sum_transform.constant)
    assert all(dataset[0][1] == random_data_reader2[0] + random_sum_transform.constant)

    # Single transform (list)
    with pytest.raises(AssertionError):
        dataset = SimpleDataset(
            [random_data_reader, random_data_reader2],
            [random_sum_transform],
        )


def test_simple_dataset_multiple_transforms_multiple_readers(
    random_data_reader: _SimpleReader,
    random_data_reader2: _SimpleReader,
    random_sum_transform: _SumTransform,
    random_sum_transform2: _SumTransform,
):
    """Test the SimpleDataset class with multiple readers and multiple transforms
    passed as object. Some of the transforms are None.
    """
    # List of transforms
    dataset = SimpleDataset(
        [random_data_reader, random_data_reader2],
        [random_sum_transform, random_sum_transform2],
    )

    assert len(dataset) == len(random_data_reader)
    assert all(dataset[0][0] == random_data_reader[0] + random_sum_transform.constant)
    assert all(dataset[0][1] == random_data_reader2[0] + random_sum_transform2.constant)

    # None in the list of transforms (1º element)
    dataset = SimpleDataset(
        [random_data_reader, random_data_reader2],
        [random_sum_transform, None],
    )

    assert len(dataset) == len(random_data_reader)
    assert all(dataset[0][0] == random_data_reader[0] + random_sum_transform.constant)
    assert all(dataset[0][1] == random_data_reader2[0])

    # None in the list of transforms (2º element)
    dataset = SimpleDataset(
        [random_data_reader, random_data_reader2],
        [None, random_sum_transform2],
    )

    assert len(dataset) == len(random_data_reader)
    assert all(dataset[0][0] == random_data_reader[0])
    assert all(dataset[0][1] == random_data_reader2[0] + random_sum_transform2.constant)


def test_simple_dataset_transform_multiple_readers_return_single(
    random_data_reader: _SimpleReader,
    random_data_reader2: _SimpleReader,
    random_sum_transform: _SumTransform,
):
    """Test the SimpleDataset class with multiple readers and transforms, but
    returning a single object. This should raise an AssertionError, as it is not
    possible to return a single object (only a tuple) when there are multiple
    readers.
    """
    with pytest.raises(AssertionError):
        dataset = SimpleDataset(
            [random_data_reader, random_data_reader2],
            [random_sum_transform, random_sum_transform],
            return_single=True,
        )


def test_subset_indices(simple_dataset):
    indices = [0, 2, 4]
    subset = Subset(simple_dataset, indices)
    assert len(subset) == len(indices)
    for i, idx in enumerate(indices):
        assert all(subset[i][0] == simple_dataset[idx][0])


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
