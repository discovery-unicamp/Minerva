import numpy as np
import pytest
from minerva.data.datasets.base import SimpleDataset


class _SimpleReader:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class _SumTransform:
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
