import pytest
import torch
from minerva.data.datasets.binary_tree_subset import BinaryTreeSubset


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.data = list(range(n))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def test_binary_tree_subset_basic():
    ds = DummyDataset(10)
    subset = BinaryTreeSubset(ds, 5)
    assert len(subset) == 5
    assert len(set(subset.indices)) == 5
    assert all(0 <= idx < len(ds) for idx in subset.indices)


def test_binary_tree_subset_full_dataset():
    ds = DummyDataset(8)
    subset = BinaryTreeSubset(ds, 8)
    assert len(subset) == 8
    assert sorted(subset.indices) == sorted(set(subset.indices))
    assert set(subset.indices).issubset(set(range(len(ds))))


def test_binary_tree_subset_invalid_size_zero():
    ds = DummyDataset(5)
    with pytest.raises(ValueError):
        BinaryTreeSubset(ds, 0)


def test_binary_tree_subset_invalid_size_too_large():
    ds = DummyDataset(4)
    with pytest.raises(ValueError):
        BinaryTreeSubset(ds, 5)


def test_binary_tree_subset_str():
    ds = DummyDataset(6)
    subset = BinaryTreeSubset(ds, 3)
    s = str(subset)
    assert "Binary Tree Subset" in s
    assert "3 samples" in s


def test_binary_tree_subset_indices_distribution():
    ds = DummyDataset(10)
    subset = BinaryTreeSubset(ds, 5)
    assert subset.indices == [1, 2, 5, 7, 8]
