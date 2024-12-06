import pytest
from pathlib import Path
from minerva.data.readers.base_file_iterator import BaseFileIterator


def test_base_file_iterator_initialization():
    files = ["file1.txt", "file2.txt", "file3.txt"]
    iterator = BaseFileIterator(files)
    assert len(iterator) == 3
    assert all(isinstance(f, Path) for f in iterator.files)


def test_base_file_iterator_sorting_numeric_with_underscore():
    files = ["file_10.txt", "file_2.txt", "file_1.txt"]
    iterator = BaseFileIterator(
        files, sort_method=["numeric"], delimiter="_", key_index=1
    )
    sorted_files = [f.name for f in iterator.files]
    assert sorted_files == ["file_1.txt", "file_2.txt", "file_10.txt"]


def test_base_file_iterator_sorting_text_with_underscore():
    files = ["file_B.txt", "file_A.txt", "file_C.txt"]
    iterator = BaseFileIterator(
        files, sort_method=["text"], delimiter="_", key_index=1
    )
    sorted_files = [f.name for f in iterator.files]
    assert sorted_files == ["file_A.txt", "file_B.txt", "file_C.txt"]


def test_base_file_iterator_sorting_multiple_parts_with_underscore():
    files = ["file_1_partA.txt", "file_2_partB.txt", "file_1_partB.txt"]
    iterator = BaseFileIterator(
        files, sort_method=["numeric", "text"], delimiter="_", key_index=[1, 2]
    )
    sorted_files = [f.name for f in iterator.files]
    assert sorted_files == [
        "file_1_partA.txt",
        "file_1_partB.txt",
        "file_2_partB.txt",
    ]


def test_base_file_iterator_sorting_multiple_parts_with_underscore_different_index():
    files = ["file_2_partA.txt", "file_1_partA.txt", "file_1_partB.txt"]
    iterator = BaseFileIterator(
        files, sort_method=["text", "numeric"], delimiter="_", key_index=[2, 1]
    )
    sorted_files = [f.name for f in iterator.files]
    assert sorted_files == [
        "file_1_partA.txt",
        "file_2_partA.txt",
        "file_1_partB.txt",
    ]


def test_base_file_iterator_sorting_reverse_with_underscore():
    files = ["file_1.txt", "file_2.txt", "file_3.txt"]
    iterator = BaseFileIterator(
        files, sort_method=["numeric"], delimiter="_", reverse=True, key_index=1
    )
    sorted_files = [f.name for f in iterator.files]
    assert sorted_files == ["file_3.txt", "file_2.txt", "file_1.txt"]


def test_base_file_iterator_invalid_sort_method():
    files = ["file1.txt", "file2.txt", "file3.txt"]
    with pytest.raises(ValueError):
        BaseFileIterator(files, sort_method=["invalid"])


def test_base_file_iterator_invalid_key_index():
    files = ["file1.txt", "file2.txt", "file3.txt"]
    with pytest.raises(ValueError):
        BaseFileIterator(files, key_index=[1, 2], sort_method=["numeric"])


def test_base_file_iterator_invalid_sort_method_key_index_length():
    files = ["file1.txt", "file2.txt", "file3.txt"]
    with pytest.raises(ValueError):
        BaseFileIterator(files, key_index=[1], sort_method=["numeric", "text"])


def test_base_file_iterator_no_custom_sorting():
    files = ["file3.txt", "file1.txt", "file2.txt"]
    iterator = BaseFileIterator(files)
    unsorted_files = [f.name for f in iterator.files]
    assert unsorted_files == sorted(["file3.txt", "file1.txt", "file2.txt"])
