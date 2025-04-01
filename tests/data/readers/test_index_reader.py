import pytest

from minerva.data.readers.index_reader import IndexReader


def test_index_reader_getitem():
    reader = IndexReader()
    assert reader[0] == 0
    assert reader[1] == 1
    assert reader[10] == 10


def test_index_reader_len():
    reader = IndexReader(len=100)
    assert len(reader) == 100


def test_index_reader_len_none():
    reader = IndexReader()
    with pytest.raises(TypeError):
        len(reader)


def test_index_reader_getitem_negative_index():
    reader = IndexReader()
    with pytest.raises(IndexError):
        reader[-1]


def test_index_reader_len_zero():
    with pytest.raises(AssertionError):
        reader = IndexReader(len=0)


def test_index_reader_len_negative():
    with pytest.raises(AssertionError):
        reader = IndexReader(len=-1)
