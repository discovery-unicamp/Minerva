import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
import tifffile as tiff
from unittest import mock
from minerva.data.readers.tiff_reader import TiffReader


@pytest.fixture
def create_temp_tiff_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some dummy TIFF files
        file_paths = []
        for i in range(30):
            file_path = Path(temp_dir) / f"image_{i}.tif"
            tiff.imwrite(file_path, np.random.rand(10, 10))
            file_paths.append(file_path)
        yield temp_dir, file_paths


def test_tiff_reader_initialization(create_temp_tiff_files):
    temp_dir, file_paths = create_temp_tiff_files
    reader = TiffReader(temp_dir)
    assert len(reader) == len(file_paths)


def test_tiff_reader_sort_numeric(create_temp_tiff_files):
    temp_dir, file_paths = create_temp_tiff_files
    reader = TiffReader(temp_dir, sort_numeric=True, delimiter="_", key_index=1)
    sorted_files = sorted(file_paths, key=lambda f: int(f.stem.split("_")[1]))
    assert [f.name for f in reader.files] == [f.name for f in sorted_files]


def test_tiff_reader_invalid_directory():
    with pytest.raises(NotADirectoryError):
        TiffReader("invalid_directory")


def test_tiff_reader_getitem(create_temp_tiff_files):
    temp_dir, file_paths = create_temp_tiff_files
    reader = TiffReader(temp_dir)
    for i in range(len(reader)):
        assert isinstance(reader[i], np.ndarray)


def test_tiff_reader_str_repr(create_temp_tiff_files):
    temp_dir, file_paths = create_temp_tiff_files
    reader = TiffReader(temp_dir)
    assert (
        str(reader)
        == f"TiffReader(path={temp_dir}. Number of files: {len(file_paths)}"
    )
    assert repr(reader) == str(reader)
