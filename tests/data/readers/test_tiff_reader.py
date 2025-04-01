import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
from unittest import mock
from PIL import Image
from minerva.data.readers.tiff_reader import TiffReader


@pytest.fixture
def create_temp_tiff_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some dummy TIFF files
        file_paths = []
        for i in range(30):
            file_path = Path(temp_dir) / f"image_{i}.tiff"
            image = Image.fromarray(
                (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
            )
            image.save(file_path)
            file_paths.append(file_path)
        yield temp_dir, file_paths


def test_tiff_reader_initialization(create_temp_tiff_files):
    temp_dir, file_paths = create_temp_tiff_files
    reader = TiffReader(temp_dir)
    assert len(reader) == len(file_paths)


def test_tiff_reader_no_custom_sorting(create_temp_tiff_files):
    temp_dir, file_paths = create_temp_tiff_files
    reader = TiffReader(temp_dir)
    sorted_files = sorted(file_paths)
    assert [f.name for f in reader.files] == [f.name for f in sorted_files]


def test_tiff_reader_sort_numeric(create_temp_tiff_files):
    temp_dir, file_paths = create_temp_tiff_files
    reader = TiffReader(
        temp_dir, sort_method=["numeric"], delimiter="_", key_index=1
    )
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


def test_tiff_reader_getitem_shape(create_temp_tiff_files):
    temp_dir, file_paths = create_temp_tiff_files
    reader = TiffReader(temp_dir)
    for i in range(len(reader)):
        img = reader[i]
        assert img.shape == (10, 10, 3)


def test_tiff_reader_getitem_content(create_temp_tiff_files):
    temp_dir, file_paths = create_temp_tiff_files
    reader = TiffReader(temp_dir)
    for i in range(len(reader)):
        img = reader[i]
        assert np.all(img >= 0) and np.all(img <= 255)


def test_tiff_reader_getitem_index_error(create_temp_tiff_files):
    temp_dir, file_paths = create_temp_tiff_files
    reader = TiffReader(temp_dir)
    with pytest.raises(IndexError):
        reader[len(reader)]


def test_tiff_reader_getitem_reverse(create_temp_tiff_files):
    temp_dir, file_paths = create_temp_tiff_files
    reader = TiffReader(temp_dir, reverse=True)
    for i in range(len(reader)):
        img = reader[i]
        assert isinstance(img, np.ndarray)
