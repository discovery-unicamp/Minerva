import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
from unittest import mock
from PIL import Image
from minerva.data.readers.png_reader import PNGReader


@pytest.fixture
def create_temp_png_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some dummy PNG files
        file_paths = []
        for i in range(30):
            file_path = Path(temp_dir) / f"image_{i}.png"
            image = Image.fromarray(
                (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
            )
            image.save(file_path)
            file_paths.append(file_path)
        yield temp_dir, file_paths


def test_png_reader_initialization(create_temp_png_files):
    temp_dir, file_paths = create_temp_png_files
    reader = PNGReader(temp_dir)
    assert len(reader) == len(file_paths)


def test_png_reader_sort_numeric(create_temp_png_files):
    temp_dir, file_paths = create_temp_png_files
    reader = PNGReader(temp_dir, sort_numeric=True, delimiter="_", key_index=1)
    sorted_files = sorted(file_paths, key=lambda f: int(f.stem.split("_")[1]))
    assert [f.name for f in reader.files] == [f.name for f in sorted_files]


def test_png_reader_invalid_directory():
    with pytest.raises(NotADirectoryError):
        PNGReader("invalid_directory")


def test_png_reader_getitem(create_temp_png_files):
    temp_dir, file_paths = create_temp_png_files
    reader = PNGReader(temp_dir)
    for i in range(len(reader)):
        assert isinstance(reader[i], np.ndarray)


def test_png_reader_str_repr(create_temp_png_files):
    temp_dir, file_paths = create_temp_png_files
    reader = PNGReader(temp_dir)
    assert (
        str(reader)
        == f"PNGReader(path={temp_dir}. Number of files: {len(file_paths)}"
    )
    assert repr(reader) == str(reader)
