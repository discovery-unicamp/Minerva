from typing import Union
from Reader import _Reader
import numpy as np
import tifffile as tiff
from pathlib import Path


class TiffReader(_Reader):
    """
    A class for reading TIFF files.

    Args:
        path (str): The path to the directory containing the TIFF files.

    Raises:
        ValueError: If the provided path is not a directory.

    Attributes:
        path (Path): The path to the directory containing the TIFF files.
        len (int): The number of TIFF files in the directory.

    Methods:
        __getitem__(index: Union[int, slice]) -> np.ndarray:
            Retrieves the TIFF file at the specified index.
        __len__() -> int:
            Returns the number of TIFF files in the directory.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.is_dir():
            raise ValueError(f"Path {path} is not a directory")
        self.len = len(list(self.path.glob("*.tif")))

    def __getitem__(self, index: Union[int, slice]) -> np.ndarray:
        """
        Retrieve the image data at the specified index.

        Parameters:
            index (int or slice): The index or slice to retrieve the image data from.

        Returns:
            np.ndarray: The image data as a NumPy array.

        Raises:
            ValueError: If the specified file does not exist in the given path.
        """
        if (self.path / f"{index}.tif").exists():
            return tiff.imread(sorted(self.path.glob("*.tif"))[index].as_posix())
        else:
            raise ValueError(f"File {index}.tif does not exist in {self.path}")

    def __len__(self) -> int:
        """
        Returns the length of the TiffReader object.

        :return: The length of the TiffReader object.
        :rtype: int
        """
        return self.len
