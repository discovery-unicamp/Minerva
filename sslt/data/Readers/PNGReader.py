from pathlib import Path
from Reader import _Reader
import numpy as np
from PIL import Image


class PNGReader(_Reader):
    """
    A class for reading PNG files from a directory.

    Args:
        path (str): The path to the directory containing the PNG files.

    Raises:
        ValueError: If the provided path is not a directory.

    Attributes:
        path (Path): The path to the directory containing the PNG files.
        len (int): The number of PNG files in the directory.

    Methods:
        __getitem__(index: int) -> np.ndarray: Returns the image at the specified index as a NumPy array.
        __len__() -> int: Returns the number of PNG files in the directory.

    """

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.is_dir():
            raise ValueError(f"Path {path} is not a directory")
        self.len = len(list(self.path.glob("*.png")))

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Retrieve the image at the specified index.

        Args:
            index (int): The index of the image to retrieve.

        Returns:
            np.ndarray: The image as a NumPy array.

        Raises:
            ValueError: If the image file does not exist.
        """
        if (self.path / f"{index}.png").exists():
            return np.array(Image.open(self.path / f"{index}.png"))
        else:
            raise ValueError(f"File {index}.png does not exist in {self.path}")

    def __len__(self):
        """
        Returns the length of the PNGReader object.
        """
        return self.len
