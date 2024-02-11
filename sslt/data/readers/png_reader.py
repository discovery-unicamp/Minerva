from pathlib import Path
from typing import Union
from reader import _Reader
import numpy as np
from PIL import Image


class PNGReader(_Reader):
    """This class loads a PNG file from a directory. It assumes that the PNG 
    files are named with a number as the filename, starting from 0. This is 
    shown below. 
    
    ```
    /path/
    ├── 0.png
    ├── 1.png
    ├── 2.png
    └── ...
    ```
    
    Thus, the element at index `i` will be the file `i.png`.
    """

    def __init__(self, path: Union[Path, str]):
        """This class loads a PNG file from a directory.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the directory containing the PNG files.
        """
        self.path = Path(path)
        if not self.path.is_dir():
            raise ValueError(f"Path {path} is not a directory")
        self.len = len(list(self.path.glob("*.png")))

    def __getitem__(self, index: int) -> np.ndarray:
        """Retrieve the PNG file at the specified index. The index will be 
        used as the filename of the PNG file.

        Parameters
        ----------
        index : int
            Index of the PNG file to retrieve.

        Returns
        -------
        np.ndarray
            The PNG file as a NumPy array.

        Raises
        ------
        ValueError
            If the specified file does not exist in the given path.
        """
        if (self.path / f"{index}.png").exists():
            return np.array(Image.open(self.path / f"{index}.png"))
        else:
            raise ValueError(f"File {index}.png does not exist in {self.path}")

    def __len__(self) -> int:
        """Return the number of PNG files in the directory.

        Returns
        -------
        int
            The number of PNG files in the directory.
        """
        return len(self.files)
