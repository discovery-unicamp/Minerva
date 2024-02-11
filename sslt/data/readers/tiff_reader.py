from typing import Union
from sslt.data.readers.reader import _Reader
import numpy as np
import tifffile as tiff
from pathlib import Path


class TiffReader(_Reader):
    """This class loads a TIFF file from a directory. It assumes that the TIFF 
    files are named with a number as the filename, starting from 0. This is 
    shown below. 
    
    ```
    /path/
    ├── 0.tiff
    ├── 1.tiff
    ├── 2.tiff
    └── ...
    ```
    
    Thus, the element at index `i` will be the file `i.tiff`.
    """
    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.is_dir():
            raise ValueError(f"Path {path} is not a directory")
        self.len = len(list(self.path.glob("*.tif")))

    def __getitem__(self, index: Union[int, slice]) -> np.ndarray:
        """Retrieve the TIFF file at the specified index. The index will be 
        used as the filename of the TIFF file.

        Parameters
        ----------
        index : int
            Index of the TIFF file to retrieve.

        Returns
        -------
        np.ndarray
            The TIFF file as a NumPy array.

        Raises
        ------
        ValueError
            If the specified file does not exist in the given path.
        """
        if (self.path / f"{index}.tif").exists():
            return tiff.imread(sorted(self.path.glob("*.tif"))[index].as_posix())
        else:
            raise ValueError(f"File {index}.tif does not exist in {self.path}")

    def __len__(self) -> int:
        """Return the number of TIFF files in the directory.

        Returns
        -------
        int
            The number of TIFF files in the directory.
        """
        return len(self.files)
