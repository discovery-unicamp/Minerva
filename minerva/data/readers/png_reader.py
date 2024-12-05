from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from minerva.data.readers.reader import _Reader
from minerva.utils.typing import PathLike


class PNGReader(_Reader):
    """This class loads PNG files from a directory, optionally sorting them
    numerically based on a part of the filename split by a delimiter.
    """

    def __init__(
        self,
        path: PathLike,
        sort_numeric: bool = False,
        delimiter: Optional[str] = None,
        key_index: int = 0,
    ):
        """Load PNG files from a directory.

        Parameters
        ----------
        path : PathLike
            The path to the directory containing the PNG files. Files will be 
            searched recursively.
        sort_numeric : bool, optional
            If True, sorts numerically instead of lexicographically, by default 
            False.
        delimiter : Optional[str], optional
            The delimiter to split filenames into components, by default None.
            For example, if the delimiter is '_', the filename 'image_1.tif' 
            will be split into ['image', '1'], and the sorting will be based on 
            the part at index defined by `key_index`.
        key_index : int, optional
            The index of the part of the filename to use for sorting.
            Only valid if delimiter is not None. By default 0.

        Raises
        ------
        NotADirectoryError
            If the path is not a directory.
        """
        self.path = Path(path)

        if not self.path.is_dir():
            raise NotADirectoryError(f"Path {path} is not a directory")

        self.sort_numeric = sort_numeric
        self.delimiter = delimiter
        self.key_index = key_index

        # Find all PNG files in the directory recursively
        self.files = list(self.path.rglob("*.png"))
        self._sort_files()

        # Print the file stems (name without extension)
        for f in self.files:
            print(f.name)

    def _sort_files(self):
        """Sort files based on the provided sorting options."""
        def sort_key(f: Path):
            # If no delimiter, sort lexicographically by the full filename (stem)
            if not self.delimiter:
                return int(f.stem) if self.sort_numeric and f.stem.isdigit() else f.stem

            # Otherwise, split the filename by the delimiter
            parts = f.stem.split(self.delimiter)

            # If numeric sorting is enabled, use the part specified by key_index
            if self.sort_numeric:
                try:
                    # Try to convert the part to an integer; if successful, sort numerically
                    return int(parts[self.key_index]) if parts[self.key_index].isdigit() else float('inf')
                except (IndexError, ValueError):
                    # If the part is out of range or not a number, treat as a large value
                    return float('inf')
            else:
                return parts[self.key_index]  # If not numeric, use the part as a string for lexicographical sorting

        # Sort the files using the custom sort key
        self.files.sort(key=sort_key)

    def __getitem__(self, index: int) -> np.ndarray:
        """Retrieve the PNG file at the specified index.

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
        return np.array(Image.open(self.files[index].as_posix()))

    def __len__(self) -> int:
        """Return the number of PNG files in the directory.

        Returns
        -------
        int
            The number of PNG files in the directory.
        """
        return len(self.files)

    def __str__(self) -> str:
        return f"PNGReader(path={self.path}. Number of files: {len(self.files)}"
    
    def __repr__(self) -> str:
        return str(self)