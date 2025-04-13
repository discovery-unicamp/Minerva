from pathlib import Path
from typing import Optional, Union, List
import numpy as np
from minerva.utils.typing import PathLike
from minerva.data.readers.reader import _Reader


class BaseFileIterator(_Reader):
    """A base class for iterating over files in a directory in a custom sorted
    order.
    """

    def __init__(
        self,
        files: List[PathLike],
        sort_method: Optional[List[str]] = None,
        delimiter: Optional[str] = None,
        key_index: Union[int, List[int]] = 0,
        reverse: bool = False,
    ):
        """Base class for iterating over files in a directory in a custom
        sorted order.

        Parameters
        ----------
        files : PathLike
            A list of file paths to iterate over.
        sort_method : Optional[List[str]], optional
            A list specifying how to sort each part of the filename. Each
            element can  be either "text" (lexicographical) or "numeric"
            (numerically). By default, None, which will use "numeric" if
            numeric parts are detected.
        delimiter : Optional[str], optional
            The delimiter to split filenames into components, by default None.
        key_index : Union[int, List[int]], optional
            The index (or list of indices) of the part(s) of the filename to
            use  for sorting. If a list is provided, files will be sorted
            based on  multiple parts in sequence. Thus, first by the part at
            index 0, then by the part at index 1, and so on. By default 0.
        reverse : bool, optional
            Whether to sort in reverse order, by default False.
        """
        self.files = files
        if isinstance(self.files[0], str):
            self.files = [Path(f) for f in self.files]

        # Handle key_index to be a list if it's a single integer
        self.key_index = key_index if isinstance(key_index, list) else [key_index]

        # Default sort_method to 'numeric' if not provided
        self.sort_method = sort_method or ["text"] * len(self.key_index)

        # Ensure that sort_method and key_index are of the same length
        if len(self.sort_method) != len(self.key_index):
            raise ValueError("sort_method and key_index must have the same length.")

        for method in self.sort_method:
            if method not in ["text", "numeric"]:
                raise ValueError(f"Unknown sorting method: {method}")

        self.delimiter = delimiter
        self.reverse = reverse

        self._sort_files()

    def _sort_files(self):
        """Sort files based on the provided sorting options."""

        def sort_key(f: Path):
            # If no delimiter, sort lexicographically by the full filename (stem)
            if not self.delimiter:
                return self._get_sort_values(f.stem, "text")

            # Otherwise, split the filename by the delimiter
            parts = f.stem.split(self.delimiter)

            # Generate sorting keys for each index
            return tuple(
                self._get_sort_values(parts[i], self.sort_method[j])
                for j, i in enumerate(self.key_index)
            )

        # Sort the files using the custom sort key
        self.files.sort(key=sort_key, reverse=self.reverse)  # type: ignore

    def _get_sort_values(self, value: str, method: str):
        """Get the appropriate sorting value based on the method: 'text' or 'numeric'."""
        if method == "numeric":
            try:
                return int(value) if value.isdigit() else float("inf")
            except ValueError:
                print(f"Warning: Could not convert {value} to a number")
                return float("inf")  # If not a number, treat it as a large value
        elif method == "text":
            return value  # Return the value itself if sorting lexicographically
        else:
            raise ValueError(f"Unknown sorting method: {method}")

    def __getitem__(self, index: int) -> np.ndarray:
        """Retrieve the image file at the specified index.

        Parameters
        ----------
        index : int
            Index of the image file to retrieve.

        Returns
        -------
        np.ndarray
            The image file as a NumPy array.
        """
        raise NotImplementedError("This method must be implemented in the subclass.")

    def __len__(self) -> int:
        """Return the number of image files in the directory.

        Returns
        -------
        int
            The number of image files in the directory.
        """
        return len(self.files)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}. Number of files: {len(self.files)}"

    def __repr__(self) -> str:
        return str(self)
