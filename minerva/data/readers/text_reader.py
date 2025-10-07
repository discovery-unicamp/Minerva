from pathlib import Path
import tifffile as tiff
import numpy as np
from minerva.data.readers.base_file_iterator import BaseFileIterator
from minerva.utils.typing import PathLike
from typing import Optional, Union, List


class TextReader(BaseFileIterator):
    def __init__(
        self,
        path: PathLike,
        sort_method: Optional[List[str]] = None,
        delimiter: Optional[str] = None,
        key_index: Union[int, List[int]] = 0,
        reverse: bool = False,
        filters: Optional[Union[List[str], str]] = None,
    ):
        """Load image files from a directory.

        Parameters
        ----------
        path : Union[Path, str]
            The path to the directory containing the image files. Files will be
            searched recursively.
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
        filters: Optional[Union[List[str], str]]
            An optional string or list of strings containing regular expressions
            with which to filter files by their stems. Files that match at least
            one pattern are kept, and the others are excluded. Defaults to None,
            which means no files are excluded.

        Raises
        ------
        NotADirectoryError
            If the path is not a directory.
        """
        self.root_dir = Path(path)
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"{path} is not a directory.")

        files = list(self.root_dir.rglob("*.txt"))
        super().__init__(files, sort_method, delimiter, key_index, reverse, filters)

    def __getitem__(self, index: int) -> np.ndarray:
        """Retrieve the TIFF file at the specified index."""
        p = self.files[index].as_posix()
        with open(p, "r") as f:
            data = f.read()
        return np.array(data)

    def __str__(self) -> str:
        return f"TextReader at '{self.root_dir}' ({len(self.files)} files)"
