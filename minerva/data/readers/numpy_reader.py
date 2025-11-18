from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from minerva.data.readers.base_file_iterator import BaseFileIterator
from minerva.data.readers.patched_array_reader import PatchedArrayReader
from minerva.utils.typing import PathLike
from pathlib import Path


class NumpyArrayReader(PatchedArrayReader):
    def __init__(
        self,
        data: Union[ArrayLike, PathLike],
        data_shape: Tuple[int, ...],
        stride: Optional[Tuple[int, ...]] = None,
        pad_width: Optional[Tuple[Tuple[int, int], ...]] = None,
        pad_mode: str = "constant",
        pad_kwargs: Optional[Dict] = None,
        allow_pickle: bool = True,
        npz_key: Optional[str] = None,
    ):
        if isinstance(data, PathLike):
            data = Path(data)
            if not data.is_file():
                raise FileNotFoundError(f"File not found: {data}")

            if data.suffix == ".npy":
                data = np.load(data, allow_pickle=allow_pickle)
            elif data.suffix == ".npz":
                data = np.load(data, allow_pickle=allow_pickle)[npz_key]
            else:
                raise ValueError(f"Unsupported file format: {data.suffix}")

        super().__init__(
            data=data,  # type: ignore
            data_shape=data_shape,
            stride=stride,
            pad_width=pad_width,
            pad_mode=pad_mode,
            pad_kwargs=pad_kwargs,
        )


class NumpyFolderReader(BaseFileIterator):
    def __init__(
        self,
        path: PathLike,
        sort_method: Optional[List[str]] = None,
        delimiter: Optional[str] = None,
        key_index: Union[int, List[int]] = 0,
        reverse: bool = False,
        filters: Optional[Union[List[str], str]] = None,
        allow_pickle: bool = True,
        array_key: Optional[str] = None,
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

        files = list(self.root_dir.rglob("*.npy")) + list(self.root_dir.rglob("*.npz"))
        self.allow_pickle = allow_pickle
        self.array_key = array_key
        super().__init__(files, sort_method, delimiter, key_index, reverse, filters)  # type: ignore

    def __getitem__(self, index: int) -> np.ndarray:
        """Retrieve the PNG file at the specified index."""
        p = self.files[index].as_posix()  # type: ignore
        if self.files[index].suffix == ".npz":  # type: ignore
            return np.load(p, allow_pickle=self.allow_pickle)[self.array_key]
        else:
            return np.load(p, allow_pickle=self.allow_pickle)

        return np.open(self.files[index].as_posix())

    def __str__(self) -> str:
        return f"NumpyFolderReader at '{self.root_dir}' ({len(self.files)} files)"
