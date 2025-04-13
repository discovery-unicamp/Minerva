import zarr

from minerva.data.readers.patched_array_reader import (
    PatchedArrayReader,
    LazyPaddedPatchedArrayReader,
)
from minerva.utils.typing import PathLike


class PatchedZarrReader(PatchedArrayReader):
    """Reads patches from a Zarr array. This class is a subclass of
    `PatchedArrayReader` and is designed to read patches from a Zarr array.
    """

    def __init__(
        self,
        *args,
        path: PathLike,
        **kwargs,
    ):
        """Reads patches from a Zarr array. This class is a subclass of
        `PatchedArrayReader`. All other parameters are the same as parent class.
        Please refer to the documentation of `PatchedArrayReader` for more
        information.

        Parameters
        ----------
        path : PathLike
            Path to the Zarr array.
        Notes
        -----
        1.  The Zarr array is expected to be stored on disk. If the array is not
            stored on disk, it is recommended to use the `PatchedArrayReader`
            class instead.

        2.  When using padding, the padding is applied to the entire array. This
            will load the entire array into memory. If the array is too large to
            fit into memory, it is recommended to pad before.
            See `LazyPaddedPatchedZarrReader` for an a lazy alternative.

        Examples
        ---------

        ```python
        >>> from pathlib import Path
        >>> data_path = Path("data.zarr")
        >>> reader = PatchedZarrReader(
        ...     path=data_path,
        ...     data_shape=(5, 5),
        ...     stride=(2, 5),
        ... )
        >>> print(len(reader))
        >>> print(reader[0])

        """

        self.path = path
        data = zarr.open(path)
        super().__init__(data=data, *args, **kwargs)


class LazyPaddedPatchedZarrReader(LazyPaddedPatchedArrayReader):
    """Reads patches from a Zarr array. This class is a subclass of
    `LazyPaddedPatchedArrayReader` and is designed to read patches from a Zarr array,
    performing padding in a lazy manner (padding is done in `__getitem__` call).
    If no padding is necessary, use PatchedZarrReader.
    """

    def __init__(
        self,
        *args,
        path: PathLike,
        **kwargs,
    ):
        """Reads patches from a Zarr array. This class is a subclass of
        `LazyPaddedPatchedArrayReader`. All other parameters are the same as parent class.
        Please refer to the documentation of `LazyPaddedPatchedArrayReader` for more
        information. This class can be used when padding is necessary and the whole dataset doesn't fit into memory.

        Parameters
        ----------
        path : PathLike
            Path to the Zarr array.
        Notes
        -----
        1.  The Zarr array is expected to be stored on disk. If the array is not
            stored on disk, it is recommended to use the `LazyPaddedPatchedArrayReader`
            class instead.

        2.  Padding is computed when necessary during `__getitem__` calls.

        Examples
        ---------

        ```python
        >>> from pathlib import Path
        >>> data_path = Path("data.zarr")
        >>> reader = LazyPaddedPatchedZarrReader(
        ...     path=data_path,
        ...     data_shape=(5, 5),
        ...     stride=(2, 5),
        ...     pad_width=((1,1), (0,2)),
        ... )
        >>> print(len(reader))
        >>> print(reader[0])

        """

        self.path = path
        data = zarr.open(path)
        super().__init__(data=data, *args, **kwargs)
