import os

from minerva.data.readers.zarr_reader import (
    PatchedZarrReader,
    LazyPaddedPatchedZarrReader,
)
from minerva.utils.typing import PathLike


class PatchedMDIOReader(PatchedZarrReader):
    """Reads patches from a MDIO array. This class is a subclass of
    `PatchedZarrReader` and is designed to read patches from the the
    data Zarr array inside thre MDIO array subdirectory.
    """

    def __init__(
        self,
        *args,
        path: PathLike,
        mdio_data: str = "data/chunked_012",
        **kwargs,
    ):
        """Reads patches from a MDIO array. This class is a subclass of
        `PatchedZarrReader`. All other parameters are the same as parent class.
        Please refer to the documentation of `PatchedZarrReader` for more
        information.

        Parameters
        ----------
        path : PathLike
            Path to the MDIO array.
        mdio_data: str
            Name of data array inside MDIO subdirectory. Defaults to 'data/chunked_012'
        Notes
        -----
        1.  The MDIO array is expected to be stored on disk. If the array is not
            stored on disk, it is recommended to use the `PatchedArrayReader`
            class instead.

        2.  When using padding, the padding is applied to the entire array. This
            will load the entire array into memory. If the array is too large to
            fit into memory, it is recommended to pad before.
            See `LazyPaddedPatchedMDIOReader` for an a lazy alternative.

        Examples
        ---------

        ```python
        >>> from pathlib import Path
        >>> data_path = Path("data.mdio")
        >>> reader = PatchedMDIOReader(
        ...     path=data_path,
        ...     data_shape=(5, 5),
        ...     stride=(2, 5),
        ... )
        >>> print(len(reader))
        >>> print(reader[0])

        """

        self.mdio_path = path

        path = os.path.join(path, mdio_data)
        super().__init__(path=path, *args, **kwargs)


class LazyPaddedPatchedMDIOReader(LazyPaddedPatchedZarrReader):
    """Reads patches from a MDIO array. This class is a subclass of
    `LazyPaddedPatchedZarrReader` and is designed to read patches from the the
    data Zarr array inside thre MDIO array subdirectory,
    performing padding in a lazy manner (padding is done in `__getitem__` call).
    If no padding is necessary, use PatchedMDIOReader.
    """

    def __init__(
        self,
        *args,
        path: PathLike,
        mdio_data: str = "data/chunked_012",
        **kwargs,
    ):
        """Reads patches from a MDIO array. This class is a subclass of
        `LazyPaddedPatchedZarrReader`. All other parameters are the same as parent class.
        Please refer to the documentation of `LazyPaddedPatchedZarrReader` for more
        information.

        Parameters
        ----------
        path : PathLike
            Path to the MDIO array.
        mdio_data: str
            Name of data array inside MDIO subdirectory. Defaults to 'data/chunked_012'
        Notes
        -----
        1.  The MDIO array is expected to be stored on disk. If the array is not
            stored on disk, it is recommended to use the `LazyPaddedPatchedArrayReader`
            class instead.

        2.  Padding is computed when necessary during `__getitem__` calls.

        Examples
        ---------

        ```python
        >>> from pathlib import Path
        >>> data_path = Path("data.mdio")
        >>> reader = LazyPaddedPatchedMDIOReader(
        ...     path=data_path,
        ...     data_shape=(5, 5),
        ...     stride=(2, 5),
        ...     pad_width=((1,1), (0,2)),
        ... )
        >>> print(len(reader))
        >>> print(reader[0])

        """

        self.mdio_path = path

        path = os.path.join(path, mdio_data)
        super().__init__(path=path, *args, **kwargs)
