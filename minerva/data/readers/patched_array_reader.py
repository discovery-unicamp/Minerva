from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from numpy.typing import ArrayLike

from minerva.data.readers.reader import _Reader
from minerva.utils.typing import PathLike


class PatchedArrayReader(_Reader):
    """This class is used to read data from a NumPy array. It is designed to generate
    patches from the data and provides sequential access to them. This class can
    serve as a base class for other readers.

    Assumptions:
    - The input data is expected to be a NumPy-like array, that is, it should
        support NumPy-like indexing.
    - Patches are fixed-size subarrays of the data.
    - Patches can have overlap between them.
    """

    def __init__(
        self,
        data: ArrayLike,
        data_shape: Tuple[int, ...],
        stride: Optional[Tuple[int, ...]] = None,
        pad_width: Optional[Tuple[Tuple[int, int], ...]] = None,
        pad_mode: str = "constant",
        pad_kwargs: Optional[Dict] = None,
    ):
        """Reads data from a NumPy array and generates patches from it.

        Parameters
        ----------
        data : ArrayLike
            The input array from which patches are generated.
        data_shape : Tuple[int, ...]
            The shape of the patches to be extracted. This will be the shape of
            the subarray that is returned when a patch is accessed using
            __getitem__.
        stride : Tuple[int, ...], optional
            The stride between consecutive patches. If `None`, the stide will
            be the same as `data_shape`. By default None
        pad_width : Tuple[Tuple[int, int], ...], optional
            The width of padding to be applied to the data array. By default
            `None`, that is, no padding is applied. Check the documentation of
            `numpy.pad` for more information.
        pad_mode : str, optional
            The padding mode, by default "constant". Check the documentation of
            `numpy.pad` for more information.
        pad_kwargs : dict, optional
            Additional keyword arguments for padding, by default None

        Examples
        --------

        ```python
        >>> import numpy as np
        >>> # Generate a 10x10 array
        >>> data = np.arange(100).reshape(10, 10)
        >>> # Create a reader that generates 5x5 patches with a stride of 2 in the
        >>> # first dimension and 5 in the second dimension.
        >>> reader = PatchedArrayReader(
        >>>    data,
        >>>    data_shape=(5, 5),
        >>>    stride=(2, 5),
        >>> )
        >>> # Printing the number of patches that can be extracted from the data
        >>> print(len(reader))
        6
        >>> # Printing the indices of the patches
        >>> print(reader.indices)
        [(0, 0), (0, 5), (2, 0), (2, 5), (4, 0), (4, 5)]
        >>> # Fetch the first patch and print its shape
        >>> print(reader[0].shape)
        (5, 5)
        >>> # Fetch the third patch and print its content
        >>> print(reader[2])
        [[20 21 22 23 24]
         [30 31 32 33 34]
         [40 41 42 43 44]
         [50 51 52 53 54]
         [60 61 62 63 64]]
        ```

        """
        self.data = data
        self.shape = data.shape
        self.data_shape = data_shape
        assert len(self.data.shape) == len(
            self.data_shape
        ), "Data shape and Patch shape must have same number of dimensions"
        self.stride = stride or self.data_shape
        assert len(self.stride) == len(
            self.data_shape
        ), "Stride shape and Patch shape must have same number of dimensions"
        self.pad_width = pad_width
        self.pad_mode = pad_mode
        self.pad_kwargs = pad_kwargs or {}

        # Should we pad the data?
        if self.pad_width:
            assert len(self.pad_width) == len(
                self.data.shape
            ), "Padding width and data shape must have same length"
            self._pad_data(self.pad_width, self.pad_mode, **self.pad_kwargs)

        self.indices = self._get_patches()

    def _pad_data(
        self, pad_width: Tuple[Tuple[int, int], ...], mode="constant", **kwargs
    ):
        """Apply padding to the data array.

        Parameters
        ----------
        pad_width : Tuple[Tuple[int, int], ...]
            The width of padding to be applied to the data array
        mode : str, optional
            The padding mode, by default "constant"
        """
        self.data = np.pad(self.data, pad_width=pad_width, mode=mode, **kwargs)
        self.shape = self.data.shape

    def _get_patches(self) -> List[Tuple[int, ...]]:
        """Compute the left upper corner indices of the patches that will be
        extracted from the data array. The patches are extracted with a stride
        between them. A list of indices is returned, where each index is a tuple
        of integers representing the coordinates of the left upper corner of the
        patches.

        Returns
        -------
        List[Tuple[int, ...]]
            A list of indices (coordinates) representing the left upper corner
            of the patches.
        """
        indices = []

        # Calculate the maximum index in each dimension
        max_indices = tuple(
            (self.data.shape[i] - self.data_shape[i]) // self.stride[i] + 1
            for i in range(len(self.data.shape))
        )

        # Generate indices for left upper corner of patches
        for index in np.ndindex(*max_indices):
            corner_index = tuple(index[i] * self.stride[i] for i in range(len(index)))
            indices.append(corner_index)

        return indices

    def __len__(self) -> int:
        """Return the number of patches that can be extracted from the data
        array.

        Returns
        -------
        int
            The number of patches that can be extracted from the data array.
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> np.ndarray:
        """Fetch a patch from the data array.

        Parameters
        ----------
        idx : int
            The index of the patch to be fetched.

        Returns
        -------
        np.ndarray
            The patch that was fetched from the data array with shape
            `data_shape`
        """
        left_upper_corner = self.indices[idx]
        slice_obj = tuple(
            slice(i, i + s) for i, s in zip(left_upper_corner, self.data_shape)
        )
        return self.data[slice_obj]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(samples={len(self.indices)}, shape={self.data_shape}, dtype={self.data.dtype})"


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
            data=data,
            data_shape=data_shape,
            stride=stride,
            pad_width=pad_width,
            pad_mode=pad_mode,
            pad_kwargs=pad_kwargs,
        )


class LazyPaddedPatchedArrayReader(PatchedArrayReader):
    """Reads patches from a NumPy array.
    This class is a subclass of `PatchedArrayReader` and is designed to perform padding only when the patch
    consumed by `__get_item__` is in a region that uses the padding (boundary regions).
    If no padding is necessary, use PatchedArrayReader.
    """

    def _pad_data(
        self, pad_width: Tuple[Tuple[int, int], ...], mode="constant", **kwargs
    ):
        """Apply padding to the data array.

        Parameters
        ----------
        pad_width : Tuple[Tuple[int, int], ...]
            The width of padding to be applied to the data array
        mode : str, optional
            The padding mode, by default "constant"
        """
        if mode in [
            "maximum",
            "mean",
            "median",
            "minimum",
            "wrap",
        ]:  # TODO: add support if necessary
            raise NotImplementedError(f"Pad mode not supported: {mode}")
        self.shape = tuple(i + p[0] + p[1] for i, p in zip(self.data.shape, pad_width))

    def _get_patches(self) -> List[Tuple[str, Tuple[int, ...]]]:
        """Compute the left upper corner indices of the patches that will be
        extracted from the data array. The patches are extracted with a stride
        between them. A list of indices is returned, where each index is a tuple
        of integers representing the coordinates of the left upper corner of the
        patches.

        Returns
        -------
        List[Tuple[int, ...]]
            A list of indices (coordinates) representing the left upper corner
            of the patches.
        """
        indices = []

        # Calculate the maximum index in each dimension
        max_indices = tuple(
            (self.shape[i] - self.data_shape[i]) // self.stride[i] + 1
            for i in range(len(self.shape))
        )
        pad_loc_opt = [
            "n",  # no padding necessary
            "l",  # lower padding
            "u",  # upper padding
            "b",  # both lower and upper paddings
        ]
        # Generate indices for left upper corner of patches
        for index in np.ndindex(*max_indices):
            corner_index = tuple(index[i] * self.stride[i] for i in range(len(index)))
            if self.pad_width:
                pad_loc = ["n"] * len(corner_index)
                for i, (ci, p, d, s) in enumerate(
                    zip(corner_index, self.pad_width, self.data.shape, self.data_shape)
                ):
                    cur = 0
                    if ci - p[0] < 0:  # lower boundary check
                        cur += 1
                    if ci - p[0] + s >= d:  # upper boundary check
                        cur += 2
                    pad_loc[i] = pad_loc_opt[cur]
                pad_loc = "".join(pad_loc)
            else:
                pad_loc = "n"
            indices.append((pad_loc, corner_index))

        return indices

    def __getitem__(self, idx: int) -> np.ndarray:
        """Fetch a patch from the data array.

        Parameters
        ----------
        idx : int
            The index of the patch to be fetched.

        Returns
        -------
        np.ndarray
            The patch that was fetched from the data array with shape
            `data_shape`
        """
        pad_loc, padded_left_upper_corner = self.indices[idx]

        if self.pad_width:
            data_pad_width = self.pad_width
        else:
            warnings.warn(
                "Padding is not being used! Non-LazyPadded class is recommended, e.g., PatchedArrayReader"
            )
            data_pad_width = [(0, 0)] * len(self.data_shape)

        original_left_upper_corner = tuple(
            max(i - p[0], 0) for i, p in zip(padded_left_upper_corner, data_pad_width)
        )
        slice_base = tuple(
            slice(i, i + s) for i, s in zip(original_left_upper_corner, self.data_shape)
        )
        base_patch = self.data[slice_base]

        # no padding necessary
        if not ("l" in pad_loc or "u" in pad_loc or "b" in pad_loc):
            item = base_patch

        # padding cases
        elif self.pad_mode in [
            "constant",
            "edge",
            "linear_ramp",
            "empty",
            "reflect",
            "symmetric",
        ]:
            pad_width = []
            for opt, p in zip(pad_loc, data_pad_width):
                cur_pad_l = p[0] if opt in ["l", "b"] else 0
                cur_pad_u = p[1] if opt in ["u", "b"] else 0
                pad_width.append((cur_pad_l, cur_pad_u))
            padded_patch = np.pad(
                base_patch, pad_width=pad_width, mode=self.pad_mode, **self.pad_kwargs
            )
            adjusted_left_upper_corner = tuple(
                (0 if p[0] == 0 else i)
                for i, p in zip(padded_left_upper_corner, pad_width)
            )
            slice_obj = tuple(
                slice(i, i + s)
                for i, s in zip(adjusted_left_upper_corner, self.data_shape)
            )
            item = padded_patch[slice_obj]
        else:
            raise ValueError(f"Invalid Value for pad_mode: {self.pad_mode}")
        return item
