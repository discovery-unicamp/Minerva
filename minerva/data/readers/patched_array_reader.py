from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from minerva.data.readers.reader import _Reader


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
        self.data_shape = data_shape
        self.stride = stride or self.data_shape
        self.pad_width = pad_width
        self.pad_mode = pad_mode
        self.pad_kwargs = pad_kwargs or {}

        # Should we pad the data?
        if self.pad_width:
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
