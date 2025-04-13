from typing import List, Tuple
import torch
import torch.nn.functional as F


class MaxPoolingTransposingSqueezingAdapter:
    def __init__(self, kernel_size: int = 128):
        """
        This class takes a 3D tensor and performs max pooling along the time dimension.
        The tensor is first transposed, then max pooling is applied, and finally,
        the tensor is transposed back and squeezed to remove the singleton dimension.
        This operation helps in reducing the dimensionality of the tensor while retaining
        the most significant features.
        It comes from rebar repository https://github.com/maxxu05/rebar , also mentioned at
        the paper https://arxiv.org/pdf/2311.00519 : "At the end of the encoder,
        we utilize a global max pooling layer to pool over time."

        Parameters
        ----------
        kernel_size : int, optional (default=128)
            The size of the window over which the max pooling operation is applied.

        Examples
        --------
        >>> import torch
        >>> from minerva.models.adapters import MaxPoolingTransposingSqueezingAdapter
        >>> tensor = torch.randn(10, 128, 64)  # Example input tensor with shape (batch_size, time_steps, features)
        >>> adapter = MaxPoolingTransposingSqueezingAdapter(kernel_size=128)
        >>> result = adapter(tensor)
        >>> print(result.shape)
        torch.Size([10, 64])

        Notes
        -----
        This class is designed to be used as an adapter in deep learning models where
        dimensionality reduction is required. It is particularly useful in scenarios involving
        time-series data or sequential data processing.
        """
        self.kernel_size = kernel_size

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.max_pooling_adapter(tensor)

    def max_pooling_adapter(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies transposing, max polling and squeezing to the input tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            The input tensor to be processed. The expected shape of the tensor is (batch_size, time_steps, features).

        Returns
        -------
        torch.Tensor
            The processed tensor after applying max pooling. The shape of the tensor will be (batch_size, features).
        """
        return (
            F.max_pool1d(tensor.transpose(1, 2), kernel_size=self.kernel_size)
            .transpose(1, 2)
            .squeeze(1)
        )


class PermuteAdapter:
    def __init__(self, permutation: List[int], contiguous: bool = True):
        self.permutation = permutation
        self.contiguous = contiguous

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.permute(*self.permutation)
        if self.contiguous:
            return tensor.contiguous()
        else:
            return tensor
