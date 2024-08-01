import torch
import torch.nn.functional as F

class MaxPoolingAdapter:
    def __init__(self, kernel_size:int=128):
        self.kernel_size = kernel_size

    def __call__(self,tensor: torch.Tensor) -> torch.Any:
        return self.max_pooling_adapter(tensor)
    
    def max_pooling_adapter(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies max pooling to the input tensor.

        This method takes a 3D tensor and performs max pooling along the time dimension.
        The tensor is first transposed, then max pooling is applied, and finally, the tensor is transposed back
        and squeezed to remove the singleton dimension. This operation helps in reducing the dimensionality
        of the tensor while retaining the most significant features.

        Parameters
        ----------
        tensor : torch.Tensor
            The input tensor to be processed. The expected shape of the tensor is (batch_size, time_steps, features).
        
        Returns
        -------
        torch.Tensor
            The processed tensor after applying max pooling. The shape of the tensor will be (batch_size, features).
        
        Examples
        --------
        >>> import torch
        >>> from adapters import MaxPoolingAdapter
        >>> tensor = torch.randn(10, 128, 64)  # Example input tensor with shape (batch_size, time_steps, features)
        >>> adapter = MaxPoolingAdapter(kernel_size=128)
        >>> result = adapter.max_pooling_adapter(tensor)
        >>> print(result.shape)
        torch.Size([10, 64])

        Notes
        -----
        This function is designed to be used as an adapter in deep learning models where dimensionality reduction
        is required. It is particularly useful in scenarios involving time-series data or sequential data processing.
        """
        return F.max_pool1d(tensor.transpose(1, 2), kernel_size=self.kernel_size).transpose(1, 2).squeeze(1)
