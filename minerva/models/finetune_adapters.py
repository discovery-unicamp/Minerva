import torch
import torch.nn as nn
import math


class LoRA(nn.Module):
    """
    LoRA (Low-Rank Adaptation) for Linear Layers.

    This module applies low-rank adaptation to an existing linear layer. LoRA enables fine-tuning
    of pre-trained models efficiently by introducing learnable low-rank matrices that adapt
    the weights of the original layer while keeping its parameters frozen.

    Parameters
    ----------
    original_module : torch.nn.Module
        The original linear or transformer layer (e.g., `torch.nn.Linear`) to which LoRA is applied.
        It must have `in_features` and `out_features` attributes.
    bias : bool, optional
        Whether to include a bias term in the LoRA adaptation layers. Default is True.
    alpha : float, optional
        The scaling factor for the LoRA output. Default is 1.
    r : int, optional
        The rank of the low-rank matrices used for adaptation. Default is 4.

    Attributes
    ----------
    original_module : torch.nn.Module
        The original module that LoRA adapts.
    matrix_A : torch.nn.Linear
        The low-rank matrix `A` with dimensions `(in_features, r)`.
    matrix_B : torch.nn.Linear
        The low-rank matrix `B` with dimensions `(r, out_features)`.
    scaling : float
        The scaling factor applied to the LoRA adaptation output.

    Methods
    -------
    init_weights():
        Initializes the weights of the low-rank matrices `A` and `B`. Matrix `A` is initialized
        using Kaiming uniform initialization, and matrix `B` is initialized with zeros.
    forward(x):
        Computes the forward pass through the adapted module.

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> from lora_module import LoRA

    >>> # Original linear layer
    >>> original_layer = nn.Linear(128, 64)

    >>> # Wrap the original layer with LoRA
    >>> lora_layer = LoRA(original_layer, alpha=2, r=8)

    >>> # Input tensor
    >>> x = torch.randn(16, 128)  # batch size of 16

    >>> # Forward pass
    >>> output = lora_layer(x)
    >>> print(output.shape)
    torch.Size([16, 64])
    """

    def __init__(
        self,
        original_module: torch.nn.Module,
        bias: bool = True,
        alpha: int = 1,
        r: int = 4,
    ):
        super(LoRA, self).__init__()

        self.original_module = original_module
        self.matrix_A = torch.nn.Linear(original_module.in_features, r, bias=bias)
        self.matrix_B = torch.nn.Linear(r, original_module.out_features, bias=bias)
        self.scaling = alpha / r

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for the low-rank matrices.

        Matrix `A` is initialized with Kaiming uniform initialization, which is suitable for
        layers with ReLU activations. Matrix `B` is initialized with zeros to ensure that
        the original module's behavior is not perturbed at the start.
        """
        torch.nn.init.kaiming_uniform_(self.matrix_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.matrix_B.weight)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the LoRA module.

        Computes the output as the sum of the original module's output and the low-rank
        adaptation output, scaled by the specified `scaling` factor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor with shape `(batch_size, in_features)`.

        Returns
        -------
        torch.Tensor
            The output tensor with shape `(batch_size, out_features)`.

        Notes
        -----
        The output is computed as:
        .. math::
            y = \text{original_module}(x) + \text{scaling} \cdot B(A(x)),
        where `A` and `B` are the learnable low-rank matrices.
        """
        return self.original_module(x) + self.scaling * self.matrix_B(self.matrix_A(x))
