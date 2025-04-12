from torch import nn
from typing import Sequence


class MLP(nn.Sequential):
    """
    A multilayer perceptron (MLP) implemented as a subclass of nn.Sequential.

    This MLP is composed of a sequence of linear layers interleaved with ReLU activation
    functions, except for the final layer which remains purely linear.

    Example
    -------

    >>> mlp = MLP(10, 20, 30, 40)
    >>> print(mlp)
    MLP(
        (0): Linear(in_features=10, out_features=20, bias=True)
        (1): ReLU()
        (2): Linear(in_features=20, out_features=30, bias=True)
        (3): ReLU()
        (4): Linear(in_features=30, out_features=40, bias=True)
    )
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation_cls: type = nn.ReLU,
        *args,
        **kwargs,
    ):
        """
        Initializes the MLP with specified layer sizes.

        Parameters
        ----------
        layer_sizes : Sequence[int]
            A sequence of positive integers indicating the size of each layer.
            At least two integers are required, representing the input and output layers.
        activation_cls : type
            The class of the activation function to use between layers. Default is nn.ReLU.
        *args
            Additional arguments passed to the activation function.
        **kwargs
            Additional keyword arguments passed to the activation function.

        Raises
        ------
        AssertionError
            If fewer than two layer sizes are provided or if any layer size is not a positive integer.
        AssertionError
            If activation_cls does not inherit from torch.nn.Module.
        """

        assert (
            len(layer_sizes) >= 2
        ), "Multilayer perceptron must have at least 2 layers"
        assert all(
            ls > 0 and isinstance(ls, int) for ls in layer_sizes
        ), "All layer sizes must be positive integers"

        assert issubclass(
            activation_cls, nn.Module
        ), "activation_cls must inherit from torch.nn.Module"

        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(activation_cls(*args, **kwargs))
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        super().__init__(*layers)
