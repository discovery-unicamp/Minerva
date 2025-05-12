import torch.nn as nn
from typing import Sequence, Optional, List


class MLP(nn.Sequential):
    """
    A flexible multilayer perceptron (MLP) implemented as a subclass of nn.Sequential.

    This class allows you to quickly build an MLP with:
    - Custom layer sizes
    - Configurable activation functions
    - Optional intermediate operations (e.g., BatchNorm, Dropout) after each linear layer
    - An optional final operation (e.g., normalization, final activation)

    Parameters
    ----------
    layer_sizes : Sequence[int]
        A list of integers specifying the sizes of each layer. Must contain at least two values:
        the input and output dimensions.
    activation_cls : type, optional
        The activation function class (must inherit from nn.Module) to use between layers.
        Defaults to nn.ReLU.
    intermediate_ops : Optional[List[Optional[nn.Module]]], optional
        A list of modules (e.g., nn.BatchNorm1d, nn.Dropout) to apply after each linear layer
        and before the activation. Each item corresponds to one linear layer. Use `None` to skip
        an operation for that layer. Must be the same length as the number of linear layers.
    final_op : Optional[nn.Module], optional
        A module to apply after the last layer (e.g., a final activation or normalization).

    *args, **kwargs :
        Additional arguments passed to the activation function constructor.

    Example
    -------
    >>> from torch import nn
    >>> mlp = MLP(
    ...     [128, 256, 64, 10],
    ...     activation_cls=nn.ReLU,
    ...     intermediate_ops=[nn.BatchNorm1d(256), nn.BatchNorm1d(64), None],
    ...     final_op=nn.Sigmoid()
    ... )
    >>> print(mlp)
    MLP(
        (0): Linear(in_features=128, out_features=256, bias=True)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=256, out_features=64, bias=True)
        (4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=64, out_features=10, bias=True)
        (7): Sigmoid()
    )
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation_cls: type = nn.ReLU,
        intermediate_ops: Optional[List[Optional[nn.Module]]] = None,
        final_op: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):

        assert (
            len(layer_sizes) >= 2
        ), "Multilayer perceptron must have at least 2 layers"
        assert all(
            isinstance(ls, int) and ls > 0 for ls in layer_sizes
        ), "All layer sizes must be positive integers"
        assert issubclass(
            activation_cls, nn.Module
        ), "activation_cls must inherit from torch.nn.Module"

        num_layers = len(layer_sizes) - 1

        if intermediate_ops is not None:
            if len(intermediate_ops) != num_layers:
                raise ValueError(
                    f"Length of intermediate_ops ({len(intermediate_ops)}) must match number of layers ({num_layers})"
                )

        layers = []
        for i in range(num_layers):
            in_dim, out_dim = layer_sizes[i], layer_sizes[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))

            if intermediate_ops is not None and intermediate_ops[i] is not None:
                layers.append(intermediate_ops[i])

            if activation_cls is not None:
                layers.append(activation_cls(*args, **kwargs))

        if final_op is not None:
            layers.append(final_op)

        super().__init__(*layers)
