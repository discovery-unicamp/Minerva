from torch import nn

class MLP(nn.Sequential):
    """
    A multilayer perceptron (MLP) implemented as a subclass of nn.Sequential.
    
    The MLP consists of a series of linear layers interleaved with ReLU activation functions,
    except for the last layer which is purely linear.
    
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

    def __init__(self, *layer_sizes: int):
        """
        Initializes the MLP with the given layer sizes.

        Parameters
        ----------
        *layer_sizes: int
            A variable number of positive integers specifying the size of each layer.
            There must be at least two integers, representing the input and output layers.
        
        Raises
        ------
        AssertionError: If less than two layer sizes are provided.
        
        AssertionError: If any layer size is not a positive integer.
        """
        assert (
            len(layer_sizes) >= 2
        ), "Multilayer perceptron must have at least 2 layers"
        assert all(
            ls > 0 and isinstance(ls, int) for ls in layer_sizes
        ), "All layer sizes must be a positive integer"

        layers = []
        for i in range(len(layer_sizes) - 2):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(layer_sizes[-2], layer_sizes[-1])]

        super().__init__(*layers)
