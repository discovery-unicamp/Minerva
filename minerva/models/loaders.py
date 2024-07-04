from collections import OrderedDict
from minerva.utils.typing import PathLike
from typing import Any, Callable, Dict, List, Union
import torch
import wrapt
import re


class ExtractedModel(torch.nn.ModuleDict):
    """Class representing a submodel extracted from a larger model.
    This class runs forward pass through all the modules in the model, in order
    they were added (as OrderedDict).
    """

    def forward(self, x: Any) -> Any:
        """RUns forward pass through all the modules in the model, in order.

        Parameters
        ----------
        x : Any
            The input to forward pass through the model.

        Returns
        -------
        Any
            The output of the forward pass through the model, after passing
            through all the modules.
        """
        for module in self.values():
            x = module(x)
        return x


class IntermediateLayerGetter:
    """This class extracts intermediate layers from a model and create a new
    ExtractedModel with the extracted layers. The ExtractedModel allows
    performing a foward pass though extracted layers. Note that, if the model
    that will be extracted follows a complex structure forward  logic instead
    of a simple sequential logic, this class may not work as expected.
    """

    def __init__(self, layers: Union[List[str], Dict[str, str]]):
        """Extracts intermediate layers from a model and create a new
        ExtractedModel with the extracted layers, in order they were extracted.

        Parameters
        ----------
        layers : Union[List[str], Dict[str, str]]
            This parameter can be:

            - A list of layer names to be extracted from the model. In this
                case, the names of the layers in the ExtractedModel will be the
                same as the names of the layers in the model.

            - A dictionary with the keys being the names of the layers in the
                model and the values being the names of the layers in the
                ExtractedModel. In this case, the names of the layers in the
                ExtractedModel will be the values of the dictionary.

            The name of layers to be extracted can be a regular expression.

            Note that order is not important. The order of the layers added to
            the ExtractedModel will be the same as the order of the layers in
            the model.
        """
        self.layers = layers
        if isinstance(layers, list):
            self.layers = {name: name for name in layers}

    def __call__(self, model: torch.nn.Module) -> ExtractedModel:
        """Extracts intermediate layers from a model and create a new
        ExtractedModel with the extracted layers, in order they were extracted,
        that is, the order in which they are in the model.

        Parameters
        ----------
        model : torch.nn.Module
            The model from which the layers will be extracted.

        Returns
        -------
        torch.nn.Module
            _description_
        """
        # Extract the layers from the model
        layers = OrderedDict()
        for name, module in model.named_children():
            # Check if the name of the layer matches any of the patterns
            if any(re.search(pattern, name) for pattern in self.layers.keys()):
                layers[self.layers[name]] = module

        # Return a new ExtractedModel with the extracted layers
        return ExtractedModel(layers)


class FromPretrained(wrapt.ObjectProxy):
    """This class loads a model from a checkpoint file, extract the desired
    submodel and wraps it in a FromPretrained object. The FromPretrained object
    acts as a proxy to the submodel, allowing to call it as if it was the
    original model. All the attributes and methods of the submodel are
    accessible through the FromPretrained object directly.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        ckpt_path: PathLike,
        extractor: Callable[[torch.nn.Module], torch.nn.Module] = None,
    ):
        """This class perform the following steps:

        1. Load the state_dict from the checkpoint file and load into the model.
       
        2. Extract the desired submodel using the extractor function.


        Parameters
        ----------
        model : torch.nn.Module
            The model to be loaded from the checkpoint file.
        ckpt_path : PathLike
            The path to the checkpoint file from which the model will be loaded.
        extractor : Callable[[torch.nn.Module], torch.nn.Module], optional
            The extractor function to be used to extract the desired submodel
            from the loaded model. The default is None, that is, use the module
            as it is loaded from the file.
        """
        super().__init__(model)
        self.__wrapped__ = model
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        msg = self.__wrapped__.load_state_dict(state_dict)
        print(f"Model loaded from {ckpt_path}: {msg}")
        if extractor is not None:
            self.__wrapped__ = extractor(self.__wrapped__)

    # Aditional methods to make wrapped object callable and to allow pickling
    def __getattr__(self, name):
        return getattr(self.__wrapped__, name)

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)

    def __reduce_ex__(self, proto):
        return self.__wrapped__.__reduce_ex__(proto)


class FromModel(wrapt.ObjectProxy):
    """This class loads a complete model (pickable) from a model file, extract
    the desired submodel and wraps it in a FromModel object. The FromModel
    object acts as a proxy to the submodel, allowing to call it as if it was the
    original model. All the attributes and methods of the submodel are
    accessible through the FromModel object directly.
    """

    def __init__(
        self,
        model_path: PathLike,
        extractor: Callable[[torch.nn.Module], torch.nn.Module] = None,
    ):
        """This class perform the following steps:
        
        1. Load the whole model from the model file (pickable).
        
        2. Extract the desired submodel using the extractor function.
        
        Parameters
        ----------
        model_path : PathLike
            Path to the model file from which the model will be loaded.
        extractor : Callable[[torch.nn.Module], torch.nn.Module], optional
            The extractor function to be used to extract the desired submodel
            from the loaded model. The default is None, that is, use the module
            as it is loaded from the file.
        """

        model = torch.load(model_path, map_location="cpu")
        super().__init__(model)
        self.__wrapped__ = model
        if extractor is not None:
            self.__wrapped__ = extractor(self.__wrapped__)

    def __getattr__(self, name):
        return getattr(self.__wrapped__, name)

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)
