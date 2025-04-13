from collections import OrderedDict
from minerva.utils.typing import PathLike
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import wrapt
import re


class LoadableModule:
    # Interface for loadable modules. This is a dummy class that should be
    # inherited by classes that can be loaded from a file.
    # Allows type hinting for classes that can be loaded from a file.
    pass


class ModuleExtractor:
    # Interface for module extractors. This is a dummy class that should be
    # inherited by classes that can extract modules from a model.
    # Allows type hinting for classes that can extract modules from a model.
    # User should implement the __call__ method.

    def __call__(self, model: torch.nn.Module) -> torch.nn.Module:
        raise NotImplementedError


class ExtractedModel(torch.nn.ModuleDict):
    """Class representing a submodel extracted from a larger model.
    This class runs forward pass through all the modules in the model, in order
    they were added (as OrderedDict).
    """

    def forward(self, x: Any) -> Any:
        """Runs forward pass through all the modules in the model, in order.

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
        for _, module in self.items():
            x = module(x)
        return x


class IntermediateLayerGetter(ModuleExtractor):
    """This class extracts intermediate layers from a model and create a new
    ExtractedModel with the extracted layers. The ExtractedModel allows
    performing a foward pass though extracted layers. Note that, if the model
    that will be extracted follows a complex structure forward  logic instead
    of a simple sequential logic, this class may not work as expected.
    """

    def __init__(self, layers: Union[List[str], Dict[str, str], List[Tuple[str, str]]]):
        """Extracts intermediate layers from a model and create a new
        ExtractedModel with the extracted layers, in order they were extracted.

        Parameters
        ----------
        layers : Union[List[str], Dict[str, str], List[Tuple[str, str]]
            This parameter can be:

            - A list of strings corresponding to the names of the layers to be
                extracted from the model. In this case, the names of the layers
                in the ExtractedModel will be the same as the names of the
                layers in the model.

            - A list of tuples with two strings, the first being the name of
                the layer in the model and the second being the name of the
                layer in the ExtractedModel. In this case, the names of the
                layers in the ExtractedModel will be the second element of the
                tuples. This is the only option allowing repeating extracted
                layers, which is particularly useful when we want to repeat
                non-trainable layers, such as normalization, pooling, and
                activation layers.

            - A dictionary with the keys being the names of the layers in the
                model and the values being the names of the layers in the
                ExtractedModel. In this case, the names of the layers in the
                ExtractedModel will be the values of the dictionary

        Notes
        -----
            - The name of layers to be extracted can be a regular expression.

            - The layers will be extracted in the order they are passed in the
                layers parameter or in the dictionary keys.


        Raises
        ------
        ValueError
            - If the layers parameter is not a list of strings or a list of
            tuples.

            - If the layers parameter is an empty list.

        """
        if not layers:
            raise ValueError("No layers to extract.")

        self.layers = layers
        if isinstance(layers, dict):
            self.layers = [(name, new_name) for name, new_name in layers.items()]
        elif isinstance(layers, list):
            if isinstance(layers[0], str):
                self.layers = [(name, name) for name in layers]
            elif isinstance(layers[0], Iterable):
                # self.layers = layers
                pass
            else:
                raise ValueError(
                    "Invalid type for layers parameter. Must be a list of strings or a list of tuples."
                )
        else:
            raise ValueError(
                "Invalid type for layers parameter. Must be a list of strings or a list of tuples."
            )

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
        for layer_name, new_name in self.layers:
            found = False
            for name, module in model.named_children():
                if re.search(layer_name, name):
                    layers[new_name] = module
                    found = True

            if not found:
                raise ValueError(f"Layer '{layer_name}' not found in model.")

        # Return a new ExtractedModel with the extracted layers
        return ExtractedModel(layers)


class FromPretrained(wrapt.ObjectProxy, LoadableModule):
    # Your docstrings and previous code remain unchanged...

    def __init__(
        self,
        model: torch.nn.Module,
        ckpt_path: Optional[PathLike] = None,
        filter_keys: Optional[List[str]] = None,
        keys_to_rename: Optional[Dict[str, str]] = None,
        strict: bool = False,
        ckpt_key: Optional[str] = "state_dict",
        extractor: Optional[ModuleExtractor] = None,
        error_on_missing_keys: bool = True,
        ckpt_load_weights_only: bool = True,
    ):
        """Load a model from a checkpoint file and wrap it in a FromPretrained
        object. The FromPretrained object acts as a proxy to the model, allowing
        to call it as if it was the original model. All the attributes and
        methods of the model are accessible through the FromPretrained object
        directly.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be loaded (initialized randomly).
        ckpt_path : Optional[PathLike], optional
            The path to the checkpoint file from which the model will be loaded.
            If None, the model will be loaded without any state_dict, that is,
            nothing will be done to the model, it will remain as it is. By
            default None
        filter_keys : Optional[List[str]], optional
            List of regular expressions to filter keys from the state_dict.
            Only keys that match any of the regular expressions will be kept.
            If None, all keys will be kept. By default None.
        keys_to_rename : Optional[Dict[str, str]], optional
            A dictionary with keys being regular expressions and values being
            prefixes to be added to the keys that match the regular expressions.
            If prefix is an empty string, the matched part of the key will be
            removed. The keys that do not match any regular expression will
            remain the same. If a key matches multiple regular expressions, the
            first one will be used. Finally, if a empty string is used as key,
            all keys will have the prefix added (this have priority over other
            keys). By default None
        strict : bool, optional
            If True, the state_dict must match the keys of the model exactly.
            If False, the state_dict can have extra keys that will be ignored.
            By default False
        ckpt_key : Optional[str], optional
            The key in the checkpoint file where the state_dict is stored. If
            None, the whole checkpoint will be used as state_dict. Else, the
            value of the key will be used as state_dict. By default "state_dict".
        extractor : Optional[ModuleExtractor], optional
            Once model is loaded, the extractor will be called with the model
            as argument. The extractor should return the desired submodel (for
            instance, without some final layers). By default None
        error_on_missing_keys : bool, optional
            If True, raise an error if some keys are missing in the state_dict
            when loading the model. If False, ignore missing keys.
            By default True
        ckpt_load_weights_only : bool, optional
            If True, load only the weights from the checkpoint. If False, load
            the whole checkpoint. By default True
        """
        super().__init__(model)
        self.__wrapped__ = model
        if ckpt_path is not None:
            # Load the state_dict from the checkpoint
            ckpt = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=ckpt_load_weights_only,
            )

            # Get the state_dict from the checkpoint
            if ckpt_key is not None:
                state_dict = ckpt.get(ckpt_key, ckpt)
            else:
                state_dict = ckpt

            # Filter keys if needed
            if filter_keys is not None:
                d = OrderedDict()
                # Iterate over all keys in the state_dict
                for k, v in state_dict.items():
                    # Check if key name matches any of the filter keys
                    # If it does, add it to the new state_dict (and break)
                    # Thus, if multiple filter keys match the same key, the
                    #   first one will be used
                    # If no filter matches, the key will be ignored (not added)
                    for pattern in filter_keys:
                        if re.search(pattern, k):
                            d[k] = v
                            break

                # Update the state_dict
                state_dict = d

            # Rename keys with prefix
            if keys_to_rename is not None:
                print(f"Performing key renaming with: {keys_to_rename}")
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_k = k
                    if "" in keys_to_rename:
                        new_k = f"{keys_to_rename['']}{k}"
                    else:
                        # Iterate over all keys to rename. If a key matches the
                        #  regular expression, add the prefix to the key
                        #  and break. Else keep the key as it is.
                        for old_key, new_prefix in keys_to_rename.items():
                            if re.match(old_key, k):
                                # If the new prefix is an empty string, all keys
                                #  we will remove the matched part of the key
                                if new_prefix == "":
                                    new_k = re.sub(old_key, new_prefix, k)
                                    break
                                # If the new prefix is not an empty string, we
                                #  will add the prefix to the key
                                else:
                                    # new_k = f"{new_prefix}{k}"
                                    new_k = re.sub(old_key, new_prefix, k)
                                    break
                            else:
                                continue
                    print(f"\tRenaming key: {k} -> {new_k} (changed: {k != new_k})")
                    new_state_dict[new_k] = v

                state_dict = new_state_dict

            # Load the modified state_dict
            missing_keys, unexpected_keys = self.__wrapped__.load_state_dict(
                state_dict, strict=strict
            )
            if error_on_missing_keys and missing_keys:
                raise ValueError(f"Missing keys: {missing_keys}")

            print(f"Model loaded from {ckpt_path}")

            # Print message with missing and unexpected keys
            if missing_keys:
                print(
                    f"When loading model, the following keys are missing: {missing_keys}"
                )

            if unexpected_keys:
                print(
                    f"When loading model, the following keys are unexpected: {unexpected_keys}. ",
                    end="",
                )
                if not strict:
                    print("Ignoring unexpected keys.")
                else:
                    print()
        else:
            print("WARNING: Model loaded without state_dict.")

        if extractor is not None:
            print("Extracting submodel...")
            self.__wrapped__ = extractor(self.__wrapped__)

    # Aditional methods to make wrapped object callable and to allow pickling
    def __getattr__(self, name):
        return getattr(self.__wrapped__, name)

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)

    def __reduce_ex__(self, proto):
        return self.__wrapped__.__reduce_ex__(proto)

    def __repr__(self):
        return self.__wrapped__.__repr__()

    def __str__(self):
        return self.__wrapped__.__str__()


class FromModel(wrapt.ObjectProxy, LoadableModule):
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
