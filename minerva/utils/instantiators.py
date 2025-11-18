from typing import Optional, Union, Type, TypeVar
from jsonargparse import ArgumentParser
import yaml
import json
from minerva.utils.typing import PathLike
from pathlib import Path


class ParserException(Exception):
    """Custom exception for parser errors."""

    pass


T = TypeVar("T")


def _instantiate_cls(
    cls: Type[T],
    config_dict: dict,
    additional_kwargs: Optional[dict] = None,
) -> T:
    """Instantiate a class from a configuration dictionary.

    This function uses the `ArgumentParser` from `jsonargparse` to parse the
    configuration dictionary and instantiate the class with the provided
    arguments. It also allows for additional keyword arguments to be passed.

    Parameters
    ----------
    cls : Type[T]
        The class to instantiate.
    config_dict : dict
        The configuration dictionary containing the parameters for the class,
        following the structure expected by `jsonargparse`.
    additional_kwargs : Optional[dict], optional
        Additional arguments that override or extend the configuration dictionary.
        It uses dot notation for nested parameters, e.g., `{"nested.param": "value"}`.

    Returns
    -------
    T
        An instance of the class `cls` initialized with the parameters from
        `config_dict` and `additional_kwargs`.

    Raises
    ------
    ParserException
        If there is an error during parsing or instantiation, a `ParserException` is raised.
    """
    arg_name = "value"

    parser = ArgumentParser()
    parser.add_argument(arg_name, type=cls, help=f"Instantiator for {cls.__name__}")
    args = [str(config_dict)]
    if additional_kwargs is not None:
        for k, v in additional_kwargs.items():
            if not k.startswith("--"):
                k = f"--{arg_name}.{k}"
            args.append(f"{k}={v}")
    try:
        parsed_config = parser.parse_args(args)
        instantiated_cls = parser.instantiate_classes(parsed_config)
    except SystemExit as e:
        raise ParserException(f"Error instantiating class {cls.__name__}")
    return instantiated_cls.get(arg_name)


def instantiate_cls(
    cls: Type[T],
    config: Union[dict, PathLike],
    additional_kwargs: Optional[dict] = None,
) -> T:
    """Instantiate a class from a configuration dictionary.

    This function uses the `ArgumentParser` from `jsonargparse` to parse the
    configuration dictionary and instantiate the class with the provided
    arguments. It also allows for additional keyword arguments to be passed.

    Parameters
    ----------
    cls : Type[T]
        The class to instantiate.
    config : Union[dict, PathLike]
        The configuration dictionary containing the parameters for the class,
        following the structure expected by `jsonargparse`.
        It can also be a path to a JSON or YAML file containing the configuration.
        If a path is provided, it will read the file and parse its contents
        into a dictionary. Supported formats are JSON (`.json`) and YAML (`.yaml`,
        `.yml`).
    additional_kwargs : Optional[dict], optional
        Additional arguments that override or extend the configuration dictionary.
        It uses dot notation for nested parameters, e.g., `{"nested.param": "value"}`.

    Returns
    -------
    T
        An instance of the class `cls` initialized with the parameters from
        `config_dict` and `additional_kwargs`.

    Raises
    ------
    ParserException
        If there is an error during parsing or instantiation, a `ParserException`
        is raised.
    FileNotFoundError
        If the provided path does not exist when a path-like object is given.
    ValueError
        If the provided configuration is neither a dictionary nor a valid path-like object,
        or if the file format is unsupported.

    Examples
    --------

    >>> from minerva.utils.instantiators import instantiate_cls
    >>> from minerva.models.nets.base import SimpleSupervisedModel
    >>> model_config = {
            "class_path": "minerva.models.nets.base.SimpleSupervisedModel",
            "init_args": {
                "backbone": {
                    "class_path": "minerva.models.nets.time_series.cnns.CNN_PF_Backbone",
                    "init_args": {"include_middle": True},
                },
                "fc": {
                    "class_path": "minerva.models.nets.mlp.MLP",
                    "init_args": {"layer_sizes": [768, 128, 6]},
                },
                "loss_fn": {"class_path": "torch.nn.CrossEntropyLoss"},
                "flatten": True,
            },
        }
    >>> model = instantiate_cls(SimpleSupervisedModel, model_config)
    """
    if isinstance(config, dict):
        config_dict = config
    elif isinstance(config, PathLike):
        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"Config file {path} does not exist.")
        if path.suffix == ".json":
            with open(path, "r") as f:
                config_dict = json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

    else:
        raise ValueError(f"Unsupported config type: {type(config)}.")

    return _instantiate_cls(
        cls=cls,
        config_dict=config_dict,
        additional_kwargs=additional_kwargs,
    )
