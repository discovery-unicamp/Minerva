from abc import abstractmethod


import inspect
from typing import Any


def parametrizable(cls):
    class Parametrizable(cls):
        def __init__(self, *args, **kwargs):
            # Extract only named parameters
            named_args = inspect.signature(cls.__init__).parameters
            named_args_list = list(named_args.keys())[1:]  # Exclude 'self'
            named_kwargs = {
                k: kwargs[k] for k in named_args_list if k in kwargs
            }
            super().__init__(*args, **kwargs)
            self._parameters = named_kwargs

    return Parametrizable


class Pipeline:
    @abstractmethod
    def run(self) -> Any:
        raise NotImplementedError
