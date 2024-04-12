from abc import abstractmethod
from lightning.pytorch.core.mixins import HyperparametersMixin

from typing import Any


class Pipeline(HyperparametersMixin):
    def __init__(self):
        self.save_hyperparameters()

    @abstractmethod
    def run(self) -> Any:
        raise NotImplementedError

    def __call__(self):
        return self.run()