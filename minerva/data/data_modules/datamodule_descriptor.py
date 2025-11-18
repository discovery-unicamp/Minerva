from abc import ABC, abstractmethod
from typing import Any


class DataModuleDescriptor(ABC):
    """
    Interface for describing datasets.

    Each implementation must initialize with train/val/test datasets,
    and return a detailed description in the __call__() method.
    """

    def __init__(self, train_dataset: Any, val_dataset: Any, test_dataset: Any):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    @abstractmethod
    def __call__(self) -> str:
        """
        Returns a description based on the provided datasets.
        """
        pass
