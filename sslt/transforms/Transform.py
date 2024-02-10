
from typing import Any


class _Transform():
    """
    This class represents a transform.
    """

    def __call__(self) -> Any:
        """
        Placeholder method for calling the transform.
        This method should be overridden in subclasses.
        """
        raise NotImplementedError()


class TransformPipeline(_Transform):
    """
    A pipeline of transforms that can be applied to data sequentially.
    """

    def __init__(self, *transforms: _Transform) -> None:
        self.transforms = transforms

    def __call__(self, x: Any) -> Any:
        """
        Apply the transforms in the pipeline to the input data.

        Args:
            x (Any): The input data to be transformed.

        Returns:
            Any: The transformed data.
        """
        for transform in self.transforms:
            x = transform(x)
        return x
