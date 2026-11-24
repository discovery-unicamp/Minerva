from typing import Any, Callable, Dict, List

from .transform import _as_keys, _Transform


class ExternalTransform(_Transform):
    """Run an external single-input callable (e.g. a torchvision transform) on
    every listed key (inward compat wrapper).

    The callable is applied to each key separately, so a random external
    transform draws independent parameters per key. Do not use it to keep
    image and label in sync; use the keyed transforms for that.
    """

    def __init__(self, keys_to_transform: List[str], transform: Callable[[Any], Any]):
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to pass through ``transform``.
        transform : Callable
            Any callable taking one object and returning one object, e.g.
            ``torchvision.transforms.Compose``.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.transform = transform

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._map_keys(data, self.transform)
