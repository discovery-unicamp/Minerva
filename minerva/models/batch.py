"""Access helpers for batches produced by the Minerva 1.0 data pipeline.

In 1.0 every sample, and therefore every collated batch, is a flat dict keyed
by name (see the data pipeline standard). Models read what they need *by key*.
By convention supervised models read ``"image"`` (the input) and ``"label"``
(the target), and two-view SSL techniques read ``"view1_image"`` and
``"view2_image"`` (the output of ``MultiViewTransform``). Every key is
overridable per call.

For compatibility with external loops that still yield tuples (the outward
wrapper), every helper also accepts the legacy positional layout:
``(x, y)`` for supervised batches and ``((view1, view2), y)`` (or bare ``(view1, view2)``) for views.
"""

from typing import Any, Mapping, Sequence, Tuple

INPUT_KEY = "image"
TARGET_KEY = "label"
VIEW_KEYS = ("view1_image", "view2_image")


def get_input(batch: Any, key: str = INPUT_KEY) -> Any:
    """Return the model input of a batch (``batch[key]``, or ``batch[0]`` for tuples)."""
    if isinstance(batch, Mapping):
        return batch[key]
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def get_target(batch: Any, key: str = TARGET_KEY) -> Any:
    """Return the target of a batch (``batch[key]``, or ``batch[1]`` for tuples)."""
    if isinstance(batch, Mapping):
        return batch[key]
    return batch[1]


def get_input_target(
    batch: Any, input_key: str = INPUT_KEY, target_key: str = TARGET_KEY
) -> Tuple[Any, Any]:
    """Return ``(input, target)`` of a batch."""
    return get_input(batch, input_key), get_target(batch, target_key)


def get_views(batch: Any, keys: Sequence[str] = VIEW_KEYS) -> Tuple[Any, ...]:
    """Return the views of a contrastive batch.

    Reads ``batch[key]`` for each key of a dict batch. A legacy tuple batch is
    either ``((view1, view2, ...), label)`` or the bare views
    ``(view1, view2, ...)``.
    """
    if isinstance(batch, Mapping):
        return tuple(batch[k] for k in keys)
    if isinstance(batch[0], (tuple, list)):
        return tuple(batch[0])
    return tuple(batch)


def get_many(batch: Any, keys: Sequence[str]) -> Tuple[Any, ...]:
    """Return ``batch[key]`` for each key; for tuple batches the first ``len(keys)`` elements."""
    if isinstance(batch, Mapping):
        return tuple(batch[k] for k in keys)
    return tuple(batch[: len(keys)])
