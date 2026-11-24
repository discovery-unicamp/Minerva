from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from minerva.transforms.transform import _as_keys, _RandomTransform


class ClassRatioCropTransform(_RandomTransform):
    """Crop every listed key at the same position, retrying until no class
    dominates the cropped label.

    This is a conditional crop: ``label_key`` is read to decide the crop
    position, and the accepted position is applied to every key in
    ``keys_to_transform`` (which normally includes ``label_key``). All arrays
    must have (H, W) as their last two axes, e.g. (C, H, W) images and (H, W)
    masks.
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        label_key: str,
        target_size: Tuple[int, int],
        max_ratio: float = 0.75,
        max_attempts: int = 10,
        ignore_index: int = -1,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        keys_to_transform : List[str]
            Keys of the entries to crop. They all get the same crop.
        label_key : str
            Key of the (H, W) label whose class ratios decide the crop.
        target_size : Tuple[int, int]
            Crop size as (height, width).
        max_ratio : float, optional
            Maximum fraction of valid pixels one class may occupy in the
            crop, by default 0.75.
        max_attempts : int, optional
            Attempts before accepting the last crop, by default 10.
        ignore_index : int, optional
            Label value excluded from the ratio computation, by default -1.
        seed : int, optional
            Seed for reproducible crop positions, by default None.
        """
        self._init_rng(seed)
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.label_key = label_key
        self.target_h_size = target_size[0]
        self.target_w_size = target_size[1]
        self.max_ratio = max_ratio
        self.max_attempts = max_attempts
        self.ignore_index = ignore_index

    def _draw(self, extent: int) -> int:
        if extent <= 0:
            return 0
        return self._randint(0, extent + 1)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        mask = data[self.label_key]
        h, w = mask.shape[-2:]
        th, tw = self.target_h_size, self.target_w_size

        for i in range(self.max_attempts):
            top = self._draw(h - th)
            left = self._draw(w - tw)
            cropped = mask[..., top : top + th, left : left + tw]
            valid = cropped[cropped != self.ignore_index]
            last = i == self.max_attempts - 1
            if valid.size > 0:
                _, counts = np.unique(valid, return_counts=True)
                if np.max(counts / valid.size) <= self.max_ratio or last:
                    break
            elif last:
                break  # fallback: accept a crop with only ignore_index

        for key in self.keys_to_transform:
            data[key] = data[key][..., top : top + th, left : left + tw]
        return data
