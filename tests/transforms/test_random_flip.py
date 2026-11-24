import numpy as np
import torch

from minerva.transforms import RandomFlipTransform


def test_random_flip_single_axis_matches_flip_when_applied():
    torch.manual_seed(0)
    x = np.random.rand(10, 20)
    outs = [RandomFlipTransform(["x"], 0)({"x": x})["x"] for _ in range(20)]
    for out in outs:
        assert out.shape == x.shape
        assert np.array_equal(out, x) or np.array_equal(out, np.flip(x, axis=0))
    assert any(np.array_equal(o, x) for o in outs)
    assert any(np.array_equal(o, np.flip(x, axis=0)) for o in outs)


def test_random_flip_axis_list_flips_axes_independently():
    torch.manual_seed(0)
    x = np.random.rand(6, 7)
    seen = set()
    for _ in range(60):
        out = RandomFlipTransform(["x"], [0, 1], prob=0.5)({"x": x})["x"]
        for flip0 in (False, True):
            for flip1 in (False, True):
                axes = [a for a, f in ((0, flip0), (1, flip1)) if f]
                expected = np.flip(x, axis=axes) if axes else x
                if np.array_equal(out, expected):
                    seen.add((flip0, flip1))
    assert seen == {(False, False), (False, True), (True, False), (True, True)}
