import numpy as np
import torch

from minerva.transforms.tfc import TFCTransform


def test_tfc_augment_numpy():
    x_original = np.random.rand(10, 20)
    x, y1, freq, y2 = TFCTransform().augment(x_original)
    assert x.shape == y1.shape == freq.shape == y2.shape == x_original.shape


def test_tfc_augment_tensor():
    x_original = torch.rand(10, 20)
    x, y1, freq, y2 = TFCTransform().augment(x_original)
    assert x.shape == y1.shape == freq.shape == y2.shape == x_original.shape


def test_tfc_dict_call():
    data = {"signal": np.random.rand(10, 20), "label": 1}
    out = TFCTransform(["signal"])(data)
    assert set(out) == {
        "signal",
        "signal_aug",
        "signal_freq",
        "signal_freq_aug",
        "label",
    }
    assert out["label"] == 1
    assert out["signal_aug"].shape == (10, 20)


def test_tfc_augment_helper_supports_plain_callables():
    import torch

    from minerva.transforms.tfc import TFCTransform, tfc_augment

    x = np.random.rand(4, 3, 32).astype("float32")
    four = tuple(torch.zeros(1) for _ in range(4))
    assert tfc_augment(lambda v: four, x) is four  # legacy __call__ contract
    out = tfc_augment(TFCTransform(), x)  # default transform: augment()
    assert len(out) == 4
