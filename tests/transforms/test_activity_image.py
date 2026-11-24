import numpy as np

from minerva.transforms import ActivityImageTransform


def test_activity_image_transform():
    transform = ActivityImageTransform(["signal"], 9, 68)
    x = np.random.rand(9, 128)
    out = transform({"signal": x, "label": 3})

    assert out["signal"] is x  # source key is kept
    assert out["label"] == 3  # unlisted key passes through
    assert out["signal_activity_image"].shape == (37, 68)
