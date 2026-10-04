import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from minerva.analysis.metrics.fid_score import compute_frechet_distance, compute_ts_fid


def test_fid_identical_data_returns_zero():
    data = np.array([[-1.0], [0.0], [1.0]])

    score = compute_frechet_distance(data, data.copy())

    assert score == pytest.approx(0.0)


def test_fid_shifted_data():
    real = np.array([[-1.0], [0.0], [1.0]])
    generated = real + 2

    score = compute_frechet_distance(real, generated)

    assert score == pytest.approx(4.0)


def test_fid_requires_at_least_two_samples():
    real = np.zeros((1, 2))
    generated = np.zeros((3, 2))

    with pytest.raises(ValueError, match="At least two samples"):
        compute_frechet_distance(real, generated)


def test_fid_requires_matching_feature_counts():
    real = np.zeros((3, 2))
    generated = np.zeros((3, 4))

    with pytest.raises(ValueError, match="equal dimensions"):
        compute_frechet_distance(real, generated)


@pytest.fixture
def ts2vec_encoder(monkeypatch):
    # Replace the optional encoder so these unit tests do not train TS2Vec.
    encoder = Mock()
    encoder.encode.return_value = np.array([[-1.0], [0.0], [1.0]])
    module = SimpleNamespace(TS2Vec=Mock(return_value=encoder))
    monkeypatch.setitem(sys.modules, "ts2vec.ts2vec", module)
    return encoder


def test_ts_fid_trains_encoder_on_real_data(ts2vec_encoder):
    real = np.ones((3, 8, 1))
    generated = np.zeros((3, 8, 1))

    compute_ts_fid(real, generated, device="cpu", fit_kwargs={"n_iters": 1})

    ts2vec_encoder.fit.assert_called_once_with(real, verbose=False, n_iters=1)


def test_ts_fid_compares_encoded_data(ts2vec_encoder):
    real = np.ones((3, 8, 1))
    generated = np.zeros((3, 8, 1))
    ts2vec_encoder.encode.side_effect = [
        np.array([[-1.0], [0.0], [1.0]]),
        np.array([[1.0], [2.0], [3.0]]),
    ]

    score = compute_ts_fid(real, generated, device="cpu")

    assert score == pytest.approx(4.0)


def test_ts_fid_requires_matching_channel_counts():
    real = np.ones((3, 8, 1))
    generated = np.ones((3, 8, 2))

    with pytest.raises(ValueError, match="equal channel counts"):
        compute_ts_fid(real, generated)
