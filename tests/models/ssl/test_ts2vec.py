import numpy as np
import pytest
import torch

from minerva.models.ssl.ts2vec import (
    TS2Vec,
    centerize_vary_length_series,
    hierarchical_contrastive_loss,
    instance_contrastive_loss,
    pad_nan_to_target,
    split_with_nan,
    take_per_row,
    temporal_contrastive_loss,
    torch_pad_nan,
)


@pytest.fixture
def ts2vec_model():
    return TS2Vec(
        input_dims=2,
        output_dims=4,
        hidden_dims=4,
        depth=1,
        device="cpu",
        batch_size=2,
    )


@pytest.mark.parametrize(
    "loss_fn",
    [
        instance_contrastive_loss,
        temporal_contrastive_loss,
        hierarchical_contrastive_loss,
    ],
)
def test_contrastive_loss_backward(loss_fn):
    z1 = torch.randn(2, 8, 4, requires_grad=True)
    z2 = torch.randn(2, 8, 4, requires_grad=True)

    loss = loss_fn(z1, z2)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0
    assert z1.grad is not None and torch.isfinite(z1.grad).all()
    assert z2.grad is not None and torch.isfinite(z2.grad).all()


@pytest.mark.parametrize(
    "loss_fn, shape",
    [(instance_contrastive_loss, (1, 8, 4)), (temporal_contrastive_loss, (2, 1, 4))],
)
def test_contrastive_loss_without_negatives(loss_fn, shape):
    assert loss_fn(torch.randn(*shape), torch.randn(*shape)).item() == 0


def test_torch_pad_nan():
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    output = torch_pad_nan(x, left=1, right=2, dim=1)

    assert output.shape == (2, 7)
    assert torch.isnan(output[:, :1]).all()
    assert torch.isnan(output[:, -2:]).all()
    torch.testing.assert_close(output[:, 1:5], x)


@pytest.mark.parametrize("both_side, start", [(False, 0), (True, 1)])
def test_pad_nan_to_target(both_side, start):
    x = np.ones((2, 3), dtype=np.float32)
    expected = np.full((2, 6), np.nan, dtype=np.float32)
    expected[:, start : start + 3] = x

    output = pad_nan_to_target(x, 6, axis=1, both_side=both_side)

    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(pad_nan_to_target(x, 2, axis=1), x)


def test_split_with_nan():
    x = np.arange(10, dtype=np.float32).reshape(2, 5)

    output = split_with_nan(x, sections=2, axis=1)

    assert len(output) == 2
    np.testing.assert_array_equal(output[0], x[:, :3])
    np.testing.assert_array_equal(output[1][:, :2], x[:, 3:])
    assert output[1].shape == (2, 3)
    assert np.isnan(output[1][:, -1]).all()


def test_centerize_vary_length_series():
    x = np.array([[[1], [2], [np.nan], [np.nan]], [[np.nan], [np.nan], [3], [4]]])
    expected = np.array(
        [[[np.nan], [1], [2], [np.nan]], [[np.nan], [3], [4], [np.nan]]]
    )

    np.testing.assert_array_equal(centerize_vary_length_series(x), expected)


def test_take_per_row():
    x = torch.arange(24).reshape(2, 6, 2)

    output = take_per_row(x, np.array([1, 3]), num_elem=2)

    torch.testing.assert_close(output, torch.stack([x[0, 1:3], x[1, 3:5]]))


def test_ts2vec_fit(ts2vec_model):
    data = np.random.randn(2, 8, 2).astype(np.float32)
    initial_weights = [p.detach().clone() for p in ts2vec_model._net.parameters()]

    losses = ts2vec_model.fit(data, n_epochs=1)

    assert len(losses) == 1
    assert np.isfinite(losses).all()
    assert ts2vec_model.n_epochs == 1
    assert ts2vec_model.n_iters == 1
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_weights, ts2vec_model._net.parameters())
    )


@pytest.mark.parametrize(
    "encoding_window, expected_shape",
    [
        (None, (2, 8, 4)),
        ("full_series", (2, 4)),
        (3, (2, 8, 4)),
        ("multiscale", (2, 8, 12)),
    ],
)
def test_ts2vec_encode(ts2vec_model, encoding_window, expected_shape):
    data = np.random.randn(2, 8, 2).astype(np.float32)

    output = ts2vec_model.encode(data, encoding_window=encoding_window)

    assert isinstance(output, np.ndarray)
    assert output.shape == expected_shape
    assert np.isfinite(output).all()
    assert ts2vec_model.net.training


@pytest.mark.parametrize("batch_size", [2, 4])
def test_ts2vec_encode_sliding(ts2vec_model, batch_size):
    data = np.random.randn(2, 8, 2).astype(np.float32)

    output = ts2vec_model.encode(
        data, sliding_length=2, sliding_padding=1, batch_size=batch_size
    )

    assert output.shape == (2, 8, 4)
    assert np.isfinite(output).all()


def test_ts2vec_save_load(ts2vec_model, tmp_path):
    data = np.random.randn(2, 8, 2).astype(np.float32)
    expected = ts2vec_model.encode(data)
    checkpoint = tmp_path / "ts2vec.pt"
    ts2vec_model.save(checkpoint)
    restored = TS2Vec(input_dims=2, output_dims=4, hidden_dims=4, depth=1, device="cpu")

    restored.load(checkpoint)

    np.testing.assert_allclose(restored.encode(data), expected)
