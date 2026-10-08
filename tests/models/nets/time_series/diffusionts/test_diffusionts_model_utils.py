import pytest
import torch

from minerva.models.nets.time_series.diffusionts.diffusionts_model_utils import (
    LearnablePositionalEncoding,
    extract,
    normalize_to_neg_one_to_one,
    series_decomp,
)


@pytest.mark.parametrize("sequence_length", [4, 8])
def test_learnable_positional_encoding_matches_sequence_length(sequence_length):
    model = LearnablePositionalEncoding(d_model=3, dropout=0, max_len=8)
    x = torch.zeros(2, sequence_length, 3)

    output = model(x)

    assert output.shape == x.shape
    torch.testing.assert_close(output, model.pe[:, :sequence_length].expand_as(x))


def test_series_decomposition_reconstructs_input():
    model = series_decomp(kernel_size=3)
    x = torch.rand(2, 16, 2)

    residual, trend = model(x)

    torch.testing.assert_close(residual + trend, x)


def test_extract_selects_one_value_per_sample():
    values = torch.tensor([0.1, 0.2, 0.3])
    timesteps = torch.tensor([2, 0])

    result = extract(values, timesteps, x_shape=(2, 16, 2))

    torch.testing.assert_close(result, torch.tensor([[[0.3]], [[0.1]]]))


def test_normalization_maps_zero_and_one_to_minus_one_and_one():
    x = torch.tensor([0.0, 0.5, 1.0])

    result = normalize_to_neg_one_to_one(x)

    torch.testing.assert_close(result, torch.tensor([-1.0, 0.0, 1.0]))


def test_series_decomposition_keeps_constant_signal_as_trend():
    model = series_decomp(kernel_size=3)
    x = torch.ones(2, 16, 2)

    residual, trend = model(x)

    torch.testing.assert_close(trend, x)
    torch.testing.assert_close(residual, torch.zeros_like(x))
