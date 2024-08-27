from minerva.models.ssl.tfc import TFC_Model
import torch

def test_tfc_forward_default():
    model = TFC_Model(input_channels = 9, TS_length = 128, num_classes = 6, single_encoding_size = 128)
    assert model is not None, "Impossible to create TFC_Model"
    x = torch.rand(42, *(9,128))
    y = model(x, x)
    assert y.shape == (42, 6), f"Expected shape (42, 6), got {y.shape}"


def test_tfc_forward_arbitrary():
    input_shape = (7, 227)
    single_encoding_size = 200
    batch_size = 37
    num_classes = 10

    model = TFC_Model(input_channels = input_shape[0], TS_length = input_shape[1], num_classes = num_classes, single_encoding_size = single_encoding_size, batch_size=batch_size)
    assert model is not None, "Impossible to create TFC_Model"
    x = torch.rand(batch_size, *input_shape)
    y, h_time, z_time, h_freq, z_freq = model(x, x, all=True)
    assert y.shape == (batch_size, num_classes), f"Expected shape ({batch_size}, {num_classes}), got {y.shape}"

    assert len(h_time) == len(z_time) == len(h_freq) == len(z_freq) == batch_size, f"Expected shape ({batch_size}), got {len(h_time), len(z_time), len(h_freq), len(z_freq)}"
    assert z_time.shape[-1] + z_freq.shape[-1] == single_encoding_size*2, f"Expected shape {single_encoding_size*2}, got {z_time.shape[-1] + z_freq.shape[-1]}"

def test_tfc_forward_without_head():
    input_shape = (2, 3)
    single_encoding_size = 4
    batch_size = 5
    num_classes = 6

    model = TFC_Model(input_channels = input_shape[0], TS_length = input_shape[1], num_classes = num_classes, single_encoding_size = single_encoding_size, pred_head = False, batch_size=batch_size)
    assert model is not None, "Impossible to create TFC_Model"
    x = torch.rand(batch_size, * input_shape)
    h_time, z_time, h_freq, z_freq = model(x, x)
    assert len(h_time) == len(z_time) == len(h_freq) == len(z_freq) == batch_size, f"Expected shape ({batch_size}), got {len(h_time), len(z_time), len(h_freq), len(z_freq)}"
    assert z_time.shape[-1] + z_freq.shape[-1] == single_encoding_size*2, f"Expected shape {single_encoding_size*2}, got {z_time.shape[-1] + z_freq.shape[-1]}"

def test_tfc_given_conv_backbone():
    raise NotImplementedError

def test_tfc_invalid_passed_backbone():
    # "If a backbone is provided, the encoders and projectors must be None"
    raise NotImplementedError

def test_tfc_given_encoder():
    raise NotImplementedError

def test_tfc_given_projector():
    raise NotImplementedError

def test_tfc_given_ts2vec_encoder():
    raise NotImplementedError
