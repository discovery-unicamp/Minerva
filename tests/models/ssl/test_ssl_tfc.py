from minerva.models.ssl.tfc import TFC_Model
from minerva.models.nets.tfc import TFC_Backbone, TFC_Conv_Block, TFC_Standard_Projector
from minerva.models.nets.tnc import TSEncoder
import torch
from torchmetrics import F1Score, Accuracy

def test_tfc_forward_default():
    model = TFC_Model(input_channels = 9, TS_length = 128, num_classes = 6, single_encoding_size = 128)
    assert model is not None, "Impossible to create TFC_Model with default input"
    x = torch.rand(42, *(9,128))
    y = model(x, x)
    assert y.shape == (42, 6), f"Expected shape (42, 6), got {y.shape}"


def test_tfc_forward_arbitrary():
    input_shape = (7, 227)
    single_encoding_size = 200
    batch_size = 37
    num_classes = 10

    model = TFC_Model(input_channels = input_shape[0], TS_length = input_shape[1], num_classes = num_classes, single_encoding_size = single_encoding_size, batch_size=batch_size)
    assert model is not None, "Impossible to create TFC_Model with arbitrary input"
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
    assert model is not None, "Impossible to create TFC_Model without prediction head"
    x = torch.rand(batch_size, * input_shape)
    h_time, z_time, h_freq, z_freq = model(x, x)
    assert len(h_time) == len(z_time) == len(h_freq) == len(z_freq) == batch_size, f"Expected shape ({batch_size}), got {len(h_time), len(z_time), len(h_freq), len(z_freq)}"
    assert z_time.shape[-1] + z_freq.shape[-1] == single_encoding_size*2, f"Expected shape {single_encoding_size*2}, got {z_time.shape[-1] + z_freq.shape[-1]}"

def test_tfc_only_time():
    input_shape = (2, 3)
    single_encoding_size = 4
    batch_size = 5
    num_classes = 6

    model = TFC_Model(input_channels = input_shape[0], TS_length = input_shape[1], num_classes = num_classes, single_encoding_size = single_encoding_size, pipeline= "time", batch_size=batch_size, pred_head = True)
    assert model is not None, "Impossible to create TFC_Model with only time pipeline"
    x = torch.rand(batch_size, * input_shape)
    y, h_time, z_time, h_freq, z_freq = model(x, x, all=True)
    assert y.shape == (batch_size, num_classes), f"Expected shape ({batch_size}, {num_classes}), got {y.shape}"
    assert len(h_time) == len(z_time) == batch_size, f"Expected shape ({batch_size}), got {len(h_time), len(z_time)}"

def test_tfc_only_freq():
    input_shape = (2, 3)
    single_encoding_size = 4
    batch_size = 5
    num_classes = 6

    model = TFC_Model(input_channels = input_shape[0], TS_length = input_shape[1], num_classes = num_classes, single_encoding_size = single_encoding_size, pipeline= "freq", batch_size=batch_size, pred_head = True)
    assert model is not None, "Impossible to create TFC_Model with only frequency pipeline"
    x = torch.rand(batch_size, * input_shape)
    y, h_time, z_time, h_freq, z_freq = model(x, x, all=True)
    assert y.shape == (batch_size, num_classes), f"Expected shape ({batch_size}, {num_classes}), got {y.shape}"
    assert len(h_time) == len(z_time) == batch_size, f"Expected shape ({batch_size}), got {len(h_time), len(z_time)}"
    

def test_tfc_invalid_passed_backbone():
    input_shape = (2, 3)
    single_encoding_size = 4
    batch_size = 5
    num_classes = 6

    conv_time = TFC_Conv_Block(input_channels = input_shape[0])
    conv_freq = TFC_Conv_Block(input_channels = input_shape[0])
    backbone = TFC_Backbone(input_channels = input_shape[0], TS_length = input_shape[1], single_encoding_size = single_encoding_size)
    try:
        model = TFC_Model(input_channels = input_shape[0], TS_length = input_shape[1], num_classes = num_classes, single_encoding_size = single_encoding_size, time_encoder = conv_time, frequency_encoder = conv_freq, batch_size=batch_size, pred_head = True, backbone=backbone)
    except:
        pass
    else:
        raise AssertionError("Model should raise an error when both backbone and encoders are passed")
    
def test_tfc_given_encoder():
    input_shape = (2, 3)
    single_encoding_size = 4
    batch_size = 5
    num_classes = 6

    time_encoder = TFC_Conv_Block(input_channels = input_shape[0])
    frequency_encoder = TFC_Conv_Block(input_channels = input_shape[0])

    model = TFC_Model(input_channels = input_shape[0], TS_length = input_shape[1], num_classes = num_classes, single_encoding_size = single_encoding_size, time_encoder = time_encoder, frequency_encoder = frequency_encoder, batch_size=batch_size, pred_head = True)
    assert model is not None, "Impossible to create TFC_Model with given conv_backbone"
    x = torch.rand(batch_size, * input_shape)
    y, h_time, z_time, h_freq, z_freq = model(x, x, all=True)
    assert y.shape == (batch_size, num_classes), f"Expected shape ({batch_size}, {num_classes}), got {y.shape}"
    assert len(h_time) == len(z_time) == len(h_freq) == len(z_freq) == batch_size, f"Expected shape ({batch_size}), got {len(h_time), len(z_time), len(h_freq), len(z_freq)}"

def test_tfc_given_projector():
    input_shape = (2, 3)
    single_encoding_size = 4
    batch_size = 5
    num_classes = 6

    time_projector = TFC_Standard_Projector(input_channels = 180,single_encoding_size = single_encoding_size)
    frequency_projector = TFC_Standard_Projector(input_channels = 180, single_encoding_size = single_encoding_size)

    model = TFC_Model(input_channels = input_shape[0], TS_length = input_shape[1], num_classes = num_classes, single_encoding_size = single_encoding_size, time_projector = time_projector, frequency_projector = frequency_projector, batch_size=batch_size, pred_head = True)
    assert model is not None, "Impossible to create TFC_Model with given projectors"
    x = torch.rand(batch_size, * input_shape)
    y, h_time, z_time, h_freq, z_freq = model(x, x, all=True)
    assert y.shape == (batch_size, num_classes), f"Expected shape ({batch_size}, {num_classes}), got {y.shape}"
    assert len(h_time) == len(z_time) == len(h_freq) == len(z_freq) == batch_size, f"Expected shape ({batch_size}), got {len(h_time), len(z_time), len(h_freq), len(z_freq)}"

def test_tfc_given_ts2vec_encoder():
    input_shape = (2, 3)
    single_encoding_size = 4
    batch_size = 5
    num_classes = 6

    time_encoder = TSEncoder(input_dims=input_shape[0], output_dims=128,permute=True)
    frequency_encoder = TSEncoder(input_dims=input_shape[0], output_dims=128,permute=True)

    model = TFC_Model(input_channels = input_shape[0], TS_length = input_shape[1], num_classes = num_classes, single_encoding_size = single_encoding_size, time_encoder = time_encoder, frequency_encoder = frequency_encoder, batch_size=batch_size, pred_head = True)
    assert model is not None, "Impossible to create TFC_Model with given conv_backbone"
    x = torch.rand(batch_size, * input_shape)
    y, h_time, z_time, h_freq, z_freq = model(x, x, all=True)
    assert y.shape == (batch_size, num_classes), f"Expected shape ({batch_size}, {num_classes}), got {y.shape}"
    assert len(h_time) == len(z_time) == len(h_freq) == len(z_freq) == batch_size, f"Expected shape ({batch_size}), got {len(h_time), len(z_time), len(h_freq), len(z_freq)}"

def test_tfc_metrics_argument():
    num_classes = 6
    train_metrics = {"f1": F1Score(task="multiclass", num_classes=num_classes), "accuracy": Accuracy(task="multiclass", num_classes=num_classes)}
    val_metrics = {"f1": F1Score(task="multiclass", num_classes=num_classes), "accuracy": Accuracy(task="multiclass", num_classes=num_classes)}
    test_metrics = {"f1": F1Score(task="multiclass", num_classes=num_classes), "accuracy": Accuracy(task="multiclass", num_classes=num_classes)}

    model = TFC_Model(input_channels = 9, TS_length = 128, num_classes = num_classes, single_encoding_size = 128, train_metrics = train_metrics, val_metrics = val_metrics, test_metrics = test_metrics)
    assert model is not None, "Impossible to create TFC_Model with metrics"

