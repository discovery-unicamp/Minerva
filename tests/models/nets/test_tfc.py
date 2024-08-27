from minerva.models.nets.tfc import TFC_Backbone, TFC_PredicionHead
import torch

def test_tfc_backbone_forward_default():
    input_shape = (9, 128)
    single_encoding_size = 128
    batch_size = 42

    model = TFC_Backbone(input_channels=input_shape[0], TS_length=input_shape[1], single_encoding_size=single_encoding_size)
    assert model is not None
    x = torch.rand(batch_size, *input_shape)
    y = model(x, x)
    assert len(y) == 4
    h_time, z_time, h_freq, z_freq = y 
    assert len(h_time) == len(z_time) == len(h_freq) == len(z_freq) == batch_size
    assert z_time.shape[-1] + z_freq.shape[-1] == single_encoding_size*2

def test_tfc_backbone_forward_arbitrary():
    input_shape = (7, 227)
    single_encoding_size = 128
    batch_size = 37

    model = TFC_Backbone(input_channels=input_shape[0], TS_length=input_shape[1], single_encoding_size=single_encoding_size)
    assert model is not None
    x = torch.rand(batch_size, *input_shape)
    y = model(x, x)
    assert len(y) == 4
    h_time, z_time, h_freq, z_freq = y 
    assert len(h_time) == len(z_time) == len(h_freq) == len(z_freq) == batch_size
    assert z_time.shape[-1] + z_freq.shape[-1] == single_encoding_size*2

def test_tfc_prediction_head():
    single_encoding_size = 4
    batch_size = 5
    num_classes = 10

    model = TFC_PredicionHead(num_classes=num_classes, single_encoding_size=single_encoding_size)
    assert model is not None
    x = torch.rand(batch_size, single_encoding_size*2)
    y = model(x)
    assert y.shape == (batch_size, num_classes)


test_tfc_backbone_forward_arbitrary()
test_tfc_backbone_forward_default()
test_tfc_prediction_head()
