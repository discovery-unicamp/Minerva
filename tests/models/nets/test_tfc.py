from minerva.models.nets.tfc import TFC_Backbone, TFC_PredicionHead
import torch

def test_tfc_backbone_forward():
    input_shape = (7, 227)
    encoding_size = 256
    batch_size = 37

    model = TFC_Backbone(input_channels=input_shape[0], TS_length=input_shape[1], encoding_size=encoding_size)
    assert model is not None
    x = torch.rand(batch_size, *input_shape)
    y = model(x, x)
    assert len(y) == 4
    h_time, z_time, h_freq, z_freq = y 
    assert len(h_time) == len(z_time) == len(h_freq) == len(z_freq) == batch_size
    assert z_time.shape[-1] + z_freq.shape[-1] == encoding_size

test_tfc_backbone_forward()