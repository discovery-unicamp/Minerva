import torch
import sys
from minerva.models.ssl.tnc import TNC
from minerva.models.nets.tnc import RnnEncoder, DilatedConvEncoder, Discriminator_TNC

def test_tnc_model():
    backbone = RnnEncoder(hidden_size=100, in_channel=6, encoding_size=320, cell_type='GRU', num_layers=1)
    projection_head = Discriminator_TNC(input_size=320,max_pool=False)
    pretext_model = TNC(backbone=backbone, projection_head=projection_head)
    input_shape = (2, 128, 6) #Bs x timesteps x channels
    # input_shape = (1, 6,128) 
    expected_output_shape = (10, 320) #bs x mc_sample size, encodings


    # shape of x_t, X_close, X_distant: 
    # 
    # torch.Size([16, 5, 128, 6]),

    # Create random input data torch.Size([2, 128, 6]),
    x = torch.rand(*input_shape)

    # Create random input windows based on sample size # torch.Size([2, 5, 128, 6])
    input_shape_dc = (2, 5,128, 6) 
    x_d = torch.rand(*input_shape_dc)
    x_c = torch.rand(*input_shape_dc)


    # Output forward method: return z,y 
    output = pretext_model.forward(x,x_d,x_c)

    # Test the forward method
    # print(output[0].shape)

    if isinstance(output, tuple):
        output = output[0]

    assert (
        output.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output.shape}"

    #TODO add for TS2Vec encoder



    