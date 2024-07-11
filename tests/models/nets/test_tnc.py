import torch
import sys
from minerva.models.ssl.tnc import TNC
from minerva.models.nets.tnc import RnnEncoder, DilatedConvEncoder, Discriminator_TNC

def test_tnc_model():
    backbone = RnnEncoder(hidden_size=100, in_channel=6, encoding_size=320, cell_type='GRU', num_layers=1)
    projection_head = Discriminator_TNC(input_size=320,max_pool=False)
    pretext_model = TNC(backbone=backbone, projection_head=projection_head)
    
    
    input_shape = (2, 128, 6) #Bs x timesteps x channels
    expected_output_shape = torch.Size([10]) #bs x mc_sample size

    # Create random input data torch.Size([2, 128, 6]),
    x = torch.rand(*input_shape)

    # Create random input windows based on sample size # torch.Size([2, 5, 128, 6])
    input_shape_dc = (2, 5,128, 6) 
    x_d = torch.rand(*input_shape_dc)
    x_c = torch.rand(*input_shape_dc)

    # Output forward method: return z,y 
    output = pretext_model.forward(x,x_d,x_c)
    
    print(expected_output_shape)
    if isinstance(output, tuple):
        output = output[0]
        # print(output.shape)

    assert (
        output.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output.shape}"

    #TODO add for TS2Vec encoder
    


    