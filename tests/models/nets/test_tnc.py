import torch
from minerva.models.ssl.tnc import TNC
from minerva.models.nets.tnc import RnnEncoder, TSEncoder, Discriminator_TNC

def test_rnn_encoder_discriminator():
    backbone = RnnEncoder(hidden_size=100, in_channel=6, encoding_size=320, cell_type='GRU', num_layers=1)
    input_shape = (2, 128, 6) #Bs x timesteps x channels
    expected_output_shape = torch.Size([2, 320]) #bs x encoding size

    # Create random input data torch.Size([2, 128, 6]),
    x = torch.rand(*input_shape)

    # Output forward method
    output_x = backbone.forward(x)
    
    print(expected_output_shape, output_x.shape)

    assert (
        output_x.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output_x.shape}"

    # now test this output with the discriminator
    # we will need a second input tensor
    x_d = torch.rand(*input_shape)
    output_x_d = backbone.forward(x_d)
    projection_head = Discriminator_TNC(input_size=320,max_pool=False)
    
    output_discriminator = projection_head(output_x,output_x_d)
    expected_output_shape_discriminator = torch.Size([2]) #bs
    print(expected_output_shape_discriminator, output_discriminator.shape)

    assert (
        output_discriminator.shape == expected_output_shape_discriminator
    ), f"Expected output shape {expected_output_shape_discriminator}, but got {output_discriminator.shape}"

    


def test_ts2vec_encoder_discriminator():
    backbone = TSEncoder(input_dims=6, output_dims=320, hidden_dims=64, depth=10)
    input_shape = (2, 128, 6) #Bs x timesteps x channels
    expected_output_shape = torch.Size([2,128, 320]) #bs x encoding size

    # Create random input data torch.Size([2, 128, 6]),
    x = torch.rand(*input_shape)

    # Output forward method
    output_x = backbone.forward(x)
    
    print(expected_output_shape, output_x.shape)

    assert (
        output_x.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output_x.shape}"

    # now test this output with the discriminator
    # we will need a second input tensor
    x_d = torch.rand(*input_shape)
    output_x_d = backbone.forward(x_d)
    projection_head = Discriminator_TNC(input_size=320,max_pool=True)
    
    output_discriminator = projection_head(output_x,output_x_d)
    expected_output_shape_discriminator = torch.Size([2]) #bs
    print(expected_output_shape_discriminator, output_discriminator.shape)

    assert (
        output_discriminator.shape == expected_output_shape_discriminator
    ), f"Expected output shape {expected_output_shape_discriminator}, but got {output_discriminator.shape}"


def test_tnc_model_rnn():
    # first define backbone, projection head and pretext model
    backbone = RnnEncoder(hidden_size=100, in_channel=6, encoding_size=320, cell_type='GRU', num_layers=1)
    projection_head = Discriminator_TNC(input_size=320,max_pool=False)
    pretext_model = TNC(backbone=backbone, projection_head=projection_head)
    
    # define an input tensor shape that will come from a datamodule
    input_shape = (2, 128, 6) #Bs x timesteps x channels
    expected_output_shape = torch.Size([10]) #bs x mc_sample size

    # Create random input data torch.Size([2, 128, 6]),
    x = torch.rand(*input_shape)

    # Create random input windows based on sample size # torch.Size([2, 5, 128, 6])
    input_shape_dc = (2, 5,128, 6) 
    x_d = torch.rand(*input_shape_dc)
    x_c = torch.rand(*input_shape_dc)

    # Output forward method
    output = pretext_model.forward(x,x_d,x_c)

    # note that TNC will produce a tensor with more points than original vector
    # in regards to the batch size. if an input tensor has shape (2, 128, 6) and the parameter of sample size of 5
    #  then the output will have shape (10), which is 2 multiplied by 5
    if isinstance(output, tuple):
        output = output[0]
        print(expected_output_shape, output.shape)

    assert (
        output.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output.shape}"


def test_tnc_model_ts2vec():
    # first define backbone, projection head and pretext model
    backbone = TSEncoder(input_dims=6, output_dims=320, hidden_dims=64, depth=10)
    projection_head = Discriminator_TNC(input_size=320,max_pool=True)
    pretext_model = TNC(backbone=backbone, projection_head=projection_head)
    
    # define an input tensor shape that will come from a datamodule
    input_shape = (2, 128, 6) #Bs x timesteps x channels
    expected_output_shape = torch.Size([10]) #bs x mc_sample size

    # Create random input data torch.Size([2, 128, 6]),
    x = torch.rand(*input_shape)

    # Create random input windows based on sample size # torch.Size([2, 5, 128, 6])
    input_shape_dc = (2, 5,128, 6) 
    x_d = torch.rand(*input_shape_dc)
    x_c = torch.rand(*input_shape_dc)

    # Output forward method
    output = pretext_model.forward(x,x_d,x_c)

    # note that TNC will produce a tensor with more points than original vector
    # in regards to the batch size. if an input tensor has shape (2, 128, 6) and the parameter of sample size of 5
    #  then the output will have shape (10), which is 2 multiplied by 5
    if isinstance(output, tuple):
        output = output[0]
        print(expected_output_shape, output.shape)

    assert (
        output.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output.shape}"


def test_rnn_downstream():
    # first define backbone and prediction head
    backbone = RnnEncoder(hidden_size=100, in_channel=6, encoding_size=320, cell_type='GRU', num_layers=1)
       
    # define an input tensor shape that will come from a datamodule
    input_shape = (2, 128, 6) #Bs x timesteps x channels
    expected_output_shape = torch.Size([2, 320]) #bs x encoding size

    # Create random input data torch.Size([2, 128, 6]),
    x = torch.rand(*input_shape)

    # Output forward method
    output = backbone.forward(x)

    print(expected_output_shape, output.shape)

    assert (
        output.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output.shape}"

    # now test this output with the prediction_head   
    prediction_head = torch.nn.Sequential(
    torch.nn.Linear(320, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 6)
    )
    output_prediction_head = prediction_head(output)
    expected_output_shape_discriminator = torch.Size([2,6]) #bs x features
    print(expected_output_shape_discriminator, output_prediction_head.shape)

    assert (
        output_prediction_head.shape == expected_output_shape_discriminator
    ), f"Expected output shape {expected_output_shape_discriminator}, but got {output_prediction_head.shape}"

def test_ts2vec_downstream():
    # first define backbone and prediction head
    backbone = TSEncoder(input_dims=6, output_dims=320, hidden_dims=64, depth=10)
       
    # define an input tensor shape that will come from a datamodule
    input_shape = (2, 128, 6) #Bs x timesteps x channels
    expected_output_shape = torch.Size([2,128, 320]) #bs x encoding size

    # Create random input data torch.Size([2, 128, 6]),
    x = torch.rand(*input_shape)

    # Output forward method
    output = backbone.forward(x)

    print(expected_output_shape, output.shape)

    assert (
        output.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output.shape}"

    # now test this output with the prediction_head   
    prediction_head = torch.nn.Sequential(
    torch.nn.Linear(320, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 6)
    )

    output_prediction_head = prediction_head(output)
    expected_output_shape_discriminator = torch.Size([2,128,6]) 
    #bs x timesteps x features. that is why there is an incompatible shape using directly the encoder
    print(expected_output_shape_discriminator, output_prediction_head.shape)

    assert (
        output_prediction_head.shape == expected_output_shape_discriminator
    ), f"Expected output shape {expected_output_shape_discriminator}, but got {output_prediction_head.shape}"

    # when the adapter is merged this can be uncommented and the lines above removed
    # before passing the output to the prediction head, we need to apply an adapter to convert the output to a 2D tensor
    # adapter = MaxPoolingTransposingSqueezingAdapter(kernel_size=128)
    # output = adapter(output)

    # output_prediction_head = prediction_head(output)
    # expected_output_shape_discriminator = torch.Size([2,6]) #bs x features
    # print(expected_output_shape_discriminator, output_prediction_head.shape)

    # assert (
    #     output_prediction_head.shape == expected_output_shape_discriminator
    # ), f"Expected output shape {expected_output_shape_discriminator}, but got {output_prediction_head.shape}"
