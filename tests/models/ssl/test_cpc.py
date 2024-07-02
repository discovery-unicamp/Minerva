import torch
from minerva.models.ssl.cpc import CPC
from minerva.models.nets.cpc_networks import CNN, HARCPCAutoregressive, PredictionNetwork, Genc_Gar, LinearClassifier, HARPredictionHead
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer

def test_cpc_model():
    # Instantiate encoder, autoregressive, and prediction head
    g_enc = CNN()  # Supondo que você tenha uma classe CNN definida
    g_ar = HARCPCAutoregressive()
    prediction_head = PredictionNetwork()

    # Test the class instantiation
    model = CPC(g_enc=g_enc, g_ar=g_ar, prediction_head=prediction_head)
    assert model is not None, "Model instantiation failed."

    # Generate a random input tensor (B, C, T) and the expected output shape
    input_shape = (1, 6, 60)  # Exemplo de forma de entrada
    expected_output_shape_z = (1, 60, 128)  # Exemplo de forma esperada de saída

    # Create random input data
    x = torch.rand(*input_shape)

    # Test the forward method
    # Output forward method: return z,y 
    output = model.forward(x)
    if isinstance(output, tuple):
        output = output[0]
    
    assert (
        output.shape == expected_output_shape_z
    ), f"Expected output shape {expected_output_shape_z}, but got {output.shape}"
    
    # Test the training step
    loss = model.training_step((x, None), 0).item()
    assert loss is not None, "Loss is None."
    assert loss >= 0, f"Expected non-negative loss, but got {loss}"

if __name__ == "__main__":
    test_cpc_model()
