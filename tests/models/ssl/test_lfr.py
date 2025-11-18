import torch
from torch.nn import Sequential, Conv2d, CrossEntropyLoss
from torchvision.transforms import Resize

from minerva.models.ssl.lfr import RepeatedModuleList, LearnFromRandomnessModel
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone


def test_lfr():

    ## Example class for projector
    class Projector(Sequential):
        def __init__(self):
            super().__init__(
                Conv2d(3, 16, 5, 2),
                Conv2d(16, 64, 5, 2),
                Conv2d(64, 16, 5, 2),
                Resize((100, 50)),
            )

    ## Example class for predictor
    class Predictor(Sequential):
        def __init__(self):
            super().__init__(Conv2d(2048, 16, 1), Resize((100, 50)))

    # Declare model
    model = LearnFromRandomnessModel(
        DeepLabV3Backbone(),
        RepeatedModuleList(5, Projector),
        RepeatedModuleList(5, Predictor),
        CrossEntropyLoss(),
        flatten=False,
    )

    # Test the class instantiation
    assert model is not None

    # # Test the forward method
    input_shape = (2, 3, 701, 255)
    expected_output_size = torch.Size([2, 5, 16, 100, 50])
    x = torch.rand(*input_shape)

    y_pred, y_proj = model(x)
    assert (
        y_pred.shape == expected_output_size
    ), f"Expected output shape {expected_output_size}, but got {y_pred.shape}"

    assert (
        y_proj.shape == expected_output_size
    ), f"Expected output shape {expected_output_size}, but got {y_proj.shape}"

    # Test the loss_fn method
    loss = model.loss_fn(y_pred, y_proj)
    assert loss is not None
    # TODO: assert the loss result

    # Test the configure_optimizers method
    optimizer = model.configure_optimizers()
    assert optimizer is not None
