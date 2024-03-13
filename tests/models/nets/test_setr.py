import pytest
import torch

from sslt.models.nets.setr import SETR_PUP


def test_setr_pup_forward():
    # Create a dummy input
    x = torch.randn(1, 3, 512, 512)

    # Initialize the SETR_PUP model
    model = SETR_PUP()

    # Perform forward pass
    output = model(x)

    # Check if the output has the expected shape
    assert output.shape == (1, model.num_classes)


def test_setr_pup_loss_func():
    # Create dummy input and target tensors
    y_hat = torch.randn(1, 1000)
    y = torch.randint(0, 1000, (1,))

    # Initialize the SETR_PUP model
    model = SETR_PUP()

    # Calculate the loss
    loss = model._loss_func(y_hat, y)

    # Check if the loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_setr_pup_single_step():
    # Create dummy input and target tensors
    x = torch.randn(1, 3, 512, 512)
    y = torch.randint(0, 1000, (1,))

    # Initialize the SETR_PUP model
    model = SETR_PUP()

    # Perform a single training step
    loss = model._single_step((x, y), 0, "train")

    # Check if the loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_setr_pup_training_step():
    # Create dummy input and target tensors
    x = torch.randn(1, 3, 512, 512)
    y = torch.randint(0, 1000, (1,))

    # Initialize the SETR_PUP model
    model = SETR_PUP()

    # Perform a training step
    output = model.training_step((x, y), 0)

    # Check if the output is None
    assert output is None


def test_setr_pup_validation_step():
    # Create dummy input and target tensors
    x = torch.randn(1, 3, 512, 512)
    y = torch.randint(0, 1000, (1,))

    # Initialize the SETR_PUP model
    model = SETR_PUP()

    # Perform a validation step
    output = model.validation_step((x, y), 0)

    # Check if the output is None
    assert output is None


def test_setr_pup_test_step():
    # Create dummy input and target tensors
    x = torch.randn(1, 3, 512, 512)
    y = torch.randint(0, 1000, (1,))

    # Initialize the SETR_PUP model
    model = SETR_PUP()

    # Perform a test step
    output = model.test_step((x, y), 0)

    # Check if the output is None
    assert output is None


def test_setr_pup_predict_step():
    # Create a dummy input
    x = torch.randn(1, 3, 512, 512)

    # Initialize the SETR_PUP model
    model = SETR_PUP()

    # Perform a predict step
    output = model.predict_step((x, None), 0, 0)

    # Check if the output has the expected shape
    assert output.shape == (1, model.num_classes)


def test_setr_pup_configure_optimizers():
    # Initialize the SETR_PUP model
    model = SETR_PUP()

    # Configure optimizers
    optimizers = model.configure_optimizers()

    # Check if the optimizers are instances of torch.optim.Optimizer
    assert isinstance(optimizers, torch.optim.Optimizer)
