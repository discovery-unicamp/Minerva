import torch
from sslt.models.nets.wisenet import WiseNet


def test_wisenet_loss():
    model = WiseNet()
    batch_size = 2
    mask_shape = (batch_size, 1, 500, 500)  # (2, 1, 500, 500)
    input_shape = *mask_shape[:2], 17, *mask_shape[2:]  # (2, 1, 17, 500, 500)

    # Input X is volume of 17 slices of 500x500, 1 channel.
    # So, it is a 5D tensor of shape (B, C, D, H, W), where B is the batch
    # size, C is the number of channels, D is the depth, H is the
    # height and W is the width.
    x = torch.rand(*input_shape)
    # The mask is a single 2-D panel of 500x500, 1 channel.
    # It is a 4D tensor of shape (B, C, H, W), where B is the batch size,
    # C is the number of channels, H is the height and W is the width.
    mask = torch.rand(*mask_shape)

    # Do the training step
    loss = model.training_step((x, mask), 0).item()
    assert loss is not None
    assert loss >= 0, f"Expected non-negative loss, but got {loss}"


def test_wisenet_predict():
    model = WiseNet()
    batch_size = 2
    mask_shape = (batch_size, 1, 500, 500)  # (2, 1, 500, 500)
    input_shape = *mask_shape[:2], 17, *mask_shape[2:]  # (2, 1, 17, 500, 500)

    # Input X is volume of 17 slices of 500x500, 1 channel.
    # So, it is a 5D tensor of shape (B, C, D, H, W), where B is the batch
    # size, C is the number of channels, D is the depth, H is the
    # height and W is the width.
    x = torch.rand(*input_shape)
    # The mask is a single 2-D panel of 500x500, 1 channel.
    # It is a 4D tensor of shape (B, C, H, W), where B is the batch size,
    # C is the number of channels, H is the height and W is the width.
    mask = torch.rand(*mask_shape)

    # Do the prediction step
    preds = model.predict_step((x, mask), 0)
    assert preds is not None
    assert (
        preds.shape == mask_shape
    ), f"Expected shape {mask_shape}, but got {preds.shape}"
