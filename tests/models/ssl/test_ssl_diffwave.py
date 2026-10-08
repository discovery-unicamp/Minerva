import pytest
import torch

from minerva.models.ssl.diffwave import DiffWave


@pytest.fixture
def small_diffwave():
    return DiffWave(
        in_channels=2,
        out_channels=2,
        res_channels=8,
        skip_channels=4,
        num_res_layers=2,
        dilation_cycle=2,
        diffusion_step_embed_dim_in=8,
        diffusion_step_embed_dim_mid=16,
        diffusion_step_embed_dim_out=8,
        T=4,
    )


def test_diffwave_forward(small_diffwave):
    x = torch.rand(2, 2, 16)
    timesteps = torch.tensor([[0], [1]])

    output = small_diffwave((x, timesteps))

    assert output.shape == x.shape


def test_diffwave_training_loss(small_diffwave):
    x = torch.rand(2, 2, 16)

    loss = small_diffwave.training_loss(torch.nn.MSELoss(), x)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_diffwave_training_loss_backward(small_diffwave):
    x = torch.rand(2, 2, 16)
    loss = small_diffwave.training_loss(torch.nn.MSELoss(), x)

    loss.backward()

    assert small_diffwave.final_conv[-1].conv.weight.grad is not None


def test_diffwave_optimizer_learning_rate(small_diffwave):
    optimizer = small_diffwave.configure_optimizers()[0]

    assert optimizer.param_groups[0]["lr"] == small_diffwave.learning_rate


def test_diffwave_sample(small_diffwave):
    samples = small_diffwave.sample(batch_size=2, segment_length=16)

    assert samples.shape == (2, 2, 16)
    assert torch.isfinite(samples).all()
