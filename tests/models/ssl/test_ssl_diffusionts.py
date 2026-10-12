from copy import deepcopy

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

from minerva.models.nets.time_series.diffusionts.diffusionts_model_utils import cond_fn
from minerva.models.nets.time_series.diffusionts.diffusionts_transformer import (
    Transformer,
)
from minerva.models.ssl.diffusionts import DiffusionTS


@pytest.fixture
def small_transformer():
    return Transformer(
        n_feat=2,
        n_channel=16,
        n_layer_enc=1,
        n_layer_dec=1,
        n_embd=8,
        n_heads=2,
        max_len=16,
        attn_pdrop=0,
        resid_pdrop=0,
    )


@pytest.fixture
def diffusionts(small_transformer):
    return DiffusionTS(small_transformer, seq_length=16, feature_size=2, timesteps=4)


@pytest.fixture
def trainer_options():
    return {
        "accelerator": "cpu",
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
    }


def test_diffusionts_training_loss(diffusionts):
    x = torch.rand(2, 16, 2)

    loss = diffusionts(x)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_diffusionts_noise_can_be_removed(diffusionts):
    x = torch.rand(2, 16, 2)
    noise = torch.randn_like(x)
    timesteps = torch.tensor([0, 1])
    noisy = diffusionts.q_sample(x, timesteps, noise)

    restored = diffusionts.predict_start_from_noise(noisy, timesteps, noise)

    torch.testing.assert_close(restored, x)


def test_diffusionts_invalid_channel_count(diffusionts):
    x = torch.rand(2, 16, 3)

    with pytest.raises(AssertionError, match="number of variable"):
        diffusionts(x)


@pytest.mark.parametrize("sampling_steps", [2, 4])
def test_diffusionts_generate_samples(small_transformer, sampling_steps):
    model = DiffusionTS(
        small_transformer,
        seq_length=16,
        feature_size=2,
        timesteps=4,
        sampling_timesteps=sampling_steps,
    )

    samples = model.generate_mts(batch_size=2)

    assert samples.shape == (2, 16, 2)
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("sampling_steps", [2, 4])
def test_diffusionts_conditioned_sampling_accepts_optional_kwargs(
    small_transformer, sampling_steps
):
    model = DiffusionTS(
        small_transformer,
        seq_length=16,
        feature_size=2,
        timesteps=4,
        sampling_timesteps=sampling_steps,
    ).eval()

    def no_condition(x, t):
        return torch.zeros_like(x)

    torch.manual_seed(42)
    omitted = model.generate_mts(batch_size=2, cond_fn=no_condition)
    torch.manual_seed(42)
    explicit = model.generate_mts(batch_size=2, model_kwargs={}, cond_fn=no_condition)

    torch.testing.assert_close(omitted, explicit)


@pytest.mark.parametrize("sampling_steps", [2, 4])
def test_diffusionts_conditioned_sampling_keeps_local_gradients(
    small_transformer, sampling_steps
):
    model = DiffusionTS(
        small_transformer,
        seq_length=16,
        feature_size=2,
        timesteps=4,
        sampling_timesteps=sampling_steps,
    ).eval()

    def classifier(x, t):
        return x.mean(dim=1)

    samples = model.generate_mts(
        batch_size=2,
        cond_fn=cond_fn,
        model_kwargs={"classifier": classifier, "y": torch.tensor([0, 1])},
    )

    assert not samples.requires_grad
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("method", ["fast_sample_infill", "sample_infill"])
def test_diffusionts_infill_refines_samples_without_retaining_graph(
    small_transformer, method
):
    model = DiffusionTS(
        small_transformer, seq_length=16, feature_size=2, timesteps=20
    ).eval()
    target = torch.rand(2, 16, 2)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:, :8] = True
    kwargs = {"sampling_timesteps": 4} if method == "fast_sample_infill" else {}

    samples = getattr(model, method)(
        shape=target.shape,
        target=target,
        partial_mask=mask,
        model_kwargs={"coef": 0.01, "learning_rate": 0.01},
        **kwargs,
    )

    assert not samples.requires_grad
    assert torch.isfinite(samples).all()
    torch.testing.assert_close(samples[mask], target[mask])


@pytest.mark.parametrize("method", ["fast_sample_infill", "sample_infill"])
def test_diffusionts_infill_requires_partial_mask(diffusionts, method):
    target = torch.zeros(2, 16, 2)
    kwargs = {"shape": target.shape, "target": target}
    if method == "fast_sample_infill":
        kwargs["sampling_timesteps"] = 2

    with pytest.raises(ValueError, match="partial_mask"):
        getattr(diffusionts, method)(**kwargs)


@pytest.mark.parametrize("num_batches", [1, 2, 4])
def test_diffusionts_accumulates_two_batches_per_step(
    diffusionts, trainer_options, num_batches
):
    batches = DataLoader(torch.rand(num_batches * 2, 16, 2), batch_size=2)
    trainer = L.Trainer(max_epochs=1, **trainer_options)

    trainer.fit(diffusionts, train_dataloaders=batches)

    assert trainer.global_step == diffusionts.step_counter == num_batches // 2


def test_diffusionts_fp32_matches_original_updates(diffusionts, trainer_options):
    original = deepcopy(diffusionts)
    original.configure_optimizers()
    data = torch.rand(8, 16, 2)
    batches = DataLoader(data, batch_size=2, generator=torch.Generator())

    torch.manual_seed(42)
    total_loss = 0
    for index, batch in enumerate(data.split(2)):
        loss = original(batch, target=batch) / original.gradient_accumulate_every
        loss.backward()
        total_loss += loss.item()
        if (index + 1) % original.gradient_accumulate_every == 0:
            torch.nn.utils.clip_grad_norm_(original.parameters(), 1.0)
            original.opt.step()
            original.sch.step(total_loss)
            original.opt.zero_grad()
            original.ema.update()
            total_loss = 0

    torch.manual_seed(42)
    trainer = L.Trainer(max_epochs=1, **trainer_options)
    trainer.fit(diffusionts, train_dataloaders=batches)

    torch.testing.assert_close(diffusionts.state_dict(), original.state_dict())


def test_diffusionts_checkpoint_contains_optimizer_and_scheduler(
    diffusionts, trainer_options, tmp_path
):
    batches = DataLoader(torch.rand(4, 16, 2), batch_size=2)
    trainer = L.Trainer(max_epochs=1, **trainer_options)
    trainer.fit(diffusionts, train_dataloaders=batches)

    path = tmp_path / "diffusionts.ckpt"
    trainer.save_checkpoint(path)
    checkpoint = torch.load(path, weights_only=False)

    torch.testing.assert_close(
        checkpoint["optimizer_states"], [diffusionts.opt.state_dict()]
    )
    assert checkpoint["lr_schedulers"] == [diffusionts.sch.state_dict()]


@pytest.mark.parametrize("num_batches", [2, 3])
def test_diffusionts_resume_matches_uninterrupted_training(
    diffusionts, trainer_options, tmp_path, num_batches
):
    diffusionts.scheduler_hparams = {"warmup": 4, "warmup_lr": 3e-5}
    continuous = deepcopy(diffusionts)
    interrupted = deepcopy(diffusionts)
    data = torch.rand(num_batches * 2, 16, 2)
    batches = DataLoader(data, batch_size=2, generator=torch.Generator())

    torch.manual_seed(42)
    full_trainer = L.Trainer(max_epochs=2, **trainer_options)
    full_trainer.fit(continuous, train_dataloaders=batches)

    torch.manual_seed(42)
    first_trainer = L.Trainer(max_epochs=1, **trainer_options)
    first_trainer.fit(interrupted, train_dataloaders=batches)
    path = tmp_path / "diffusionts.ckpt"
    first_trainer.save_checkpoint(path)

    resumed_trainer = L.Trainer(max_epochs=2, **trainer_options)
    resumed_trainer.fit(diffusionts, train_dataloaders=batches, ckpt_path=path)

    torch.testing.assert_close(diffusionts.state_dict(), continuous.state_dict())
    assert diffusionts.step_counter == continuous.step_counter


def test_diffusionts_loads_legacy_weights_strictly(diffusionts, small_transformer):
    diffusionts.configure_optimizers()
    legacy_checkpoint = {"state_dict": diffusionts.state_dict()}
    restored = DiffusionTS(
        deepcopy(small_transformer), seq_length=16, feature_size=2, timesteps=4
    )
    restored.configure_model()

    restored.on_load_checkpoint(legacy_checkpoint)
    restored.load_state_dict(legacy_checkpoint["state_dict"], strict=True)

    torch.testing.assert_close(restored.state_dict(), legacy_checkpoint["state_dict"])
