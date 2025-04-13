import math
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from minerva.models.nets.base import SimpleSupervisedModel


# --- Dummy Components ---
class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU())

    def forward(self, x):
        return self.model(x)


class DummyFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def dummy_dataset():
    x = torch.randn(100, 1, 28, 28)
    y = torch.randint(0, 10, (100,))
    return DataLoader(TensorDataset(x, y), batch_size=32)


@pytest.fixture
def model():
    return SimpleSupervisedModel(
        backbone=DummyBackbone(),
        fc=DummyFC(),
        loss_fn=nn.CrossEntropyLoss(),
        train_metrics={"acc": Accuracy(task="multiclass", num_classes=10)},
        val_metrics={"acc": Accuracy(task="multiclass", num_classes=10)},
        test_metrics={"acc": Accuracy(task="multiclass", num_classes=10)},
        learning_rate=1e-3,
        optimizer=torch.optim.Adam,
        flatten=True,
    )


# --- Forward Pass ---
def test_forward_pass(model):
    x = torch.randn(4, 1, 28, 28)
    output = model(x)
    assert output.shape == (4, 10)
    assert isinstance(output, torch.Tensor)


# --- Loss Function ---
def test_loss_computation(model):
    y_hat = torch.randn(8, 10, requires_grad=True)
    y = torch.randint(0, 10, (8,))
    loss = model._loss_func(y_hat, y)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


# --- Metrics Computation ---
def test_metric_computation(model):
    y_hat = torch.randn(16, 10)
    y = torch.randint(0, 10, (16,))
    metrics = model._compute_metrics(y_hat, y, step_name="train")
    assert "train_acc" in metrics
    assert isinstance(metrics["train_acc"], torch.Tensor)


# --- Optimizer Config ---
def test_optimizer_configuration(model):
    opt = model.configure_optimizers()
    if isinstance(opt, tuple):
        optim, sched = opt
        assert isinstance(optim[0], torch.optim.Optimizer)
    else:
        assert isinstance(opt, torch.optim.Optimizer)


# --- Training/Validation/Test Steps ---
def test_train_step(model):
    batch = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))
    loss = model.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)


def test_val_step(model):
    batch = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))
    loss = model.validation_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)


def test_test_step(model):
    batch = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))
    loss = model.test_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)


def test_predict_step(model):
    batch = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))
    preds = model.predict_step(batch, batch_idx=0)
    assert preds.shape == (4, 10)


# --- Trainer Integration ---
def test_model_trains_with_trainer(model, dummy_dataset):
    trainer = Trainer(
        accelerator="cpu",
        max_epochs=1,
        fast_dev_run=True,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, dummy_dataset, dummy_dataset)


# --- Backbone Freezing ---
def test_unfreeze_backbone_flag():
    model = SimpleSupervisedModel(
        backbone=DummyBackbone(),
        fc=DummyFC(),
        loss_fn=nn.CrossEntropyLoss(),
        freeze_backbone=False,
    )
    model.configure_optimizers()
    unfrozen = all(p.requires_grad for p in model.backbone.parameters())  # type: ignore
    assert unfrozen
    assert all(p.requires_grad for p in model.fc.parameters())  # type: ignore


def test_freeze_backbone_flag():
    model = SimpleSupervisedModel(
        backbone=DummyBackbone(),
        fc=DummyFC(),
        loss_fn=nn.CrossEntropyLoss(),
        freeze_backbone=True,
    )
    model.configure_optimizers()
    frozen = all(not p.requires_grad for p in model.backbone.parameters())  # type: ignore
    unfrozen_fc = all(p.requires_grad for p in model.fc.parameters())  # type: ignore
    assert frozen
    assert unfrozen_fc


@pytest.mark.parametrize(
    "optimizer_cls", [torch.optim.Adam, torch.optim.SGD, torch.optim.AdamW]
)
def test_supported_optimizers(optimizer_cls):
    model = SimpleSupervisedModel(
        backbone=DummyBackbone(),
        fc=DummyFC(),
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer_cls,
        learning_rate=1e-2,
    )
    optim = model.configure_optimizers()
    if isinstance(optim, tuple):
        optim = optim[0][0]
    assert isinstance(optim, torch.optim.Optimizer)
    assert math.isclose(optim.param_groups[0]["lr"], 1e-2, rel_tol=1e-6)


def test_default_optimizer_is_adam_and_no_scheduler():
    model = SimpleSupervisedModel(
        backbone=DummyBackbone(),
        fc=DummyFC(),
        loss_fn=nn.CrossEntropyLoss(),
    )
    optim = model.configure_optimizers()
    assert isinstance(optim, torch.optim.Adam)
    default_lr = optim.param_groups[0]["lr"]
    assert math.isclose(default_lr, 1e-3, rel_tol=1e-6)  # default learning rate


@pytest.mark.parametrize(
    "scheduler_cls, scheduler_kwargs",
    [
        (torch.optim.lr_scheduler.StepLR, {"step_size": 10, "gamma": 0.1}),
        (torch.optim.lr_scheduler.ExponentialLR, {"gamma": 0.9}),
        (torch.optim.lr_scheduler.CosineAnnealingLR, {"T_max": 5}),
        (torch.optim.lr_scheduler.CyclicLR, {"base_lr": 1e-4, "max_lr": 1e-2}),
        (
            torch.optim.lr_scheduler.OneCycleLR,
            {"max_lr": 1e-2, "steps_per_epoch": 10, "epochs": 1},
        ),
    ],
)
def test_supported_lr_schedulers(scheduler_cls, scheduler_kwargs):
    model = SimpleSupervisedModel(
        backbone=DummyBackbone(),
        fc=DummyFC(),
        loss_fn=nn.CrossEntropyLoss(),
        lr_scheduler=scheduler_cls,
        lr_scheduler_kwargs=scheduler_kwargs,
    )

    optimizers, schedulers = model.configure_optimizers()
    assert isinstance(optimizers[0], torch.optim.Adam)
    assert isinstance(schedulers[0], torch.optim.lr_scheduler.LRScheduler)


def test_forward_without_flatten():
    class UnflattenedBackbone(nn.Module):
        def forward(self, x):
            return x  # return unmodified 4D tensor

    model = SimpleSupervisedModel(
        backbone=UnflattenedBackbone(),
        fc=nn.Identity(),  # no transformation
        loss_fn=nn.MSELoss(),
        flatten=False,
    )

    x = torch.randn(2, 3, 4, 5)
    output = model(x)
    assert output.shape == (
        2,
        3,
        4,
        5,
    ), "Output shape should be unchanged when flatten=False"


def test_forward_with_adapter():
    class DummyBackbone(nn.Module):
        def forward(self, x):
            return torch.ones(x.size(0), 10)

    def adapter(x):
        return x + 5  # simple additive transformation

    class IdentityFC(nn.Module):
        def forward(self, x):
            return x

    model = SimpleSupervisedModel(
        backbone=DummyBackbone(),
        fc=IdentityFC(),
        adapter=adapter,
        loss_fn=nn.MSELoss(),
        flatten=False,
    )

    x = torch.randn(3, 2)
    output = model(x)
    assert torch.allclose(
        output, torch.ones(3, 10) + 5
    ), "Adapter should be applied to backbone output"


def test_metrics_logged(tmp_path):
    logger = CSVLogger(save_dir=tmp_path, name="metrics_test")

    model = SimpleSupervisedModel(
        backbone=DummyBackbone(),
        fc=DummyFC(),
        loss_fn=nn.CrossEntropyLoss(),
        train_metrics={"acc": Accuracy(task="multiclass", num_classes=10)},
        val_metrics={"acc": Accuracy(task="multiclass", num_classes=10)},
    )

    # Dummy dataset with 2 epochs of logs
    dataloader = DataLoader(
        TensorDataset(torch.randn(64, 1, 28, 28), torch.randint(0, 10, (64,))),
        batch_size=32,
    )

    trainer = Trainer(
        accelerator="cpu",
        max_epochs=2,
        logger=logger,
        enable_model_summary=False,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )

    trainer.fit(model, dataloader, dataloader)

    # Verify logs contain loss and metric
    metrics = trainer.callback_metrics
    assert "train_acc" in metrics
    assert "train_loss" in metrics
    assert isinstance(metrics["train_acc"], torch.Tensor)


def test_loss_logging_without_metrics(tmp_path):
    logger = CSVLogger(save_dir=tmp_path, name="loss_only_test")

    model = SimpleSupervisedModel(
        backbone=DummyBackbone(),
        fc=DummyFC(),
        loss_fn=nn.CrossEntropyLoss(),
        train_metrics=None,
        val_metrics=None,
    )

    dataloader = DataLoader(
        TensorDataset(torch.randn(64, 1, 28, 28), torch.randint(0, 10, (64,))),
        batch_size=32,
    )

    trainer = Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )
    trainer.fit(model, dataloader, dataloader)

    metrics = trainer.callback_metrics
    assert "train_loss" in metrics, "train_loss should be logged even without metrics"
    assert "val_loss" in metrics, "val_loss should be logged even without metrics"
    assert isinstance(metrics["train_loss"], torch.Tensor)
    assert isinstance(metrics["val_loss"], torch.Tensor)
