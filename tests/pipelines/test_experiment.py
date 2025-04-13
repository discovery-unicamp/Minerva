import pytest
import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from pathlib import Path
from lightning.pytorch import Trainer
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import torchmetrics
from minerva.utils.typing import PathLike
from minerva.pipelines.experiment import (
    ModelInstantiator,
    ModelInformation,
    ModelConfig,
    get_trainer,
    save_predictions,
    load_predictions,
    save_results,
    load_results,
    perform_train,
    perform_predict,
    perform_evaluation,
)
from minerva.data.data_modules.base import MinervaDataModule
import torch
import numpy as np
import pandas as pd
import pytest
import torchmetrics
from torch.utils.data import Dataset
from typing import Tuple, Union


# Dummy dataset
class DummyDataset(Dataset):
    def __init__(self, labels: Union[np.ndarray, torch.Tensor]):
        self.labels = labels
        self.data = np.random.randn(len(labels), 5).astype(np.float32)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        return self.data[index], int(self.labels[index])

    def __len__(self):
        return len(self.labels)


class DummySegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, labels: np.ndarray):
        self.labels = labels  # shape (N, 2, 2)

    def __getitem__(self, index):
        label = self.labels[index].astype(np.int64)
        dummy_features = np.random.rand(*label.shape).astype(np.float32)
        return dummy_features, label

    def __len__(self):
        return len(self.labels)


# Dummy Lightning model with backbone and head
class DummyLightningModel(L.LightningModule):
    def __init__(self, backbone=None, head=None):
        super().__init__()
        self.backbone = backbone or nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        self.head = head or nn.Linear(16, 3)  # 3 classes
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self(x)
        return logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# Dummy model instantiator
class DummyModelInstantiator(ModelInstantiator):
    def create_model_randomly_initialized(self) -> L.LightningModule:
        return DummyLightningModel()

    def create_model_and_load_backbone(
        self, backbone_checkpoint_path: PathLike
    ) -> L.LightningModule:
        model = DummyLightningModel()
        ckpt = torch.load(backbone_checkpoint_path)
        model.backbone.load_state_dict(ckpt["backbone"])
        return model

    def load_model_from_checkpoint(
        self, checkpoint_path: PathLike
    ) -> L.LightningModule:
        model = DummyLightningModel()
        model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
        return model


@pytest.fixture
def dummy_instantiator():
    return DummyModelInstantiator()


@pytest.fixture
def model_config(dummy_instantiator):
    info = ModelInformation(
        name="dummy_linear_model",
        backbone_name="dummy_backbone",
        task_type="classification",
        input_shape=(32,),
        output_shape=(3,),
        num_classes=3,
        return_logits=True,
    )

    return ModelConfig(
        instantiator=dummy_instantiator,
        information=info,
    )


@pytest.fixture
def dummy_log_dir(tmp_path: Path):
    return tmp_path / "logs" / "model" / "dataset" / "experiment" / "0"


def test_get_trainer_default():
    """Test the get_trainer function with default parameters."""
    log_dir = Path("/tmp/logs")
    trainer = get_trainer(log_dir=log_dir)
    assert isinstance(trainer, L.Trainer)
    assert trainer.max_epochs == 100
    assert trainer.logger is not None and trainer.logger is not False


def test_get_trainer_no_logger():
    """Test the get_trainer function with logger disabled."""
    log_dir = Path("/tmp/logs")
    trainer = get_trainer(log_dir=log_dir, enable_logging=False)
    assert isinstance(trainer, L.Trainer)
    assert trainer.logger is None
    assert trainer.max_epochs == 100


def test_save_predictions_numpy():
    """Test saving predictions as a numpy array."""
    predictions = np.array([1, 2, 3])
    path = Path("/tmp/predictions.npy")
    save_predictions(predictions, path)
    assert path.is_file()
    loaded_predictions = np.load(path)
    np.testing.assert_array_equal(predictions, loaded_predictions)


def test_save_predictions_tensor():
    """Test saving predictions as a torch tensor."""
    predictions = torch.tensor([1, 2, 3])
    path = Path("/tmp/predictions.npy")
    save_predictions(predictions, path)
    assert path.is_file()
    loaded_predictions = np.load(path)
    np.testing.assert_array_equal(predictions.numpy(), loaded_predictions)


def test_load_predictions_numpy():
    """Test loading predictions from a numpy file."""
    predictions = np.array([1, 2, 3])
    path = Path("/tmp/predictions.npy")
    np.save(path, predictions)
    loaded_predictions = load_predictions(path)
    np.testing.assert_array_equal(predictions, loaded_predictions)


def test_save_results():
    """Test saving results as a CSV file."""
    results = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    path = Path("/tmp/results.csv")
    save_results(results, path)
    assert path.is_file()
    loaded_results = pd.read_csv(path)
    pd.testing.assert_frame_equal(results, loaded_results)


def test_load_results():
    """Test loading results from a CSV file."""
    results = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    path = Path("/tmp/results.csv")
    results.to_csv(path, index=False)
    loaded_results = load_results(path)
    pd.testing.assert_frame_equal(results, loaded_results.reset_index(drop=True))


@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("per_sample", [False, True])
@pytest.mark.parametrize("argmax_axis", [1, None])
@pytest.mark.parametrize("cls_type", ["numpy", "torch"])
def test_multiclass_accuracy_and_precision(
    batch_size, per_sample, argmax_axis, cls_type
):
    num_classes = 3
    num_samples = 12
    labels = np.array([0, 1, 2, 1, 0, 2, 1, 2, 0, 1, 2, 0])
    predictions = np.array(
        [
            [0.85, 0.05, 0.02],  # class 0
            [0.10, 0.94, 0.14],  # class 1
            [0.04, 0.07, 0.91],  # class 2
            [0.11, 0.98, 0.03],  # class 1
            [0.93, 0.02, 0.01],  # class 0
            [0.06, 0.08, 0.97],  # class 2
            [0.09, 0.89, 0.02],  # class 1
            [0.02, 0.13, 0.95],  # class 2
            [0.96, 0.01, 0.03],  # class 0
            [0.12, 0.92, 0.06],  # class 1
            [0.05, 0.11, 0.99],  # class 2
            [0.90, 0.04, 0.08],  # class 0
        ]
    )  # Shape: (12, 3)

    if cls_type == "torch":
        labels = torch.from_numpy(labels)
        predictions = torch.from_numpy(predictions)

    dataset = DummyDataset(labels)
    data_module = MinervaDataModule(test_dataset=dataset)

    # Metrics
    metrics = {
        "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
        "precision": torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        ),
    }

    # If argmax_axis is None, we need to apply argmax to predictions manually
    if argmax_axis is None:
        predictions = np.argmax(predictions, axis=1)

    df = perform_evaluation(
        evaluation_metrics=metrics,
        data_module=data_module,
        predictions=predictions,  # type: ignore
        argmax_axis=argmax_axis,
        per_sample=per_sample,
        batch_size=batch_size,
        device="cpu",
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    if per_sample:
        assert len(df) == num_samples
        assert set(df.columns).issuperset({"sample", "accuracy", "precision"})
        for i in range(num_samples):
            assert df["accuracy"].iloc[i] == 1.0
            assert df["precision"].iloc[i] == 1.0
    else:
        assert df.shape[0] == 1
        assert df["accuracy"].iloc[0] == 1.0  # All predictions correct
        assert df["precision"].iloc[0] == 1.0


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("per_sample", [False, True])
@pytest.mark.parametrize("argmax_axis", [1, None])
def test_multiclass_jaccard_index_miou(batch_size, per_sample, argmax_axis):
    num_classes = 3
    labels = np.array([[[0, 1], [1, 2]], [[2, 0], [1, 1]]])  # shape (2, 2, 2)

    # Simulate perfect predictions (logits will be "one-hot" at correct class)
    predictions = np.random.rand(2, 3, 2, 2) * 0.5  # Random values (0, 0.5)

    for i in range(2):  # samples
        for h in range(2):
            for w in range(2):
                cls = labels[i, h, w]
                predictions[i, cls, h, w] = 0.90  # High value to the correct cls

    dataset = DummySegmentationDataset(labels)
    data_module = MinervaDataModule(test_dataset=dataset)

    metrics = {
        "miou": torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes),
    }

    # If argmax_axis is None, we need to apply argmax to predictions manually
    if argmax_axis is None:
        predictions = np.argmax(predictions, axis=1)

    df = perform_evaluation(
        evaluation_metrics=metrics,
        data_module=data_module,
        predictions=predictions,
        argmax_axis=argmax_axis,
        per_sample=per_sample,
        batch_size=batch_size,
        device="cpu",
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    if per_sample:
        assert df.shape[0] == 2
        assert set(df.columns).issuperset({"sample", "miou"})
        for i in range(2):
            assert df["miou"].iloc[i] == 1.0
    else:
        assert df.shape[0] == 1
        assert df["miou"].iloc[0] == 1.0  # perfect prediction


def test_partial_accuracy_and_precision():
    num_classes = 3
    labels = np.array([0, 1, 2, 0])
    predictions = np.array(
        [
            [1, 0, 0],  # correct
            [1, 0, 0],  # wrong (should be class 1)
            [0, 1, 0],  # wrong (should be class 2)
            [1, 0, 0],  # correct
        ]
    )

    dataset = DummyDataset(labels)
    data_module = MinervaDataModule(test_dataset=dataset)

    metrics = {
        "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
    }

    df = perform_evaluation(
        evaluation_metrics=metrics,
        data_module=data_module,
        predictions=predictions,
        argmax_axis=1,
        per_sample=False,
        batch_size=1,
        device="cpu",
    )

    assert df.shape[0] == 1
    assert df["accuracy"].iloc[0] == 0.5  # 2 correct out of 4


def test_metric_reset_between_calls():
    labels = np.array([0, 1])
    predictions1 = np.array([[1, 0], [0, 1]])  # correct
    predictions2 = np.array([[0, 1], [1, 0]])  # wrong

    dataset = DummyDataset(labels)
    data_module = MinervaDataModule(test_dataset=dataset)

    metric = torchmetrics.Accuracy(task="multiclass", num_classes=2)
    metrics = {"accuracy": metric}

    df1 = perform_evaluation(
        evaluation_metrics=metrics,
        data_module=data_module,
        predictions=predictions1,
        argmax_axis=1,
        per_sample=False,
        batch_size=1,
        device="cpu",
    )
    assert df1["accuracy"].iloc[0] == 1.0

    df2 = perform_evaluation(
        evaluation_metrics=metrics,
        data_module=data_module,
        predictions=predictions2,
        argmax_axis=1,
        per_sample=False,
        batch_size=1,
        device="cpu",
    )
    assert df2["accuracy"].iloc[0] == 0.0

    df1 = perform_evaluation(
        evaluation_metrics=metrics,
        data_module=data_module,
        predictions=predictions1,
        argmax_axis=1,
        per_sample=False,
        batch_size=1,
        device="cpu",
    )
    assert df1["accuracy"].iloc[0] == 1.0

    df2 = perform_evaluation(
        evaluation_metrics=metrics,
        data_module=data_module,
        predictions=predictions2,
        argmax_axis=1,
        per_sample=False,
        batch_size=1,
        device="cpu",
    )
    assert df2["accuracy"].iloc[0] == 0.0


def test_exception_on_none_dataset():
    data_module = MinervaDataModule(test_dataset=None)

    predictions = np.zeros((2, 3))
    metrics = {
        "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=3),
    }

    with pytest.raises(ValueError, match="No predict dataset found"):
        perform_evaluation(
            evaluation_metrics=metrics,
            data_module=data_module,
            predictions=predictions,
        )


def test_invalid_metric_type_raises():
    labels = np.array([0, 1])
    dataset = DummyDataset(labels)
    data_module = MinervaDataModule(test_dataset=dataset)

    predictions = np.array([[1, 0], [0, 1]])
    metrics = {"accuracy": "not_a_metric"}

    with pytest.raises(ValueError, match="is not a valid torchmetrics.Metric"):
        perform_evaluation(
            evaluation_metrics=metrics,  # type: ignore
            data_module=data_module,
            predictions=predictions,
            argmax_axis=1,
        )


# Helper function to compare if parameters are equal
def parameters_are_equal(params1, params2) -> bool:
    return all(torch.equal(p1, p2) for p1, p2 in zip(params1, params2))


def test_model_config_properties(model_config):
    assert model_config.information.name == "dummy_linear_model"
    assert model_config.information.input_shape == (32,)
    assert model_config.information.output_shape == (3,)
    assert model_config.information.num_classes == 3


def test_randomly_initialized_model(dummy_instantiator):
    model = dummy_instantiator.create_model_randomly_initialized()
    assert isinstance(model, L.LightningModule)
    x = torch.randn(4, 32)
    logits = model(x)
    assert logits.shape == (4, 3)


def test_finetune_model_loads_backbone_only(dummy_instantiator, tmp_path: Path):
    original_model = dummy_instantiator.create_model_randomly_initialized()
    original_backbone_weights = list(original_model.backbone.parameters())
    original_head_weights = list(original_model.head.parameters())

    # Save backbone only
    save_path = tmp_path / "backbone.ckpt"
    torch.save({"backbone": original_model.backbone.state_dict()}, save_path)

    # Create new model for finetuning
    finetuned_model = dummy_instantiator.create_model_and_load_backbone(save_path)
    assert isinstance(finetuned_model, L.LightningModule)

    # Compare weights
    loaded_backbone_weights = list(finetuned_model.backbone.parameters())  # type: ignore
    loaded_head_weights = list(finetuned_model.head.parameters())  # type: ignore

    assert parameters_are_equal(original_backbone_weights, loaded_backbone_weights)
    assert not parameters_are_equal(
        original_head_weights, loaded_head_weights
    ), "Head should not be loaded during finetuning"


def test_load_full_model_checkpoint(dummy_instantiator, tmp_path: Path):
    model = dummy_instantiator.create_model_randomly_initialized()
    # Save full state dict
    save_path = tmp_path / "full_model.ckpt"
    torch.save({"state_dict": model.state_dict()}, save_path)

    # Load from checkpoint
    loaded_model = dummy_instantiator.load_model_from_checkpoint(save_path)
    assert isinstance(loaded_model, L.LightningModule)

    # Ensure backbone and head weights match
    assert parameters_are_equal(
        list(model.backbone.parameters()),
        list(loaded_model.backbone.parameters()),  # type: ignore
    )
    assert parameters_are_equal(
        list(model.head.parameters()), list(loaded_model.head.parameters())  # type: ignore
    )


def test_trainer_with_logging_and_checkpointing(dummy_log_dir):
    checkpoint_metrics = [
        {"monitor": "val_loss", "mode": "min", "filename": "min_val_loss"}
    ]

    trainer = get_trainer(
        log_dir=dummy_log_dir,
        max_epochs=5,
        enable_logging=True,
        checkpoint_metrics=checkpoint_metrics,
        progress_bar_refresh_rate=1,
    )

    assert isinstance(trainer, Trainer)
    assert trainer.max_epochs == 5
    assert trainer.logger is not False
    assert len(trainer.callbacks) > 0  # type: ignore
    assert any("ModelCheckpoint" in str(type(cb)) for cb in trainer.callbacks)  # type: ignore


def test_trainer_with_logging_and_checkpointing_invalid(dummy_log_dir):
    # Missing "filename" key in checkpoint_metrics
    checkpoint_metrics = [{"monitor": "val_loss", "mode": "min"}]
    with pytest.raises(KeyError):
        trainer = get_trainer(
            log_dir=dummy_log_dir,
            max_epochs=5,
            enable_logging=True,
            checkpoint_metrics=checkpoint_metrics,
            progress_bar_refresh_rate=1,
        )

    # Missing "monitor" key in checkpoint_metrics
    checkpoint_metrics = [{"mode": "min", "filename": "min_val_loss"}]
    with pytest.raises(KeyError):
        trainer = get_trainer(
            log_dir=dummy_log_dir,
            max_epochs=5,
            enable_logging=True,
            checkpoint_metrics=checkpoint_metrics,
            progress_bar_refresh_rate=1,
        )

    # Missing "mode" key in checkpoint_metrics
    checkpoint_metrics = [{"monitor": "val_loss", "filename": "min_val_loss"}]
    with pytest.raises(KeyError):
        trainer = get_trainer(
            log_dir=dummy_log_dir,
            max_epochs=5,
            enable_logging=True,
            checkpoint_metrics=checkpoint_metrics,
            progress_bar_refresh_rate=1,
        )


def test_trainer_without_logging(dummy_log_dir):
    trainer = get_trainer(
        log_dir=dummy_log_dir,
        enable_logging=False,
        checkpoint_metrics=[],
        progress_bar_refresh_rate=1,
        accelerator="cpu",
    )

    assert isinstance(trainer, Trainer)
    assert trainer.logger is None


def test_trainer_without_progress_bar(dummy_log_dir):
    trainer = get_trainer(
        log_dir=dummy_log_dir,
        enable_logging=True,
        checkpoint_metrics=None,
        progress_bar_refresh_rate=0,
        accelerator="cpu",
    )

    assert isinstance(trainer, Trainer)
    assert all(
        "TQDMProgressBar" not in str(type(cb)) for cb in trainer.callbacks  # type: ignore
    )


def test_trainer_with_all_defaults(dummy_log_dir):
    trainer = get_trainer(log_dir=dummy_log_dir)

    assert isinstance(trainer, Trainer)
    assert trainer.max_epochs == 100
    assert trainer.logger is not False


def test_trainer_training_step(dummy_log_dir):
    model = DummyLightningModel()
    trainer = get_trainer(
        log_dir=dummy_log_dir,
        max_epochs=1,
        enable_logging=False,
        accelerator="cpu",
    )

    # Create dummy data
    x = torch.randn(8, 32)
    y = torch.randint(0, 3, (8,))
    dataset = TensorDataset(x, y)

    train_loader = DataLoader(dataset, batch_size=8)

    # Train the model
    trainer.fit(model, train_loader)
    # Check if training step was called
    assert trainer.current_epoch == 1
    assert trainer.train_dataloader is not None


def test_perform_train(dummy_log_dir):
    data_module = MinervaDataModule(
        train_dataset=TensorDataset(torch.randn(16, 32), torch.randint(0, 3, (16,))),
        val_dataset=TensorDataset(torch.randn(8, 32), torch.randint(0, 3, (8,))),
        test_dataset=TensorDataset(torch.randn(8, 32), torch.randint(0, 3, (8,))),
        batch_size=4,
    )

    model = DummyLightningModel()
    trainer = get_trainer(
        log_dir=dummy_log_dir,
        max_epochs=1,
        enable_logging=False,
        accelerator="cpu",
    )

    trained_model = perform_train(data_module, model, trainer)
    assert isinstance(trained_model, L.LightningModule)
    assert trainer.current_epoch == 1


@pytest.mark.parametrize("squeeze", [True, False])
@pytest.mark.parametrize("n_samples", [1, 4, 8])
def test_perform_predict(dummy_log_dir, squeeze, n_samples):
    data_module = MinervaDataModule(
        test_dataset=TensorDataset(
            torch.randn(n_samples, 32), torch.randint(0, 3, (n_samples,))
        ),
        batch_size=4,
        drop_last=False,
    )

    model = DummyLightningModel()
    trainer = get_trainer(
        log_dir=dummy_log_dir,
        max_epochs=1,
        enable_logging=False,
        accelerator="cpu",
    )

    predictions = perform_predict(data_module, model, trainer, squeeze=squeeze)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (n_samples, 3)
    assert predictions.dtype == np.float32  # Raw logits
    assert not np.isnan(predictions).any()  # No NaN values!
    assert not np.isinf(predictions).any()  # No Inf values!


from minerva.pipelines.experiment import (
    Experiment,
    ModelConfig,
    ModelInformation,
    ModelInstantiator,
)


@pytest.mark.parametrize("add_last_ckpt", [True, False])
def test_experiment_initialization(tmp_path, model_config, add_last_ckpt):
    """Test initialization of the Experiment class."""
    data_module = MinervaDataModule(name="dummy_dataset")

    ckpts = [
        {"monitor": "metric", "mode": "max", "filename": "metric"},
        {"monitor": "loss", "mode": "min", "filename": "loss"},
    ]

    experiment = Experiment(
        experiment_name="test_experiment",
        model_config=model_config,
        data_module=data_module,
        root_log_dir=tmp_path,
        max_epochs=10,
        accelerator="cpu",
        execution_id=1000,
        seed=42,
        checkpoint_metrics=ckpts,
        add_last_checkpoint=add_last_ckpt,
    )

    exp_path = (
        tmp_path
        / f"test_experiment/{data_module.dataset_name}/{model_config.information.name}/1000"
    )

    assert experiment.experiment_name == "test_experiment"
    assert experiment.model_config == model_config  # Comparing references
    assert experiment.data_module == data_module  # Comparing references
    assert str(experiment.log_dir) == str(exp_path)
    assert str(experiment._checkpoint_dir) == str(exp_path / "checkpoints")
    assert str(experiment._predictions_dir) == str(exp_path / "predictions")
    assert str(experiment._results_dir) == str(exp_path / "results")
    assert str(experiment._training_metrics_path) == str(exp_path / "metrics.csv")
    assert experiment.max_epochs == 10
    assert experiment.accelerator == "cpu"
    assert experiment.execution_id == "1000"
    assert experiment.seed == 42
    assert len(experiment.evaluation_metrics) == 0
    assert len(experiment.per_sample_evaluation_metrics) == 0

    assert len(experiment.checkpoint_metrics) == (3 if add_last_ckpt else 2)
    for i, ckpt in enumerate(experiment.checkpoint_metrics):
        current_ckpt = experiment.checkpoint_metrics[i]
        assert current_ckpt["monitor"] == ckpt["monitor"]
        assert current_ckpt["mode"] == ckpt["mode"]
        assert current_ckpt["filename"] == ckpt["filename"]

    if add_last_ckpt:
        assert experiment.checkpoint_metrics[-1]["monitor"] == None
        assert experiment.checkpoint_metrics[-1]["mode"] == "min"
        assert experiment.checkpoint_metrics[-1]["filename"] == "last"


def test_experiment_initialization_with_invalid_ckpts(tmp_path, model_config):
    """Test initialization of the Experiment class."""
    data_module = MinervaDataModule(name="dummy_dataset")

    ckpts = [
        [{"mode": "max", "filename": "metric"}],
        [{"monitor": "loss", "filename": "loss"}],
        [
            {"monitor": "loss", "mode": "max"},
        ],
        [{"monitor": "loss", "mode": "min", "flename": "loss"}],
        [{"monitor": "loss"}],
        [{"monitor": "loss", "mode": "min", "filename": "metric"}, {}],
        [{"monitor": "loss"}, {"monitor": "loss", "mode": "min", "filename": "metric"}],
    ]

    for ckpt in ckpts:
        with pytest.raises(ValueError, match="Checkpoint metric must contain a"):
            experiment = Experiment(
                experiment_name="test_experiment",
                model_config=model_config,
                data_module=data_module,
                root_log_dir=tmp_path,
                max_epochs=10,
                accelerator="cpu",
                execution_id=1000,
                seed=42,
                checkpoint_metrics=ckpt,
                add_last_checkpoint=False,
            )


def test_experiment_checkpoint_paths(tmp_path, model_config):
    """Test the checkpoint_paths property of the Experiment class."""
    """Test initialization of the Experiment class."""
    data_module = MinervaDataModule(name="dummy_dataset")

    ckpts = [
        {"monitor": "metric", "mode": "max", "filename": "metric"},
        {"monitor": "loss", "mode": "min", "filename": "loss"},
    ]

    experiment = Experiment(
        experiment_name="test_experiment",
        model_config=model_config,
        data_module=data_module,
        root_log_dir=tmp_path,
        max_epochs=10,
        accelerator="cpu",
        execution_id=1000,
        seed=42,
        checkpoint_metrics=ckpts,
        add_last_checkpoint=True,
    )

    # Create dummy checkpoint files
    checkpoint_dir = experiment._checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "metric.ckpt").touch()
    (checkpoint_dir / "loss.ckpt").touch()
    (checkpoint_dir / "last.ckpt").touch()

    checkpoint_paths = experiment.checkpoint_paths
    assert len(checkpoint_paths) == 3
    assert "last" in checkpoint_paths
    assert "metric" in checkpoint_paths
    assert "loss" in checkpoint_paths
    assert checkpoint_paths["last"] == checkpoint_dir / "last.ckpt"
    assert checkpoint_paths["metric"] == checkpoint_dir / "metric.ckpt"
    assert checkpoint_paths["loss"] == checkpoint_dir / "loss.ckpt"


def test_experiment_training_metrics(tmp_path, model_config):
    """Test the training_metrics property of the Experiment class."""
    data_module = MinervaDataModule(name="dummy_dataset")

    experiment = Experiment(
        experiment_name="test_experiment",
        model_config=model_config,
        data_module=data_module,
        root_log_dir=tmp_path,
        accelerator="cpu",
    )

    # No metrics file should exist at this point
    assert experiment.training_metrics_path is None
    assert experiment.training_metrics is None

    # Create dummy metrics file (as CSV Logger)
    metrics_path = experiment._training_metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"epoch": [1, 2], "accuracy": [0.8, 0.9]}).to_csv(
        metrics_path, index=False
    )
    # Assert training_metrics_path is set once the file is created
    assert str(experiment.training_metrics_path) == str(
        experiment._training_metrics_path
    )
    # Read CSV and assert the contents
    training_metrics = experiment.training_metrics
    assert training_metrics is not None
    assert "epoch" in training_metrics.columns
    assert "accuracy" in training_metrics.columns
    assert training_metrics["accuracy"].iloc[0] == 0.8
    assert training_metrics["accuracy"].iloc[1] == 0.9


def test_experiment_load_predictions(tmp_path, model_config):
    """Test the load_predictions_of_ckpt method of the Experiment class."""
    data_module = MinervaDataModule(name="dummy_dataset")

    experiment = Experiment(
        experiment_name="test_experiment",
        model_config=model_config,
        data_module=data_module,
        root_log_dir=tmp_path,
        accelerator="cpu",
    )

    # Check prediction_paths has nothing
    assert len(experiment.prediction_paths) == 0

    # Create dummy predictions file
    predictions_dir = experiment._predictions_dir
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = predictions_dir / "ckpt1.npy"
    np.save(predictions_path, np.array([1, 2, 3]))
    # Assert prediction_paths is set once the file is created
    assert len(experiment.prediction_paths) == 1

    predictions = experiment.load_predictions_of_ckpt("ckpt1")
    np.testing.assert_array_equal(predictions, np.array([1, 2, 3]))

    with pytest.raises(Exception, match="Prediction file 'ckpt2' not found"):
        experiment.load_predictions_of_ckpt("ckpt2")


def test_experiment_load_results(tmp_path, model_config):
    """Test the load_results_of_ckpt method of the Experiment class."""
    data_module = MinervaDataModule(name="dummy_dataset")

    experiment = Experiment(
        experiment_name="test_experiment",
        model_config=model_config,
        data_module=data_module,
        root_log_dir=tmp_path,
        accelerator="cpu",
    )

    # Assert result_paths has nothing
    assert len(experiment.results_paths) == 0

    # Create dummy results file
    results_dir = experiment._results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "ckpt1.csv"
    pd.DataFrame({"metric": ["accuracy"], "value": [0.9]}).to_csv(
        results_path, index=False
    )
    # Assert results_paths is set once the file is created
    assert len(experiment.results_paths) == 1

    # Load results
    results = experiment.load_results_of_ckpt("ckpt1")
    assert "metric" in results.columns
    assert "value" in results.columns
    assert results["value"].iloc[0] == 0.9

    # Load invalid result
    with pytest.raises(Exception, match="Results file 'ckpt2' not found"):
        experiment.load_results_of_ckpt("ckpt2")


def test_experiment_cleanup(tmp_path, model_config):
    """Test the cleanup method of the Experiment class."""
    data_module = MinervaDataModule(name="dummy_dataset")

    the_path = tmp_path / "dummy_dir"
    the_path.mkdir(parents=True, exist_ok=True)

    experiment = Experiment(
        experiment_name="test_experiment",
        model_config=model_config,
        data_module=data_module,
        root_log_dir=the_path,
        accelerator="cpu",
    )

    # Create dummy log directory
    log_dir = experiment.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "dummy_file.txt").touch()

    assert log_dir.exists()
    experiment.cleanup()
    assert not log_dir.exists()
    # Check if root log dir is still there (just not the experiment dir)
    assert the_path.exists()

    # Clean again (should do nothing)
    experiment.cleanup()
    assert not log_dir.exists()
    # Check if root log dir is still there (just not the experiment dir)
    assert the_path.exists()


@pytest.mark.parametrize("enable_logging", [True, False])
def test_trainer_parameters_default(tmp_path, model_config, enable_logging):
    """Test the default parameters for the trainer."""
    data_module = MinervaDataModule(name="dummy_dataset")
    experiment = Experiment(
        experiment_name="test_experiment",
        model_config=model_config,
        data_module=data_module,
        root_log_dir=tmp_path,
        add_last_checkpoint=True,
        progress_bar_refresh_rate=10,
        accelerator="cpu",
    )

    trainer_params = experiment._trainer_parameters(enable_logging=enable_logging)
    assert trainer_params["log_dir"] == experiment.log_dir
    assert trainer_params["max_epochs"] == experiment.max_epochs
    assert trainer_params["limit_train_batches"] is None
    assert trainer_params["limit_val_batches"] is None
    assert trainer_params["limit_test_batches"] is None
    assert trainer_params["limit_predict_batches"] is None
    assert trainer_params["accelerator"] == experiment.accelerator
    assert trainer_params["strategy"] == experiment.strategy
    assert trainer_params["devices"] == experiment.devices
    assert trainer_params["num_nodes"] == experiment.num_nodes
    assert trainer_params["progress_bar_refresh_rate"] == 10
    assert trainer_params["enable_logging"] == enable_logging
    assert len(trainer_params["checkpoint_metrics"]) == 1

    trainer = get_trainer(**trainer_params)
    assert isinstance(trainer, L.Trainer)
    assert trainer.max_epochs == experiment.max_epochs
    if enable_logging:
        assert isinstance(trainer.logger, CSVLogger)
    else:
        assert trainer.logger is None
    assert trainer.limit_train_batches == 1.0
    assert trainer.limit_val_batches == 1.0
    assert trainer.limit_test_batches == 1.0
    assert trainer.limit_predict_batches == 1.0


def test_trainer_parameters_debug(tmp_path, model_config):
    """Test the parameters for the trainer in debug mode."""
    data_module = MinervaDataModule(name="dummy_dataset")
    experiment = Experiment(
        experiment_name="test_experiment",
        model_config=model_config,
        data_module=data_module,
        root_log_dir=tmp_path,
        accelerator="cpu",
    )

    trainer_params = experiment._trainer_parameters(debug=True)
    assert trainer_params["max_epochs"] == experiment.NUM_DEBUG_EPOCHS
    assert trainer_params["limit_train_batches"] == experiment.NUM_DEBUG_BATCHES
    assert trainer_params["limit_val_batches"] == experiment.NUM_DEBUG_BATCHES
    assert trainer_params["limit_test_batches"] == experiment.NUM_DEBUG_BATCHES
    assert trainer_params["enable_logging"] is False
    assert trainer_params["checkpoint_metrics"] is None


def test_train_model_from_scratch(tmp_path, model_config):
    """Test training a model from scratch."""
    data_module = MinervaDataModule(
        train_dataset=TensorDataset(torch.randn(16, 32), torch.randint(0, 3, (16,))),
        val_dataset=TensorDataset(torch.randn(8, 32), torch.randint(0, 3, (8,))),
        test_dataset=TensorDataset(torch.randn(8, 32), torch.randint(0, 3, (8,))),
        batch_size=4,
    )
    experiment = Experiment(
        experiment_name="test_experiment",
        model_config=model_config,
        data_module=data_module,
        root_log_dir=tmp_path,
        accelerator="cpu",
    )

    results = experiment._train_model()
    assert isinstance(results["model"], L.LightningModule)
    assert isinstance(results["trainer"], L.Trainer)
    assert results["trainer"].current_epoch == experiment.max_epochs
    assert results["log_dir"] == experiment.log_dir
    assert results["metrics_path"] == experiment._training_metrics_path
    assert len(results["checkpoints"]) > 0


def test_train_model_with_pretrained_backbone(tmp_path, model_config):
    """Test training a model with a pretrained backbone."""
    # Save a dummy pretrained backbone checkpoint
    pretrained_backbone_path = tmp_path / "pretrained_backbone.ckpt"
    dummy_model = DummyLightningModel()
    torch.save(
        {"backbone": dummy_model.backbone.state_dict()}, pretrained_backbone_path
    )

    data_module = MinervaDataModule(
        train_dataset=TensorDataset(torch.randn(16, 32), torch.randint(0, 3, (16,))),
        val_dataset=TensorDataset(torch.randn(8, 32), torch.randint(0, 3, (8,))),
        test_dataset=TensorDataset(torch.randn(8, 32), torch.randint(0, 3, (8,))),
        batch_size=4,
    )
    experiment = Experiment(
        experiment_name="test_experiment",
        model_config=model_config,
        data_module=data_module,
        root_log_dir=tmp_path,
        pretrained_backbone_ckpt_path=pretrained_backbone_path,
        add_last_checkpoint=True,
        max_epochs=2,
        accelerator="cpu",
    )

    results = experiment._train_model()
    assert isinstance(results["model"], L.LightningModule)
    assert isinstance(results["trainer"], L.Trainer)
    assert results["trainer"].current_epoch == experiment.max_epochs
    assert results["log_dir"] == experiment.log_dir
    assert results["metrics_path"] == experiment._training_metrics_path
    assert len(results["checkpoints"]) > 0

    # check if last checkpoint exists
    assert "last" in experiment.checkpoint_paths
    assert experiment.checkpoint_paths["last"].exists()
    experiment.max_epochs = 4

    results = experiment._train_model(resume_from_ckpt="last")
    assert isinstance(results["model"], L.LightningModule)
    assert isinstance(results["trainer"], L.Trainer)
    assert results["trainer"].current_epoch == experiment.max_epochs
    assert results["log_dir"] == experiment.log_dir
    assert results["metrics_path"] == experiment._training_metrics_path
    assert len(results["checkpoints"]) > 0

    with pytest.raises(ValueError, match="Checkpoint 'not_existing' not found"):
        experiment._train_model(resume_from_ckpt="not_existing")


@pytest.mark.parametrize("save_predictions", [True, False])
@pytest.mark.parametrize("save_results", [True, False])
@pytest.mark.parametrize("ckpts_to_evaluate", [None, "metric", ["last", "metric"]])
@pytest.mark.parametrize("n_samples", [1, 9])
def test_evaluate_model(
    tmp_path, model_config, ckpts_to_evaluate, n_samples, save_predictions, save_results
):
    """Test _evaluate_model with various configurations."""
    # Create dummy data module
    data_module = MinervaDataModule(
        test_dataset=TensorDataset(
            torch.randn(n_samples, 32), torch.randint(0, 3, (n_samples,))
        ),
        batch_size=4,
        drop_last=False,
    )

    # Create experiment
    experiment = Experiment(
        experiment_name="test_experiment",
        model_config=model_config,
        data_module=data_module,
        root_log_dir=tmp_path,
        save_predictions=save_predictions,
        save_results=save_results,
        accelerator="cpu",
        evaluation_metrics={
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=3),
            "precision": torchmetrics.Precision(
                task="multiclass", num_classes=3, average="macro"
            ),
        },
        per_sample_evaluation_metrics={
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=3),
            "precision": torchmetrics.Precision(
                task="multiclass", num_classes=3, average="macro"
            ),
        },
    )

    # Create dummy checkpoints
    checkpoint_dir = experiment._checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "last.ckpt").touch()
    (checkpoint_dir / "metric.ckpt").touch()

    # Mock model instantiator to return a dummy model
    def mock_load_model_from_checkpoint(checkpoint_path):
        return DummyLightningModel()

    experiment.model_config.instantiator.load_model_from_checkpoint = (
        mock_load_model_from_checkpoint
    )

    # Perform evaluation
    results = experiment._evaluate_model(
        ckpts_to_evaluate=ckpts_to_evaluate,
        debug=False,
        print_summary=True,
    )

    # Assertions
    if ckpts_to_evaluate is None:
        ckpts_to_evaluate = ["last", "metric"]

    if not isinstance(ckpts_to_evaluate, list):
        ckpts_to_evaluate = [ckpts_to_evaluate]

    for ckpt_name in ckpts_to_evaluate:
        if save_predictions:
            assert results[ckpt_name]["predictions_path"] is not None
            assert results[ckpt_name]["predictions_path"].is_file()
        else:
            assert results[ckpt_name]["predictions_path"] is None

        if save_results:
            assert results[ckpt_name]["results_path"] is not None
            assert results[ckpt_name]["results_path_per_sample"] is not None
        else:
            assert results[ckpt_name]["results_path"] is None
            assert results[ckpt_name]["results_path_per_sample"] is None

        assert "accuracy" in results[ckpt_name]["results"].columns
        assert "precision" in results[ckpt_name]["results"].columns
        assert "accuracy" in results[ckpt_name]["results_per_sample"].columns
        assert "precision" in results[ckpt_name]["results_per_sample"].columns

        assert len(results[ckpt_name]["results"]) == 1
        assert len(results[ckpt_name]["results_per_sample"]) == n_samples


@pytest.mark.parametrize("save_predictions", [True, False])
@pytest.mark.parametrize("save_results", [True, False])
def test_evaluate_model_no_metrics(
    tmp_path, model_config, save_predictions, save_results
):
    """Test _evaluate_model with various configurations."""
    # Create dummy data module
    data_module = MinervaDataModule(
        test_dataset=TensorDataset(torch.randn(8, 32), torch.randint(0, 3, (8,))),
        batch_size=4,
        drop_last=False,
    )

    # Create experiment
    experiment = Experiment(
        experiment_name="test_experiment",
        model_config=model_config,
        data_module=data_module,
        root_log_dir=tmp_path,
        save_predictions=save_predictions,
        save_results=save_results,
        accelerator="cpu",
        evaluation_metrics=None,
        per_sample_evaluation_metrics=None,
        add_last_checkpoint=True,
    )

    # Create dummy checkpoints
    checkpoint_dir = experiment._checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "last.ckpt").touch()

    # Mock model instantiator to return a dummy model
    def mock_load_model_from_checkpoint(checkpoint_path):
        return DummyLightningModel()

    experiment.model_config.instantiator.load_model_from_checkpoint = (
        mock_load_model_from_checkpoint
    )

    # Perform evaluation
    results = experiment._evaluate_model(
        ckpts_to_evaluate="last",
        debug=False,
        print_summary=True,
    )

    for ckpt_name in ["last"]:
        assert results[ckpt_name]["results"] is None
        assert results[ckpt_name]["results_per_sample"] is None

        if save_predictions:
            assert results[ckpt_name]["predictions_path"] is not None
            assert results[ckpt_name]["predictions_path"].is_file()
        else:
            assert results[ckpt_name]["predictions_path"] is None

        if save_results:
            assert results[ckpt_name]["results_path"] is not None
            assert results[ckpt_name]["results_path_per_sample"] is not None
            assert not results[ckpt_name]["results_path"].exists()
            assert not results[ckpt_name]["results_path_per_sample"].exists()
        else:
            assert results[ckpt_name]["results_path"] is None
            assert results[ckpt_name]["results_path_per_sample"] is None


def test_evaluate_model_no_checkpoints(tmp_path, model_config):
    """Test _evaluate_model when no checkpoints are available."""
    # Create dummy data module
    data_module = MinervaDataModule(
        test_dataset=TensorDataset(torch.randn(8, 32), torch.randint(0, 3, (8,))),
        batch_size=4,
        drop_last=False,
    )
    experiment = Experiment(
        experiment_name="test_experiment",
        model_config=model_config,
        data_module=data_module,
        root_log_dir=tmp_path,
        accelerator="cpu",
    )

    with pytest.raises(ValueError, match="No checkpoints found"):
        experiment._evaluate_model()


def test_evaluate_model_invalid_checkpoint(tmp_path, model_config):
    """Test _evaluate_model with an invalid checkpoint name."""
    data_module = MinervaDataModule(
        test_dataset=TensorDataset(torch.randn(8, 32), torch.randint(0, 3, (8,))),
        batch_size=4,
        drop_last=False,
    )
    experiment = Experiment(
        experiment_name="test_experiment",
        model_config=model_config,
        data_module=data_module,
        root_log_dir=tmp_path,
        accelerator="cpu",
    )

    # Create dummy checkpoint
    checkpoint_dir = experiment._checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "valid.ckpt").touch()

    with pytest.raises(ValueError, match="Checkpoint 'invalid' not found"):
        experiment._evaluate_model(ckpts_to_evaluate="invalid")


def test_evaluate_model_no_predict_dataset(tmp_path, model_config):
    """Test _evaluate_model when no predict dataset is provided."""
    data_module = MinervaDataModule(
        test_dataset=None,
        batch_size=4,
        drop_last=False,
    )
    experiment = Experiment(
        experiment_name="test_experiment",
        model_config=model_config,
        data_module=data_module,
        root_log_dir=tmp_path,
        accelerator="cpu",
    )

    # Create dummy checkpoint
    checkpoint_dir = experiment._checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "last.ckpt").touch()

    with pytest.raises(ValueError, match="No predict dataset found"):
        experiment._evaluate_model()
