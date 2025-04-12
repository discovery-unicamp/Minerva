from abc import ABC, abstractmethod
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union
import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar as ProgressBar,
)
import numpy as np
import pandas as pd
import torchmetrics
import tqdm
from minerva.data.data_modules.base import MinervaDataModule
from minerva.pipelines.base import Pipeline
from minerva.utils.typing import PathLike
from dataclasses import asdict, dataclass
from minerva.utils.string_ops import tree_like_formating, indent_text


class ModelInstantiator(ABC):
    """Abstract base class for lazy instantiation of PyTorch Lightning models.

    This interface defines a standardized way to construct models in three
    common training scenarios:

    1. Training from scratch: the entire model (backbone + head) is randomly
       initialized.
    2. Finetuning: a pretrained backbone is loaded from a checkpoint, while the
       head is randomly initialized.
    3. Inference/Evaluation: the full model is restored from a previously
       saved checkpoint. Usually, this checkpoint is generated using one of the
       two scenarios above.

    This abstraction allows for flexible and decoupled model construction across
    various stages of the machine learning lifecycle. Thus, is expected that
    model's architecture follows the same pattern as the one below:

        +-------------------------------+
        |   Model (LightningModule)     |
        |                               |
        |     +-----------------+       |
        |     |    Backbone     |       |   --> Feature extractor
        |     +-----------------+       |
        |             |                 |
        |             v                 |
        |        +----------+           |
        |        |   Head   |           |   --> Task-specific layers
        |        +----------+           |
        +-------------------------------+

    Definitions
    -----------
    - Backbone: Core feature extractor (e.g., ResNet, Transformer encoder).
    - Head: Task-specific layers (e.g., classification head, regression head).

    Implementations of this class should handle the appropriate model loading
    logic for each use case described above.
    """

    @abstractmethod
    def create_model_randomly_initialized(self) -> L.LightningModule:
        """Create a model with both backbone and head randomly initialized.
        Typically used when training a model from scratch.

        Returns
        -------
        L.LightningModule
            A Lightning model fully initialized with random weights, ready for
            training.
        """
        raise NotImplementedError(
            "create_model_randomly_initialized must be implemented."
        )

    @abstractmethod
    def create_model_and_load_backbone(
        self, backbone_checkpoint_path: PathLike
    ) -> L.LightningModule:
        """Create a model for finetuning with a pretrained backbone and a
        new head (randomly initialized). This method should load the backbone
        weights from the specified checkpoint and attach a freshly initialized
        head for the downstream task. User must handle the logic to load the
        backbone weights into the model's state dict.

        Parameters
        ----------
        backbone_checkpoint_path : PathLike
            Path to the checkpoint containing pretrained backbone weights. The
            checkpoint must be compatible with the model architecture.

        Returns
        -------
        L.LightningModule
            The model ready for finetuning (pretrained backbone, new head).
        """
        pass

    @abstractmethod
    def load_model_from_checkpoint(
        self, checkpoint_path: PathLike
    ) -> L.LightningModule:
        """Load the full model (backbone and head) from a saved checkpoint.
        Typically used for resuming training, evaluation, or inference when the
        model must be restored in its entirety. In practice, the checkpoint
        should be one created using `create_model_and_load_backbone` or
        `create_model_randomly_initialized`.
        The checkpoint must be compatible with the model architecture.

        Parameters
        ----------
        checkpoint_path : PathLike
            Path to the checkpoint file containing the full model state.

        Returns
        -------
        L.LightningModule
            A Lightning model fully restored from checkpoint, ready for
            evaluation or inference.
        """
        raise NotImplementedError(
            "load_model_from_checkpoint must be implemented in the subclass."
        )


@dataclass
class ModelInformation:
    """Container for metadata related to a machine learning model configuration.

    This class stores essential information about a model's identity,
    architecture, data shapes, and output behavior. Such metadata is useful
    for tasks such as logging, reproducibility, automated evaluation, or
    dynamic behavior in pipelines.

    Attributes
    ----------
    name : str
        A unique identifier for the model configuration. Commonly used for
        logging, saving checkpoints, or experiment tracking.

    backbone_name : Optional[str], optional
        The name of the backbone architecture used in the model (e.g.,
        "resnet50", "vit-base"). Useful for identifying model variants or
        tracking architectural differences.

    task_type : Optional[str], optional
        The task the model is designed for (e.g., "classification",
        "segmentation", "detection"). Enables downstream logic to adapt based
        on the task type.

    input_shape : Optional[Tuple[int, ...]], optional
        Expected shape of input tensors (excluding batch size), typically in
        the format (C, H, W) for image data. For example: (3, 224, 224) for an
        RGB image of size 224x224.

    output_shape : Optional[Tuple[int, ...]], optional
        Expected shape of model outputs (excluding batch size). Examples
        include:
            - (6, 224, 224) for semantic segmentation logits with 6 classes
            - (224, 224) for semantic segmentation predictions (argmax indices)
            - (6,) for classification logits (6 classes)
            - (1,) for classification predictions as class indices

    num_classes : Optional[int], optional
        Total number of classes the model is predicting. Primarily relevant for
        classification or segmentation tasks.

    return_logits : Optional[bool], optional
        If True, the model returns raw logits. If False, it returns
        post-processed class predictions (e.g., argmax indices or
        probabilities).
    """

    name: str
    backbone_name: Optional[str] = None
    task_type: Optional[str] = None
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    num_classes: Optional[int] = None
    return_logits: Optional[bool] = None


class ModelConfig:
    """Encapsulates the full configuration of a model for use in a training or
    inference pipeline.

    A `ModelConfig` brings together two key components:

    - `ModelInstantiator`: Responsible for creating the model in different
      modes (lazily instantiated):
        - From scratch (randomly initialized)
        - Finetuning (load pretrained backbone, new head)
        - From checkpoint (fully restored model)

    - `ModelInformation`: Contains descriptive metadata about the model such
      as input/output shapes, number of classes, backbone used, and task type.

    This class serves as the primary interface for managing and accessing
    model configuration throughout the lifecycle of training, evaluation, or
    deployment.
    """

    def __init__(
        self,
        instantiator: ModelInstantiator,
        information: ModelInformation,
    ):
        """Initialize a model configuration.

        Parameters
        ----------
        instantiator : ModelInstantiator
            An instance responsible for constructing the model in various
            training modes (random init, load backbone, load full checkpoint).
            This enables lazy instantiation depending on the training phase.

        information : ModelInformation
            Metadata describing the model's architecture and behavior.
            Includes input/output shapes, task type, number of classes, and
            other relevant info useful for logging, validation, and downstream
            processing.
        """
        self.instantiator = instantiator
        self.information = information

    def __str__(self):
        return (
            f"ModelConfig\n"
            + f"├── Instantiator: {self.instantiator.__class__.__name__}\n"
            + indent_text(tree_like_formating(asdict(self.information)), spaces=0)
        )


# -------- Functional interfaces --------
def get_trainer(
    log_dir: Path,
    max_epochs: int = 100,
    limit_train_batches: Optional[Union[int, float]] = None,
    limit_val_batches: Optional[Union[int, float]] = None,
    limit_test_batches: Optional[Union[int, float]] = None,
    limit_predict_batches: Optional[Union[int, float]] = None,
    accelerator: str = "auto",
    strategy: str = "auto",
    devices: Optional[Union[int, list[int], str]] = "auto",
    num_nodes: int = 1,
    progress_bar_refresh_rate: int = 1,
    enable_logging: bool = True,
    checkpoint_metrics: Optional[List[Dict[str, str]]] = None,
    precision: str = "32-true",
    accumulate_grad_batches: int = 1,
    deterministic: bool = False,
    benchmark: bool = True,
    profiler: Optional[str] = None,
    overfit_batches: Union[int, float] = 0.0,
    sync_batchnorm: bool = False,
) -> L.Trainer:
    """Creates and configures a PyTorch Lightning Trainer instance.

    This function encapsulates all necessary options for flexible training,
    evaluation, or inference, including logging, checkpointing, device setup,
    precision, and more.

    Parameters
    ----------
    log_dir : Path
        Directory path where logs and checkpoints will be saved.

    max_epochs : int, default=100
        Maximum number of epochs for training.

    limit_train_batches : int or float, optional
        Limit on the number of training batches per epoch. Can be an integer
        (absolute number) or a float (fraction of total batches).

    limit_val_batches : int or float, optional
        Limit on the number of validation batches per epoch.

    limit_test_batches : int or float, optional
        Limit on the number of test batches per epoch.

    limit_predict_batches : int or float, optional
        Limit on the number of prediction batches.

    accelerator : str, default="auto"
        Hardware accelerator to use (e.g., "gpu", "cpu", "tpu", "auto").

    strategy : str, default="auto"
        Distributed training strategy (e.g., "ddp", "deepspeed", etc.).

    devices : int, list of int, or str, optional, default="auto"
        Devices to use for training (e.g., 1, [0,1], "auto").

    num_nodes : int, default=1
        Number of nodes to use for distributed training.

    progress_bar_refresh_rate : int, default=1
        Frequency (in steps) at which the progress bar is updated.
        Set to 0 to disable.

    enable_logging : bool, default=True
        Whether to enable CSV logging.

    checkpoint_metrics : list of dict, optional
        List of dictionaries containing checkpoint configurations. Each
        dictionary should specify "monitor", "mode", and "filename".

    precision : str, default="32-true"
        Numerical precision to use during training (e.g., 32-true, 16-mixed).

    accumulate_grad_batches : int, default=1
        Number of batches for which gradients should be accumulated before
        performing an optimizer step.

    deterministic : bool, default=False
        If True, sets deterministic behavior for reproducibility.

    benchmark : bool, default=True
        Enables the cudnn.benchmark flag for optimized performance on fixed
        input sizes.

    profiler : str, optional
        Enables performance profiling (e.g., "simple", "advanced").

    overfit_batches : int or float, default=0.0
        Uses a fraction or number of batches for both training and validation
        to quickly debug overfitting behavior.

    sync_batchnorm : bool, default=False
        Synchronizes batch norm layers across devices during distributed
        training.

    Returns
    -------
    L.Trainer
        A configured PyTorch Lightning Trainer instance.
    """

    if enable_logging:
        logger = CSVLogger(
            save_dir=log_dir.parents[1],
            name=log_dir.parents[0].name,
            version=log_dir.name,
        )
    else:
        logger = False

    callbacks = []
    enable_checkpointing = False
    if checkpoint_metrics:
        for ckpt_metric in checkpoint_metrics:
            ckpt_kwargs = {
                "monitor": ckpt_metric["monitor"],
                "mode": ckpt_metric["mode"],
                "filename": ckpt_metric["filename"],
                "save_last": False,
                "enable_version_counter": False,
            }
            callbacks.append(ModelCheckpoint(**ckpt_kwargs))

        enable_checkpointing = True
    else:
        enable_checkpointing = False

    enable_progress_bar = True
    log_every_n_steps = None

    if progress_bar_refresh_rate == 0:
        enable_progress_bar = False
    else:
        callbacks.append(ProgressBar(refresh_rate=progress_bar_refresh_rate))

    return L.Trainer(
        accelerator=accelerator,
        devices=devices,  # type: ignore
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=enable_checkpointing,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=True,
        log_every_n_steps=log_every_n_steps,
        max_epochs=max_epochs,
        strategy=strategy,
        num_nodes=num_nodes,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        limit_predict_batches=limit_predict_batches,
        precision=precision,  # type: ignore
        accumulate_grad_batches=accumulate_grad_batches,
        deterministic=deterministic,
        benchmark=benchmark,
        inference_mode=True,
        profiler=profiler,
        overfit_batches=overfit_batches,
        sync_batchnorm=sync_batchnorm,
    )


def save_predictions(
    predictions: Union[np.ndarray, torch.Tensor], path: PathLike
) -> None:
    """Save predictions to a given path.

    Parameters
    ----------
    predictions : Union[np.ndarray, torch.Tensor]
        The prediction data to save.
    path : PathLike
        The path where the predictions will be saved.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    if not isinstance(predictions, np.ndarray):
        raise ValueError("Predictions must be a numpy array.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == "npz":
        np.savez(path, predictions=predictions)
    else:
        np.save(path, predictions)

    print(f"Predictions saved to {path}")


def load_predictions(path: PathLike) -> np.ndarray:
    """Load a prediction from a given path.

    Parameters
    ----------
    path : PathLike
        The path to the prediction file.

    Returns
    -------
    np.ndarray
        The loaded prediction data.
    """
    path = Path(path)
    if path.suffix == "npz":
        data = np.load(path, allow_pickle=True)
        if "predictions" in data:
            return data["predictions"]
        else:
            raise ValueError("No 'predictions' key found in the npz file.")
    else:
        return np.load(path, allow_pickle=True)


def save_results(results: pd.DataFrame, path: PathLike, index: bool = False) -> None:
    """Save results to a given path.

    Parameters
    ----------
    results : pd.DataFrame
        The results data to save.
    path : PathLike
        The path where the results will be saved.
    index : bool, optional
        Whether to save the index of the DataFrame, by default False
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=index)
    print(f"Results saved to {path}")


def load_results(path: PathLike) -> pd.DataFrame:
    """Load results from a given path.

    Parameters
    ----------
    path : PathLike
        The path to the results file.

    Returns
    -------
    pd.DataFrame
        The loaded results data.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"File {path} does not exist.")
    return pd.read_csv(path)


def perform_train(
    data_module: MinervaDataModule,
    model: L.LightningModule,
    trainer: L.Trainer,
    resume_from_ckpt: Optional[PathLike] = None,
) -> L.LightningModule:
    """Train the model using the provided data module and trainer.

    Parameters
    ----------
    data_module : MinervaDataModule
        The data module containing the training and validation datasets.
    model : L.LightningModule
        The model to be trained.
    trainer : L.Trainer
        The trainer instance to use for training.
    resume_from_ckpt : Optional[PathLike], optional
        A path to a checkpoint in which to resume training. If None, training
        starts from scratch. By default None

    Returns
    -------
    L.LightningModule
        The trained model.
    """
    trainer.fit(model, data_module, ckpt_path=resume_from_ckpt)
    return model


def perform_predict(
    data_module: MinervaDataModule,
    model: L.LightningModule,
    trainer: L.Trainer,
    squeeze: bool = False,
) -> np.ndarray:
    """Perform predictions using the provided data module and trainer.

    Parameters
    ----------
    data_module : MinervaDataModule
        The data module containing the dataset for predictions.
    model : L.LightningModule
        The model to be used for predictions.
    trainer : L.Trainer
        The trainer instance to use for predictions.
    squeeze : bool, optional
        If True, squeeze the predictions to remove single-dimensional entries
        from the shape of the predictions (except from first dimension). By
        default False

    Returns
    -------
    np.ndarray
        The predictions as a numpy array.
    """
    list_of_predicted_batches = trainer.predict(model, data_module)
    predictions = torch.cat(list_of_predicted_batches, dim=0)  # type: ignore
    predictions_dimensions = list(range(len(predictions.shape)))
    if squeeze and len(predictions_dimensions) > 1:
        # Squeeze all except the first dimension (batch dimension)
        predictions_dimensions = tuple(predictions_dimensions[1:])
        predictions = predictions.squeeze(predictions_dimensions)
    predictions = predictions.float().cpu().numpy()
    return predictions


def perform_evaluation(
    evaluation_metrics: Dict[str, torchmetrics.Metric],
    data_module: MinervaDataModule,
    predictions: np.ndarray,
    argmax_axis: Optional[int] = None,
    per_sample: bool = False,
    batch_size: int = 1,
    device: str = "cpu",
) -> pd.DataFrame:
    """Evaluates predictions using provided evaluation metrics and a data module

    This function compares predicted values against ground truth labels from
    a prediction dataset. It supports both aggregate evaluation over the entire
    dataset and per-sample evaluation. Metrics should be compatible with
    `torchmetrics`.

    Parameters
    ----------
    evaluation_metrics : dict of str to torchmetrics.Metric
        A dictionary mapping metric names to `torchmetrics.Metric` instances.

    data_module : MinervaDataModule
        A data module that contains the `predict_dataset` used for evaluation.

    predictions : np.ndarray
        An array of predictions generated by the model.

    argmax_axis : int, optional
        If provided, applies `torch.argmax` along this axis to the predictions
        before metric evaluation.

    per_sample : bool, default=False
        If True, computes metrics individually for each sample. Otherwise,
        evaluates metrics over the entire dataset in batches.

    batch_size : int, default=1
        Batch size used for evaluation when `per_sample` is False.

    device : str, default="cpu"
        The device (e.g., "cpu", "cuda") on which metric computations will run.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing computed metric values. If `per_sample` is True,
        each row corresponds to one sample. Otherwise, a single-row summary is
        returned.
    """

    dataset = data_module.predict_dataset
    if dataset is None:
        raise ValueError(
            "No predict dataset found in the data module. "
            + "Please provide a predict dataset to perform evaluation."
        )

    for metric_name, metric in evaluation_metrics.items():
        if not isinstance(metric, torchmetrics.Metric):
            raise ValueError(
                f"Metric {metric_name} is not a valid torchmetrics.Metric."
            )
        metric.to(device)

    if not per_sample:
        # ys -> numpy array -> torch tensor (to keep dtype)
        y = np.array([dataset[i][1] for i in range(len(dataset))])  # type: ignore
        y = torch.from_numpy(y)
        y_hat = (
            torch.from_numpy(predictions)
            if isinstance(predictions, np.ndarray)
            else predictions
        )
        # Perform argmax if needed
        if argmax_axis is not None:
            y_hat = torch.argmax(y_hat, dim=argmax_axis)

        # Compute metrics
        y_dataloader = torch.utils.data.DataLoader(
            y,  # type: ignore
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        y_hat_dataloader = torch.utils.data.DataLoader(
            y_hat,  # type: ignore
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        for y_batch, y_hat_batch in tqdm.tqdm(
            zip(y_dataloader, y_hat_dataloader),
            desc="Computing metrics",
            total=len(y_dataloader),
        ):
            y_batch = y_batch.to(device)
            y_hat_batch = y_hat_batch.to(device)
            for metric in evaluation_metrics.values():
                metric.update(y_hat_batch, y_batch)

        # Compute all metrics
        result_metrics = {"sample": "all"}
        for name, metric in evaluation_metrics.items():
            result_metrics[name] = metric.compute().item()
            metric.reset()

        # Convert to DataFrame
        result_metrics_df = pd.DataFrame([result_metrics])
        return result_metrics_df

    else:
        results = []
        for i in tqdm.tqdm(
            range(len(predictions)),
            total=len(predictions),
            desc="Computing metrics (per sample)",
        ):
            result = {"sample": i}
            _, label = dataset[i]  # type: ignore
            prediction = predictions[i]

            if isinstance(label, np.ndarray):
                y = torch.from_numpy(label)
            elif isinstance(label, torch.Tensor):
                y = label
            # int, float, bool or str
            else:
                y = torch.from_numpy(np.array(label))

            if isinstance(prediction, np.ndarray):
                y_hat = torch.from_numpy(prediction)
            elif isinstance(prediction, torch.Tensor):
                y_hat = prediction
            # int, float, bool or str
            else:
                y_hat = torch.from_numpy(np.array(prediction))

            # Perform argmax if needed
            if argmax_axis is not None:
                y_hat = torch.argmax(y_hat.unsqueeze(0), dim=argmax_axis)

            y, y_hat = y.squeeze().unsqueeze(0), y_hat.squeeze().unsqueeze(0)
            y, y_hat = y.to(device), y_hat.to(device)

            # Compute metrics
            for name, metric in evaluation_metrics.items():
                metric.update(y_hat, y)
                result[name] = metric.compute().item()
                metric.reset()

            results.append(result)

        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        return results_df


class Experiment(Pipeline):
    NUM_DEBUG_EPOCHS = 3
    NUM_DEBUG_BATCHES = 10

    def __init__(
        self,
        # Base parameters
        experiment_name: str,
        model_config: ModelConfig,
        data_module: MinervaDataModule,
        # Logging and checkpointing parameters
        pretrained_backbone_ckpt_path: Optional[PathLike] = None,
        root_log_dir: PathLike = "./logs",
        execution_id: Union[str, int] = 0,
        checkpoint_metrics: Optional[List[Dict[str, str]]] = None,
        # Trainer-related parameters
        max_epochs: int = 100,
        accelerator: str = "gpu",
        devices: Optional[Union[int, list[int], str]] = 1,
        strategy: str = "auto",
        num_nodes: int = 1,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        limit_predict_batches: Optional[Union[int, float]] = None,
        # Prediction parameters
        evaluation_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        per_sample_evaluation_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        # Other parameters
        seed: Optional[int] = None,
        progress_bar_refresh_rate: int = 1,
        profiler: Optional[str] = None,
        save_predictions: bool = True,
        save_results: bool = True,
        add_last_checkpoint: bool = True,
    ):
        """An experiment is a pipeline that contains all the parameters needed
        to train and evaluate a model, as well as to manage the logging,
        checkpointing, prediction, and results processes in a coherent way.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment. This name will be used to create a
            directory for the experiment in the log directory.
        model_config : ModelConfig
            The model configuration. This object contains the model instantiator
            and the model information.
        data_module : MinervaDataModule
            The data module. This object contains the training, validation, and
            test datasets, as well as the data loaders. For now, datasets must
            return a 2 element tuple (input, label) for each sample.
        pretrained_backbone_ckpt_path : Optional[PathLike], optional
            The path to the pretrained backbone checkpoint. This is used to
            finetune the model. If None, the model will be trained from
            scratch. This parameter handles the lazy instantiation of the model
            and calls `create_model_and_load_backbone` method of the model
            instantiator if `pretrained_backbone_ckpt_path` is not None or
            `create_model_randomly_initialized` method if it is None. By
            default None
        root_log_dir : PathLike, optional
            Root directory for logging and checkpoints. This directory will be
            used to create a subdirectory for the experiment. By default ./logs
        execution_id : Union[str, int], optional
            The execution ID for the experiment. This ID will be used to create
            a subdirectory for the experiment in the log directory. This is
            useful when running the experiment multiple times with the same
            parameters. By default 0
        checkpoint_metrics : Optional[List[Dict[str, str]]], optional
            The checkpoint metrics. This is a list of dictionaries that contain
            the checkpoint metrics. Each dictionary must contain the keys
            "monitor", "mode", and "filename". The "monitor" key is the name of
            the metric to monitor, the "mode" key is the mode of the metric
            ("min" or "max"), and the "filename" key is the name of the
            checkpoint file. The "monitor" key can be None if the checkpoint is
            the last one. By default None
        max_epochs : int, optional
            Number of epochs to train the model. This parameter is passed to the
            `get_trainer` function. By default 100.
        accelerator : str, optional
            The accelerator to use for training. This parameter is passed to the
            `get_trainer` function. By default "gpu". Possible values are
            "cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto". If "auto" is
            selected, the accelerator will be automatically selected based on
            the available hardware. By default "gpu"
        devices : Optional[Union[int, list[int], str]], optional
            Number of accelerators to use for training. This parameter is
            passed to the `get_trainer` function. By default 1.
        strategy : str, optional
            Strategy to use for distributed training. This parameter is passed
            to the `get_trainer` function. By default "auto".
        num_nodes : int, optional
            Number of nodes to use for distributed training. This parameter is
            passed to the `get_trainer` function. By default 1.
        limit_train_batches : Optional[Union[int, float]], optional
            Limit the number of training batches to use. This parameter is
            passed to the `get_trainer` function. By default None. If None, all
            batches will be used. If an integer is provided, it will be the
            absolute number of batches. If a float is provided, it will be the
            fraction of the total number of batches. For example, 0.1 means 10%
            of the training batches will be used.
        limit_val_batches : Optional[Union[int, float]], optional
            Limit the number of validation batches to use. This parameter is
            passed to the `get_trainer` function. By default None. If None, all
            batches will be used. If an integer is provided, it will be the
            absolute number of batches. If a float is provided, it will be the
            fraction of the total number of batches. For example, 0.1 means 10%
            of the validation batches will be used.
        limit_test_batches : Optional[Union[int, float]], optional
            Limit the number of test batches to use. This parameter is
            passed to the `get_trainer` function. By default None. If None, all
            batches will be used. If an integer is provided, it will be the
            absolute number of batches. If a float is provided, it will be the
            fraction of the total number of batches. For example, 0.1 means 10%
            of the test batches will be used.
        limit_predict_batches : Optional[Union[int, float]], optional
            Limit the number of prediction batches to use. This parameter is
            passed to the `get_trainer` function. By default None. If None, all
            batches will be used. If an integer is provided, it will be the
            absolute number of batches. If a float is provided, it will be the
            fraction of the total number of batches. For example, 0.1 means 10%
            of the prediction batches will be used.
        evaluation_metrics : Optional[Dict[str, torchmetrics.Metric]], optional
            A dictionary of evaluation metrics to use for the predictions. The
            keys are the names of the metrics and the values are the
            `torchmetrics.Metric` objects. These metrics are calculated using
            all the predictions. By default None.
        per_sample_evaluation_metrics : Optional[ Dict[str, torchmetrics.Metric] ], optional
            A dictionary of evaluation metrics to use for the predictions. The
            keys are the names of the metrics and the values are the
            `torchmetrics.Metric` objects. These metrics are calculated using
            each prediction separately, that is, applyied per sample. By
            default None.
        seed : Optional[int], optional
            The seed to use for the experiment, by default None
        progress_bar_refresh_rate : int, optional
            The refresh rate of the progress bar (in batches). If 0, the
            progress bar is disabled. If 1, the progress bar is updated every
            batch. By default 1
        profiler : Optional[str], optional
            A profiler to use for the experiment. This parameter is passed to
            the `get_trainer` function. By default None.
        save_predictions : bool, optional
            If True, the predictions will be saved to the log directory. By
            default True
        save_results : bool, optional
            If True, the results will be saved to the log directory. By
            default True
        add_last_checkpoint : bool, optional
            If True, the last checkpoint will be added to the list of checkpoint
            metrics. By default True.

        Raises
        ------
        ValueError
            If the checkpoint metrics are not valid or do not contain the
            required keys.

        Notes
        ------
        - This class assumes that the `MinervaDataModule` class returns a
            (input, label) tuple for each sample in the dataset. The input is
            the data and the label is the ground truth/target.
        """
        # ------- Base parameters -------
        self.experiment_name = experiment_name
        self.model_config = model_config
        self.data_module = data_module

        # ------- Logging and checkpointing parameters -------
        self.pretrained_backbone_ckpt_path = pretrained_backbone_ckpt_path
        self.root_log_dir = Path(root_log_dir)
        self.execution_id = str(execution_id)
        self.checkpoint_metrics = checkpoint_metrics or []
        # Check if checkpoint metrics are valid
        for ckpt_metric in self.checkpoint_metrics:
            if not isinstance(ckpt_metric, dict):
                raise ValueError("Checkpoint metric must be a dictionary.")
            for key in ["monitor", "mode", "filename"]:
                if key not in ckpt_metric:
                    raise ValueError(f"Checkpoint metric must contain a '{key}' key.")
        # Add the "last" checkpoint metric if not already present
        if add_last_checkpoint:
            if not any(
                ckpt_metric.get("filename") == "last"
                for ckpt_metric in self.checkpoint_metrics
            ):
                self.checkpoint_metrics.append(
                    {"monitor": None, "mode": "min", "filename": "last"}  # type: ignore
                )

        # -------  Trainer-related parameters -------
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.strategy = strategy
        self.num_nodes = num_nodes
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.limit_test_batches = limit_test_batches
        self.limit_predict_batches = limit_predict_batches

        # ------- Prediction parameters -------
        self.evaluation_metrics = evaluation_metrics or {}
        self.per_sample_evaluation_metrics = per_sample_evaluation_metrics or {}

        # ------- Other parameters -------
        self.seed = seed
        self.progress_bar_refresh_rate = progress_bar_refresh_rate
        self.profiler = profiler
        self.save_predictions = save_predictions
        self.save_results = save_results

        # ------- Initialize the pipeline -------
        log_dir = (
            self.root_log_dir
            / self.experiment_name
            / self.data_module.dataset_name
            / self.model_config.information.name
            / self.execution_id
        )

        super().__init__(
            log_dir=log_dir,
            cache_result=False,
            save_run_status=False,
            seed=seed,
            ignore=["model_config", "data_module"],
        )

        self._checkpoint_dir = log_dir / "checkpoints"
        self._predictions_dir = log_dir / "predictions"
        self._results_dir = log_dir / "results"
        self._training_metrics_path = log_dir / "metrics.csv"

    # ------------ Acessors ------------
    # Here we have acess to:
    # - checkpoint paths
    # - metrics and metrics path
    # - prediction paths
    # - results and results path

    @property
    def checkpoint_paths(self) -> Dict[str, Path]:
        """Returns a dictionary of checkpoint paths for the experiment.

        The keys are the checkpoint names, and the values are the corresponding
        paths to the checkpoints.

        Returns
        -------
        Dict[str, Path]
            A dictionary mapping checkpoint names to their respective paths.
        """
        return {p.stem: p for p in self._checkpoint_dir.glob("*.ckpt") if p.is_file()}

    @property
    def training_metrics_path(self) -> Optional[Path]:
        """The path to the training metrics file.

        Returns
        -------
        Optional[Path]
            The path to the metrics file if it exists, otherwise None.
        """
        if self._training_metrics_path.is_file():
            return self._training_metrics_path
        return None

    @property
    def training_metrics(self) -> Optional[pd.DataFrame]:
        """Returns the training metrics as a pandas DataFrame.
        If the metrics file does not exist, returns None.

        Returns
        -------
        Optional[pd.DataFrame]
            A DataFrame containing the training metrics.
        """
        # Check if the metrics file exists and is a file
        path = self.training_metrics_path
        if path:
            return pd.read_csv(self._training_metrics_path)
        else:
            return None

    @property
    def prediction_paths(self) -> Dict[str, Path]:
        """Returns a dictionary of prediction paths for the experiment.

        The keys are the prediction names, and the values are the corresponding
        paths to the predictions.

        Returns
        -------
        Dict[str, Path]
            A dictionary mapping prediction names to their respective paths.
        """
        return {p.stem: p for p in self._predictions_dir.glob("*.npy") if p.is_file()}

    def load_predictions_of_ckpt(self, name: str) -> np.ndarray:
        """Load predictions from a file.

        Parameters
        ----------
        name : str
            The name of the prediction file (without extension).

        Returns
        -------
        np.ndarray
            The loaded predictions as a numpy array.
        """
        try:
            path = self.prediction_paths[name]
            return load_predictions(path)
        except KeyError:
            raise Exception(
                f"Prediction file '{name}' not found in {self._predictions_dir}"
            )

    @property
    def results_paths(self) -> Dict[str, Path]:
        """Returns a dictionary of results paths for the experiment.

        The keys are the result names, and the values are the corresponding
        paths to the results.

        Returns
        -------
        Dict[str, Path]
            A dictionary mapping result names to their respective paths.
        """
        return {p.stem: p for p in self._results_dir.glob("*.csv") if p.is_file()}

    def load_results_of_ckpt(self, name: str) -> pd.DataFrame:
        """Load results from a file.

        Parameters
        ----------
        name : str
            The name of the result file (without extension).

        Returns
        -------
        pd.DataFrame
            The loaded results as a pandas DataFrame.
        """
        try:
            path = self.results_paths[name]
            return load_results(path)
        except KeyError:
            raise Exception(f"Results file '{name}' not found in {self._results_dir}")

    # ---------- Trainer ---------

    def _trainer_parameters(
        self, enable_logging: bool = True, debug: bool = False
    ) -> Dict[str, Any]:
        """Return the parameters for the trainer based on the current on debug
        and logging settings.

        Parameters
        ----------
        enable_logging : bool, optional
            If True, logging will be enabled, by default True
        debug : bool, optional
            If True,  model will be trained with a few batches and for a few
            epochs only. Logging will always be disabled, by default False

        Returns
        -------
        Dict[str, Any]
            All the parameters for the `get_trainer` function.
        """
        return {
            "log_dir": self.log_dir,
            "max_epochs": self.NUM_DEBUG_EPOCHS if debug else self.max_epochs,
            "limit_train_batches": (
                self.NUM_DEBUG_BATCHES if debug else self.limit_train_batches
            ),
            "limit_val_batches": (
                self.NUM_DEBUG_BATCHES if debug else self.limit_val_batches
            ),
            "limit_test_batches": (
                self.NUM_DEBUG_BATCHES if debug else self.limit_test_batches
            ),
            "limit_predict_batches": (
                self.NUM_DEBUG_BATCHES if debug else self.limit_predict_batches
            ),
            "accelerator": self.accelerator,
            "strategy": self.strategy,
            "devices": self.devices,
            "num_nodes": self.num_nodes,
            "progress_bar_refresh_rate": self.progress_bar_refresh_rate,
            "enable_logging": False if debug else enable_logging,
            "checkpoint_metrics": None if debug else self.checkpoint_metrics,
            "precision": "32-true",
            "deterministic": False,
            "benchmark": True,
            "profiler": self.profiler,
        }

    # ---------- FIT Experiment methods ---------
    @staticmethod
    def __typing_string(value):
        if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
            return f"shape={tuple(value.shape)}"
        else:
            return f"scalar with type={type(value).__name__}"

    def _print_train_summary(
        self,
        model: L.LightningModule,
        trainer_params: Dict[str, Any],
        debug: bool = False,
        resume_from_ckpt: Optional[str] = None,
    ) -> None:

        print("\n" + "=" * 80)
        print(
            f"Experiment: {self.experiment_name} {'(DEBUG)' if debug else ''}".center(
                80
            )
        )
        print("=" * 80)

        #  ------------ Model info ------------
        finetune_backbone = self.pretrained_backbone_ckpt_path is not None
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("🧠 Model")
        print(f"   ├── Name: {self.model_config.information.name}")
        print(f"   ├── Finetune: {'Yes' if finetune_backbone else 'No'}")
        if finetune_backbone:
            print(
                f"   |    └── Pretrained Backbone Checkpoint: {self.pretrained_backbone_ckpt_path}"
            )
        print(f"   ├── Resumed From: {resume_from_ckpt or 'Beginning'}")
        print(
            f"   ├── Expected Input Shape: {self.model_config.information.input_shape}"
        )
        print(
            f"   ├── Expected Output Shape: {self.model_config.information.output_shape}"
        )
        print(f"   ├── Total Params: {total_params:,}")
        try:
            print(
                f"   └── Trainable Params: {trainable_params:,} ({trainable_params / total_params:.2%})"
            )
        except ZeroDivisionError:
            print("   └── Trainable Params: 0 (0.00%)")

        # ------------ Dataset info ------------
        print("\n📊 Dataset")
        train_data = self.data_module.train_dataset
        val_data = self.data_module.val_dataset
        if train_data:
            x, y = train_data[0]
            print(f"   ├── Train Samples: {len(train_data)}")
            print(f"   |   ├── Input Shape: {self.__typing_string(x)}")
            print(f"   |   └── Label Shape: {self.__typing_string(y)}")
        else:
            print("   ├── Train Dataset: None")

        if val_data:
            x, y = val_data[0]
            print(f"   └── Validation Samples: {len(val_data)}")
            print(f"       ├── Input Shape: {self.__typing_string(x)}")
            print(f"       └── Label Shape: {self.__typing_string(y)}")
        else:
            print("   └── Validation Dataset: None")

        # ------------ Logging & checkpoints ------------
        ckpt_filenames = ", ".join(
            [
                f"{m['filename']}.ckpt"
                for m in self.checkpoint_metrics
                if m.get("filename")
            ]
        )
        checkpoints_exist = self.checkpoint_paths

        print("\n💾 Logging & Checkpoints")
        print(f"   ├── Log Dir: {self.log_dir}")
        print(f"   ├── Metrics Path: {self._training_metrics_path}")
        print(f"   └── Checkpoints Dir: {self._checkpoint_dir}")
        if checkpoints_exist:
            print(f"       ├── Files: {ckpt_filenames or 'None'}")
            print(f"       └── ⚠️ Existing checkpoints found! It will be overwritten!")
        else:
            print(f"       └── Files: {ckpt_filenames or 'None'}")

        # ------------ Trainer configuration ------------
        print("\n⚙️ Trainer Config")
        print(f"   ├── Max Epochs: {trainer_params['max_epochs']}")
        print(f"   ├── Train Batches: {trainer_params['limit_train_batches']}")
        print(f"   ├── Accelerator: {trainer_params['accelerator']}")
        print(f"   ├── Strategy: {trainer_params['strategy']}")
        print(f"   ├── Devices: {trainer_params['devices']}")
        print(f"   ├── Num Nodes: {trainer_params['num_nodes']}")
        print(f"   └── Seed: {self.seed}")

    def _train_model(
        self,
        resume_from_ckpt: Optional[str] = None,
        debug: bool = False,
        print_summary: bool = True,
    ) -> Dict[str, Any]:
        data_module = self.data_module

        # If pre-trained backbone is provided, load the model with the
        # pre-trained backbone (usually for finetuning)
        if self.pretrained_backbone_ckpt_path is not None:
            model = self.model_config.instantiator.create_model_and_load_backbone(
                self.pretrained_backbone_ckpt_path
            )
        # If no pre-trained backbone is provided, create a model with
        # randomly initialized backbone and head (full supervised training)
        else:
            model = self.model_config.instantiator.create_model_randomly_initialized()

        # Get the trainer
        trainer_params = self._trainer_parameters(
            enable_logging=True,
            debug=debug,
        )
        trainer = get_trainer(**trainer_params)

        # Check if need to resume from a checkpoint
        checkpoints = self.checkpoint_paths
        if resume_from_ckpt:
            if resume_from_ckpt not in checkpoints:
                raise ValueError(
                    f"Checkpoint '{resume_from_ckpt}' not found in {list(checkpoints.keys())}"
                )
            resume_from_ckpt = checkpoints[resume_from_ckpt]  # type: ignore

        # Print the training summary
        if print_summary:
            self._print_train_summary(
                model=model,
                trainer_params=trainer_params,
                debug=debug,
                resume_from_ckpt=resume_from_ckpt,
            )

        # Train the model
        perform_train(
            data_module=data_module,
            model=model,
            trainer=trainer,
            resume_from_ckpt=resume_from_ckpt,
        )

        return {
            "data_module": data_module,
            "model": model,
            "trainer": trainer,
            "log_dir": self.log_dir,
            "metrics_path": self._training_metrics_path,
            "checkpoints": self.checkpoint_paths,
        }

    # ---------- EVALUATE Experiment methods ---------
    def _print_evaluation_summary(
        self,
        trainer_params: Dict[str, Any],
        debug: bool = False,
        ckpt_path: Optional[PathLike] = None,
        predictions_path: Optional[PathLike] = None,
        results_path: Optional[PathLike] = None,
    ) -> None:
        print("\n" + "=" * 80)
        print(
            f"Evaluation: {self.experiment_name} ({ckpt_path.name}) {'(DEBUG)' if debug else ''}".center(
                80
            )
        )
        print("=" * 80)

        # ------------ Checkpoint Info ------------
        print("💾 Checkpoint")
        print(f"   ├── Checkpoint Path: {ckpt_path}")
        print(f"   └── Predictions Path: {predictions_path or 'Not saved'}")

        # ------------ Dataset Info ------------
        print("\n📊 Dataset")
        predict_data = self.data_module.predict_dataset
        if predict_data:
            x, y = predict_data[0]
            print(f"   ├── Predict Samples: {len(predict_data)}")
            print(f"   ├── Input: {self.__typing_string(x)}")
            print(f"   └── Label: {self.__typing_string(y)}")
        else:
            print("   └── Predict Dataset: None")

        # ------------ Evaluation Metrics ------------
        print("\n📈 Evaluation Metrics")
        if self.evaluation_metrics or self.per_sample_evaluation_metrics:
            for name, metric in self.evaluation_metrics.items():
                print(f"   ├── {name}: {metric.__class__.__name__}")
            for name, metric in self.per_sample_evaluation_metrics.items():
                print(f"   ├── {name}: {metric.__class__.__name__} (PER_SAMPLE)")
        else:
            print("   └── No evaluation metrics provided.")

        # ------------ Trainer Configuration ------------
        print("\n⚙️ Trainer Config")
        print(f"   ├── Max Epochs: {trainer_params['max_epochs']}")
        print(f"   ├── Predict Batches: {trainer_params['limit_predict_batches']}")
        print(f"   ├── Accelerator: {trainer_params['accelerator']}")
        print(f"   ├── Strategy: {trainer_params['strategy']}")
        print(f"   ├── Devices: {trainer_params['devices']}")
        print(f"   ├── Num Nodes: {trainer_params['num_nodes']}")
        print(f"   └── Seed: {self.seed}")

    def _evaluate_model(
        self,
        ckpts_to_evaluate: Optional[Union[str, List[str]]] = None,
        print_summary: bool = True,
        debug: bool = False,
    ):
        # --------- Checkpoints -------
        checkpoints_to_use = self.checkpoint_paths
        # Check which checkpoints to evaluate (else, evaluate all)
        if ckpts_to_evaluate is not None:
            if isinstance(ckpts_to_evaluate, str):
                ckpts_to_evaluate = [ckpts_to_evaluate]

            try:
                checkpoints_to_use = {
                    ckpt: checkpoints_to_use[ckpt] for ckpt in ckpts_to_evaluate
                }
            except KeyError as e:
                raise ValueError(f"Checkpoint {e} not found in {checkpoints_to_use}")
        # Check if any checkpoint is found
        if len(checkpoints_to_use) == 0:
            raise ValueError(f"No checkpoints found in {self._checkpoint_dir}")

        # --------- Dataset -------
        checkpoint_results = {}
        data_module = self.data_module
        if data_module.predict_dataset is None:
            raise ValueError(
                "No predict dataset found in the data module. Please provide a predict dataset to perform evaluation."
            )

        for ckpt_name, ckpt_path in checkpoints_to_use.items():
            predictions_file = None
            results_filename = None
            results_filename_per_sample = None
            results = None
            per_sample_results = None

            if self.save_predictions and not debug:
                predictions_file = self._predictions_dir / f"{ckpt_name}.npy"

            if self.save_results and not debug:
                results_filename = self._results_dir / f"{ckpt_name}.csv"
                results_filename_per_sample = (
                    self._results_dir / f"{ckpt_name}_per_sample.csv"
                )

            # Load the model from the checkpoint
            model = self.model_config.instantiator.load_model_from_checkpoint(ckpt_path)

            # Trainer
            trainer_params = self._trainer_parameters(
                enable_logging=False,
                debug=debug,
            )
            trainer = get_trainer(**trainer_params)

            if print_summary:
                self._print_evaluation_summary(
                    trainer_params=trainer_params,
                    debug=debug,
                    ckpt_path=ckpt_path,
                    predictions_path=predictions_file,
                    results_path=results_filename,
                )

            # Perform prediction
            predictions = perform_predict(
                data_module=data_module,
                model=model,
                trainer=trainer,
            )

            if predictions_file is not None:
                save_predictions(predictions, predictions_file)
            else:
                print("Predictions not saved...")

            # Perform evaluation
            if self.evaluation_metrics:
                results = perform_evaluation(
                    evaluation_metrics=self.evaluation_metrics,
                    data_module=data_module,
                    predictions=predictions,
                    argmax_axis=(
                        1 if self.model_config.information.return_logits else None
                    ),
                    per_sample=False,
                    batch_size=self.data_module._predict_dataloader_kwargs[
                        "batch_size"
                    ],
                    device="cuda" if self.accelerator == "gpu" else "cpu",
                )

                if results_filename is not None:
                    save_results(results, results_filename, index=False)
                else:
                    print(f"Results not saved...")
            else:
                print("No evaluation metrics provided. Skipping evaluation.")

            # Perform per-sample evaluation
            if self.per_sample_evaluation_metrics:
                per_sample_results = perform_evaluation(
                    evaluation_metrics=self.per_sample_evaluation_metrics,
                    data_module=data_module,
                    predictions=predictions,
                    argmax_axis=(
                        1 if self.model_config.information.return_logits else None
                    ),
                    per_sample=True,
                    batch_size=self.data_module._predict_dataloader_kwargs[
                        "batch_size"
                    ],
                    device="cuda" if self.accelerator == "gpu" else "cpu",
                )

                if results_filename_per_sample is not None:
                    save_results(
                        per_sample_results,
                        results_filename_per_sample,
                        index=False,
                    )
                else:
                    print(f"Results not saved...")
            else:
                print(
                    "No per-sample evaluation metrics provided. Skipping per-sample evaluation."
                )

            # Store the results
            checkpoint_results[ckpt_name] = {
                "predictions_path": predictions_file,
                "results_path": results_filename,
                "results_path_per_sample": results_filename_per_sample,
                "results": results,
                "results_per_sample": per_sample_results,
            }

            print(f"Checkpoint {ckpt_name} evaluated!")

        return checkpoint_results

    # ---------- Default pipeline entrypoints and other methods ---------
    def _run(
        self,
        task: str,
        debug: bool = False,
        resume_from_ckpt: Optional[str] = None,
        print_summary: bool = True,
        ckpts_to_evaluate: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        if task == "fit":
            return self._train_model(
                resume_from_ckpt=resume_from_ckpt,
                print_summary=print_summary,
                debug=debug,
            )
        elif task == "evaluate":
            return self._evaluate_model(
                ckpts_to_evaluate=ckpts_to_evaluate,
                print_summary=print_summary,
                debug=debug,
            )
        elif task == "fit-evaluate":
            # Train the model
            self._train_model(
                resume_from_ckpt=resume_from_ckpt,
                print_summary=print_summary,
                debug=debug,
            )
            # Evaluate the model
            eval_results = self._evaluate_model(
                ckpts_to_evaluate=ckpts_to_evaluate,
                print_summary=print_summary,
                debug=debug,
            )
            return eval_results
        else:
            raise ValueError(
                f"Unknown task '{task}'. Supported tasks are: 'fit', 'evaluate', or 'fit-evaluate'"
            )

    def cleanup(self):
        """Clean up the experiment by removing the log directory."""
        if self.log_dir.exists():
            shutil.rmtree(self.log_dir)
            print(f"Experiment at '{self.log_dir}' cleaned up.")
        else:
            print(f"Experiment at '{self.log_dir}' not found.")

    @property
    def status(self) -> Dict[str, Any]:
        d = {}
        d["experiment_name"] = self.experiment_name
        d["log_dir"] = self.log_dir
        d["checkpoints"] = self.checkpoint_paths
        d["training_metrics"] = self.training_metrics_path
        d["prediction_paths"] = self.prediction_paths
        d["results_paths"] = self.results_paths

        state = "not executed"
        if len(d["checkpoints"]) > 0:
            state = "executed"

        if len(d["prediction_paths"]) > 0:
            state = "predicted"

        if len(d["results_paths"]) > 0:
            state = "evaluated"

        d["state"] = state

        return d

    # ---------- Python methods ---------

    def __str__(self) -> str:
        def indent_text(text, spaces=6):
            """Indent each line of a string by a given number of spaces."""
            if not text:
                return "No data."
            return "\n".join(
                " " * spaces + line if line.strip() else line
                for line in text.split("\n")
            )

        exp_name = f"🚀 Experiment: {self.experiment_name} 🚀"
        pretrained_backbone = (
            self.pretrained_backbone_ckpt_path
            if self.pretrained_backbone_ckpt_path
            else "FROM SCRATCH"
        )

        limit_batches = "Limit batches: "

        return (
            f"{'=' * 80}\n"
            f"{' ' * ((80 - len(exp_name)) // 2)}{exp_name}\n"
            f"{'=' * 80}\n"
            f"\n🛠 Execution Details\n"
            f"   ├── Execution ID: {self.execution_id}\n"
            f"   ├── Log Dir: {self.log_dir}\n"
            f"   ├── Seed: {self.seed}\n"
            f"   ├── Accelerator: {self.accelerator}\n"
            f"   ├── Devices: {self.devices}\n"
            f"   ├── Max Epochs: {self.max_epochs}\n"
            f"   ├── Train Batches: {self.limit_train_batches or 'all'}\n"
            f"   ├── Val Batches: {self.limit_val_batches or 'all'}\n"
            f"   └── Test Batches: {self.limit_test_batches or 'all'}\n"
            f"\n"
            # f"{'=' * 50}\n"
            f"🧠 Model Information\n"
            f"   ├── Model Name: {self.model_config.information.name}\n"
            f"   ├── Pretrained Backbone: {pretrained_backbone}\n"
            f"   ├── Input Shape: {self.model_config.information.input_shape}\n"
            f"   ├── Output Shape: {self.model_config.information.output_shape}\n"
            f"   └── Num Classes: {self.model_config.information.num_classes}\n"
            f"\n📂 Dataset Information\n"
            f"{indent_text(str(self.data_module), spaces=6)}\n"
        )
