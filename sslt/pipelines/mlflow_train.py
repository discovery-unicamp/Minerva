from io import BytesIO
from typing import Dict, List, final
import fsspec
import lightning as L
from lightning.pytorch.loggers import Logger, MLFlowLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    EarlyStopping,
    ModelSummary,
)
import torch

from ssl_tools.callbacks.performance import PerformanceLogger
from ssl_tools.callbacks.save_best import PickleBestModelAndLoad
from ssl_tools.pipelines.base import Pipeline
import mlflow
from ssl_tools.models.ssl.classifier import SSLDiscriminator
from ssl_tools.pipelines.utils import load_model_mlflow


class LightningTrainMLFlow(Pipeline):
    def __init__(
        self,
        # Required paramters
        experiment_name: str,
        model_name: str,
        # Optional parameters
        run_name: str = None,
        accelerator: str = "cpu",
        devices: int = 1,
        num_nodes: int = 1,
        strategy: str = "auto",
        max_epochs: int = 1,
        batch_size: int = 1,
        limit_train_batches: int | float = 1.0,
        limit_val_batches: int | float = 1.0,
        checkpoint_monitor_metric: str = None,
        checkpoint_monitor_mode: str = "min",
        patience: int = None,
        log_dir: str = "./mlruns",
        model_tags: Dict[str, str] = None
    ):
        """Train a model using Lightning framework.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment.
        model_name : str
            Name of the model.
        dataset_name : str
            Name of the dataset.
        run_name : str, optional
            The name of the run, by default None
        accelerator : str, optional
            The accelerator to use, by default "cpu"
        devices : int, optional
            Number of accelerators to use, by default 1
        num_nodes : int, optional
            Number of nodes, by default 1
        strategy : str, optional
            Training strategy, by default "auto"
        max_epochs : int, optional
            Maximium number of epochs, by default 1
        batch_size : int, optional
            Batch size, by default 1
        limit_train_batches : int | float, optional
            Limit the number of batches to train, by default 1.0
        limit_val_batches : int | float, optional
            Limit the number of batches to test, by default 1.0
        checkpoint_monitor_metric : str, optional
            The metric to monitor for checkpointing, by default None
        checkpoint_monitor_mode : str, optional
            The mode for checkpointing, by default "min"
        patience : int, optional
            The patience for early stopping, by default None
        log_dir : str, optional
            Location where logs will be saved, by default "./runs"
        """

        super().__init__()
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.run_name = run_name
        self.accelerator = accelerator
        self.devices = devices
        self.num_nodes = num_nodes
        self.strategy = strategy
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.checkpoint_monitor_metric = checkpoint_monitor_metric
        self.checkpoint_monitor_mode = checkpoint_monitor_mode
        self.patience = patience
        self.log_dir = log_dir
        self.model_tags = model_tags
        self._hparams = dict(self.hparams)

    def get_model(self) -> L.LightningModule:
        raise NotImplementedError

    def get_data_module(self) -> L.LightningDataModule:
        raise NotImplementedError

    def get_trainer(
        self, logger: Logger, callacks: List[L.Callback]
    ) -> L.Trainer:
        return L.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            num_nodes=self.num_nodes,
            strategy=self.strategy,
            max_epochs=self.max_epochs,
            logger=logger,
            callbacks=callacks,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
        )

    def get_callbacks(self) -> List[L.Callback]:
        callbacks = []

        model_summary = ModelSummary(max_depth=3)
        callbacks.append(model_summary)

        model_checkpoint = ModelCheckpoint(
            monitor=self.checkpoint_monitor_metric,
            mode=self.checkpoint_monitor_mode,
            # save_last=True
        )
        callbacks.append(model_checkpoint)

        if self.patience:
            early_stopping = EarlyStopping(
                monitor=self.checkpoint_monitor_metric,
                patience=self.patience,
                mode=self.checkpoint_monitor_mode,
            )
            callbacks.append(early_stopping)

        performance_logger = PerformanceLogger()
        callbacks.append(performance_logger)

        # device_stats_monitor = DeviceStatsMonitor()
        # callbacks.append(device_stats_monitor)
        best_logger = PickleBestModelAndLoad(
            model_name=self.model_name,
            model_tags=self.model_tags,
        )
        callbacks.append(best_logger)

        rich_progress_bar = RichProgressBar()
        callbacks.append(rich_progress_bar)

        return callbacks

    def get_logger(self) -> Logger:
        return MLFlowLogger(
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            save_dir=self.log_dir,
            log_model=True,
            tags=self.model_tags
        )

    def run(self) -> L.LightningModule:
        # Get all required components
        model = self.get_model()
        datamodule = self.get_data_module()
        logger = self.get_logger()
        callbacks = self.get_callbacks()
        trainer = self.get_trainer(logger, callbacks)

        # Log the experiment hyperparameters
        logger.log_hyperparams(self._hparams)

        # Do fit and return trained model
        trainer.fit(model, datamodule)
        return model



class LightningFineTuneMLFlow(LightningTrainMLFlow):
    def __init__(
        self,
        registered_model_name: str,
        registered_model_tags: Dict[str, str] = None,
        update_backbone: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.registered_model_name = registered_model_name
        self.registered_model_tags = registered_model_tags
        self.model_version = dict()
        self._mlflow_client = None
        self.update_backbone = update_backbone
        
    @property
    def client(self):
        if self._mlflow_client is None:
            print(f"Initializing MLFlow client at {self.log_dir}...")
            self._mlflow_client = mlflow.client.MlflowClient(
                tracking_uri=self.log_dir
            )
        return self._mlflow_client
    
    def load_model(self) -> L.LightningModule:
        model, model_version = load_model_mlflow(
            self.client,
            self.registered_model_name,
            self.registered_model_tags,
        )
        self.model_version = model_version
        self._hparams.update({"model_version": model_version})
        return model