from sslt.pipelines.cli import auto_main


import os
from typing import Tuple
import lightning as L

from sslt.data.data_modules.F3_attribute import F3AttributeDataModule
from sslt.models.nets.wisenet import WiseNet


from sslt.pipelines.base import parametrizable

from sslt.pipelines.regression.pipelines.train import LightningTrain
from sslt.pipelines.regression.pipelines.evaluate import RegressorEvaluator


@parametrizable
class WisenetTrain(LightningTrain):
    """
    Class for training a U-Net model using Lightning framework.
    """

    MODEL = "WiseNet"
    DATASET = "F3"

    def __init__(
        self,
        experiment_name: str,
        original_path: str,
        attribute_path: str,
        
        n_channels: int = 1,
        learning_rate: float = 1e-3,
        num_workers: int = None,
        data_shape: Tuple[int, int, int] = (17, 128, 128),
        model_name: str = MODEL,
        dataset_name: str = DATASET,
        
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
        log_dir: str = "./runs",
        stage: str = "train",
        
    ):
        """Class for training a U-Net model using Lightning framework.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment.
        original_path : str
            The path to the original ZARR data.
        attribute_path : str
            The path to the attribute ZARR data.
        n_channels : int, optional
            Number of channels, by default 1
        bilinear : bool, optional
            If True, use bilinear interpolation, by default False
        learning_rate : float, optional
            The learning rate for Adam optimizer, by default 1e-3
        num_workers : int, optional
            Number of workrs to parallel load data, by default None. If None
            the number of workers is set to the number of CPUs.
        data_shape : Tuple[int, int, int], optional
            The shape of each patch, by default (1, 128, 128)
        model_name : str, optional
            Name of the model, by default "Unet"
        dataset_name : str, optional
            Name of the dataset, by default "F3"
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
        stage : str, optional
            The stage, by default "train"
        """
        super().__init__(
            experiment_name=experiment_name,
            model_name=model_name,
            dataset_name=dataset_name,
            run_name=run_name,
            accelerator=accelerator,
            devices=devices,
            num_nodes=num_nodes,
            strategy=strategy,
            max_epochs=max_epochs,
            batch_size=batch_size,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            checkpoint_monitor_metric=checkpoint_monitor_metric,
            checkpoint_monitor_mode=checkpoint_monitor_mode,
            patience=patience,
            log_dir=log_dir,
            stage=stage,
        )
        self.n_channels = n_channels
        self.learning_rate = learning_rate
        self.original_path = original_path
        self.attribute_path = attribute_path
        self.data_shape = data_shape
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )


    def get_model(self) -> L.LightningModule:
        """
        Returns the U-Net model.

        Returns:
            L.LightningModule: The U-Net model.
        """
        return WiseNet(
            in_channels=self.n_channels,
        )

    def get_data_module(self) -> L.LightningDataModule:
        """
        Returns the data module for loading the data.

        Returns:
            L.LightningDataModule: The data module.
        """
        return F3AttributeDataModule(
            original_path=self.original_path,
            attribute_path=self.attribute_path,
            data_shape=self.data_shape,
            num_workers=self.num_workers,
        )


@parametrizable
class WiseNetEvaluate(RegressorEvaluator):
    MODEL = "WiseNet"
    DATASET = "F3"

    def __init__(
        self,
        # Required
        experiment_name: str,
        original_path: str,
        attribute_path: str,
        load_checkpoint_from: str,
        n_channels: int = 1,
        learning_rate: float = 1e-3,
        num_workers: int = None,
        data_shape: Tuple[int, int, int] = (17, 128, 128),
        model_name: str = MODEL,
        dataset_name: str = DATASET,

        # Optional parameters
        run_name: str = None,
        accelerator: str = "cpu",
        devices: int = 1,
        num_nodes: int = 1,
        strategy: str = "auto",
        batch_size: int = 1,
        limit_predict_batches: int | float = 1.0,
        log_dir: str = "./runs",
        stage: str = "evaluate",
    ):
        super().__init__(
            experiment_name=experiment_name,
            model_name=model_name,
            dataset_name=dataset_name,
            load_checkpoint_from=load_checkpoint_from,
            
            run_name=run_name,
            accelerator=accelerator,
            devices=devices,
            num_nodes=num_nodes,
            strategy=strategy,
            batch_size=batch_size,
            limit_predict_batches=limit_predict_batches,
            log_dir=log_dir,
            stage=stage,
        )

        self.n_channels = n_channels
        self.learning_rate = learning_rate
        self.original_path = original_path
        self.attribute_path = attribute_path
        self.data_shape = data_shape
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )

    def get_model(self) -> L.LightningModule:
        """
        Returns the U-Net model.

        Returns:
            L.LightningModule: The U-Net model.
        """
        return WiseNet(
            in_channels=self.n_channels,
        )

    def get_data_module(self) -> L.LightningDataModule:
        """
        Returns the data module for loading the data.

        Returns:
            L.LightningDataModule: The data module.
        """
        return F3AttributeDataModule(
            original_path=self.original_path,
            attribute_path=self.attribute_path,
            data_shape=self.data_shape,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    auto_main({"train": WisenetTrain, "evaluate": WiseNetEvaluate})
