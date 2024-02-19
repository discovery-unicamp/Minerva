from typing import List, Tuple, Dict, final
import lightning as L
from lightning.pytorch.loggers import Logger, MLFlowLogger


from sslt.pipelines.base import parametrizable, Pipeline

from sslt.pipelines.regression.pipelines.utils import (
    get_split_data_loader,
    dataset_from_dataloader,
    compute_reconstruction_metrics,
    reconstruct_from_patches,
)

import torch

import numpy as np
import plotly.graph_objects as go


class RegressorEvaluator(Pipeline):
    def __init__(
        self,
        # Required parameters
        experiment_name: str,
        model_name: str,
        dataset_name: str,
        load_checkpoint_from: str,
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
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.accelerator = accelerator
        self.devices = devices
        self.num_nodes = num_nodes
        self.strategy = strategy
        self.load_checkpoint = load_checkpoint_from
        self.batch_size = batch_size
        self.limit_predict_batches = limit_predict_batches
        self.log_dir = log_dir
        self.stage = stage

    def get_model(self) -> L.LightningModule:
        raise NotImplementedError

    def get_data_module(self) -> L.LightningDataModule:
        raise NotImplementedError

    def get_trainer(
        self, logger: Logger, callbacks: List[L.Callback]
    ) -> L.Trainer:
        return L.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            num_nodes=self.num_nodes,
            strategy=self.strategy,
            logger=logger,
            callbacks=callbacks,
            limit_predict_batches=self.limit_predict_batches,
        )

    def get_callbacks(self) -> List[L.Callback]:
        return []

    def get_logger(self) -> Logger:
        return MLFlowLogger(
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            save_dir=self.log_dir,
            log_model=True,
            tags={
                "model": self.model_name,
                "stage": self.stage,
                "dataset": self.dataset_name,
            },
        )

    def load_model(
        self, model: L.LightningModule, load_path: str = None
    ) -> L.LightningModule:
        state_dict = torch.load(load_path)["state_dict"]
        model.load_state_dict(state_dict)
        return model

    def compute_sample_metrics(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, float]:
        return compute_reconstruction_metrics(y_hat, y)

    def reconstruct_image_from_patches(
        self, patches: List[torch.Tensor], coords: List[Tuple[int, ...]]
    ) -> np.ndarray:
        return reconstruct_from_patches(patches, coords)

    def predict(self, model, dataloader, trainer):
        y_hat = trainer.predict(model, dataloader)
        y_hat = torch.cat(y_hat)
        return y_hat
    
    def _heatmap(self, data):
        fig = go.Figure(data=go.Heatmap(z=data))
        fig.update_layout(
            coloraxis=dict(colorscale="gray"),
        )
        return fig

    def run_task(self, model, datamodule, trainer):
        test_data_loader = get_split_data_loader("test", datamodule)
        test_X, test_y = dataset_from_dataloader(test_data_loader)
        y_hat = self.predict(model, test_data_loader, trainer)
        
        
        ########################################################################
        # Evaluate metrics
        metrics_sum = {}
        num_samples = len(test_y)

        for i, (_y_true, _y_hat) in enumerate(zip(test_y, y_hat)):
            metrics = self.compute_sample_metrics(_y_hat, _y_true)
            trainer.logger.log_metrics(metrics, step=i)
            # Accumulate metrics
            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0) + value

        # Calculate average metrics
        avg_metrics = {
            f"avg_{key}": float(value) / num_samples
            for key, value in metrics_sum.items()
        }
        # Log average metrics
        trainer.logger.log_metrics(avg_metrics)
        
        
        ########################################################################
        # Log images

        # Get indexes for the first axis
        axis_0_indices = sorted(
            list(
                set(
                    coord[0]
                    for coord in test_data_loader.dataset.readers[0].indices
                )
            )
        )
        inline_0_indices = [
            (i, coord)
            for i, coord in enumerate(
                test_data_loader.dataset.readers[0].indices
            )
            if coord[0] == axis_0_indices[0]
        ]
        test_X_0 = self.reconstruct_image_from_patches(
            [test_X[i] for i, coord in inline_0_indices],
            [coord for i, coord in inline_0_indices],
        )[0]


        patch_0 = test_y[0][0].cpu().numpy()
        patch_0_pred = y_hat[0][0].cpu().numpy()
        
        fig = self._heatmap(patch_0)
        trainer.logger.experiment.log_figure(
            trainer.logger.run_id, fig, "label_patch_0.png"
        )

        fig = self._heatmap(patch_0_pred)
        trainer.logger.experiment.log_figure(
            trainer.logger.run_id, fig, "predicted_patch_0.png"
        )
        
        ########################################################################
        return avg_metrics

    def log_hyperparams(self, logger):
        params = dict()
        if hasattr(self, "_parameters"):
            params = self._parameters

        logger.log_hyperparams(params)
        return params

    @final
    def run(self):
        model = self.get_model()
        model = self.load_model(model, self.load_checkpoint)
        datamodule = self.get_data_module()
        logger = self.get_logger()
        callbacks = self.get_callbacks()
        hparams = self.log_hyperparams(logger)
        trainer = self.get_trainer(logger, callbacks)
        return self.run_task(model, datamodule, trainer)
