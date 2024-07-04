from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

import lightning as L
import torch
import yaml
from torchmetrics import Metric

from minerva.pipelines.base import Pipeline
from minerva.utils.data import get_full_data_split
from minerva.utils.typing import PathLike


class SimpleLightningPipeline(Pipeline):
    """Simple pipeline to train, test, predict and evaluate models using Pytorch
    Lightning. This class is intended to be seamlessly integrated with
    jsonargparse CLI.
    """

    def __init__(
        self,
        model: L.LightningModule,
        trainer: L.Trainer,
        log_dir: PathLike = None,
        save_run_status: bool = True,
        classification_metrics: Dict[str, Metric] = None,
        regression_metrics: Dict[str, Metric] = None,
        apply_metrics_per_sample: bool = False,
    ):
        """Train/test/predict/evaluate a Pytorch Lightning model.

        It provides 4 tasks: fit, test, predict and evaluate. The fit task
        trains the model, the test task evaluates the model on the test set, the
        predict task generates predictions for the predict set and the evaluate
        task evaluates the model on the predict set and returns the metrics.

        The evaluate task can calculate classification and regression metrics,
        which is passed as arguments. The metrics are calculated per sample if
        `apply_metrics_per_sample` is True (that generate a metric for each),
        otherwise the metrics are calculated for the whole dataset (single
        metric). The last option is the default.

        Parameters
        ----------
        model : L.LightningModule
            The LightningModule to be used.
        trainer : L.Trainer
            The Lightning Trainer to be used.
        log_dir : PathLike, optional
            The default logging directory where all related pipeline files
            should be saved. By default None (uses current working directory)
        save_run_status : bool, optional
            If True, save the status of each run in a YAML file. This file will
            be saved in the working directory with the name
            `run_{pipeline_id}.yaml`. By default True.
        classification_metrics : Dict[str, Metric], optional
            The classification metrics to be used in the evaluate task. This
            dictionary should have the metric name as key and the
            `torchmetrics.Metric`-like object as value. The metric should be
            able to receive two tensors (y_true, y_pred) and return a tensor
            with the metric value. If None, no classification metrics will be
            calculated. Different from regression, the torch.argmax will be
            applied to the predictions before calculating the metrics.
            By default None.
        regression_metrics : Dict[str, Metric], optional
            The regression metrics to be used in the evaluate task. This
            dictionary should have the metric name as key and the
            `torchmetrics.Metric`-like object as value. The metric should be
            able to receive two tensors (y_true, y_pred) and return a tensor
            with the metric value. If None, no regression metrics will be
            calculated. By default None.
        apply_metrics_per_sample : bool, optional
            Apply the metrics per sample. If True, the metrics will be
            calculated for each sample and the results will be a list of
            metrics. If False, the metrics will be calculated for the whole
            dataset and the results will be a single metric (single-element
            list). By default False
        """
        if log_dir is None and trainer.log_dir is not None:
            log_dir = trainer.log_dir

        super().__init__(
            log_dir=log_dir,
            ignore=[
                "model",
                "trainer",
                "classification_metrics",
                "regression_metrics",
            ],
            cache_result=True,
            save_run_status=save_run_status,
        )
        self._model = model
        self._trainer = trainer
        self._data = None
        self._classification_metrics = classification_metrics
        self._regression_metrics = regression_metrics
        self._apply_metrics_per_sample = apply_metrics_per_sample

    # Public read-only properties
    @property
    def model(self) -> L.LightningModule:
        """The LightningModule used in the pipeline.

        Returns
        -------
        L.LightningModule
            The model used in the pipeline.
        """
        return self._model

    @property
    def trainer(self) -> L.Trainer:
        """The Lightning Trainer used in the pipeline.

        Returns
        -------
        L.Trainer
            The trainer used in the pipeline.
        """
        return self._trainer

    @property
    def data(self) -> L.LightningDataModule:
        """The LightningDataModule used in the last run of the pipeline.

        Returns
        -------
        L.LightningDataModule
            The data used in the last run of the pipeline.
        """
        return self._data

    def _calculate_metrics(
        self, metrics: Dict[str, Metric], y_hat: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, Any]:
        """Calculate the metrics for the given predictions and targets.

        Parameters
        ----------
        metrics : Dict[str, Metric]
            The metrics to be calculated. The dictionary should have the metric
            name as key and the `torchmetrics.Metric`-like object as value.
        y_hat : torch.Tensor
            The predictions tensor.
        y : torch.Tensor
            The targets tensor.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the metric name as key and the list of metric
            values as value. The list will have a single element if
            `apply_metrics_per_sample` is False, otherwise it will have a value.
        """
        results = {}
        if not self._apply_metrics_per_sample:
            y, y_hat = [y], [y_hat]

        for metric_name, metric in metrics.items():
            final_results = [
                metric(y_i, y_hat_i).float().item() for y_i, y_hat_i in zip(y, y_hat)
            ]
            results[metric_name] = final_results

        return results

    # Private methods
    def _fit(self, data: L.LightningDataModule, ckpt_path: Optional[PathLike]):
        """Fit the model using the given data.

        Parameters
        ----------
        data : L.LightningDataModule
            The data module to be used. The data module should have the
            `train_dataloader` method implemented.
        ckpt_path : PathLike
            The checkpoint path to be used. If None, no checkpoint will be used.
        """
        return self._trainer.fit(
            model=self._model, datamodule=data, ckpt_path=ckpt_path
        )

    def _test(self, data: L.LightningDataModule, ckpt_path: Optional[PathLike]):
        """Test the model using the given data.

        Parameters
        ----------
        data : L.LightningDataModule
            The data module to be used. The data module should have the
            `test_dataloader` method implemented.
        ckpt_path : PathLike
            The checkpoint path to be used. If None, no checkpoint will be used.
        """
        return self._trainer.test(
            model=self._model, datamodule=data, ckpt_path=ckpt_path
        )

    def _predict(
        self,
        data: L.LightningDataModule,
        ckpt_path: Optional[PathLike] = None,
    ) -> torch.Tensor:
        """Predict using the given data.

        Parameters
        ----------
        data : L.LightningDataModule
            The data module to be used. The data module should have the
            `predict_dataloader` method implemented.
        ckpt_path : PathLike
            The checkpoint path to be used. If None, no checkpoint will be used.

        Returns
        -------
        torch.Tensor
            The predictions tensor.
        """
        return self._trainer.predict(
            model=self._model, datamodule=data, ckpt_path=ckpt_path
        )

    def _evaluate(
        self,
        data: L.LightningDataModule,
        ckpt_path: Optional[PathLike] = None,
    ) -> Dict[str, Any]:
        """Evaluate the model and calculate regression and/or classification
        metrics.

        Parameters
        ----------
        data : L.LightningDataModule
            The data module to be used. The data module should have the
            `predict_dataloader` method implemented.
        ckpt_path : PathLike
            The checkpoint path to be used. If None, no checkpoint will be used.

        Returns
        -------
        Dict[str, Dict[str, Any]
            A dictionary with metrics.
        """
        metrics = defaultdict(dict)

        X, y = get_full_data_split(data, "predict")
        y = torch.tensor(y, device="cpu")

        y_hat = self.trainer.predict(self._model, datamodule=data, ckpt_path=ckpt_path)
        y_hat = torch.cat(y_hat).detach().cpu()

        if self._classification_metrics is not None:
            y_hat = torch.argmax(y_hat, dim=1)
            metrics["classification"] = self._calculate_metrics(
                self._classification_metrics, y_hat, y
            )

        if self._regression_metrics is not None:
            print(y.shape, y_hat.shape)

            metrics["regression"] = self._calculate_metrics(
                self._regression_metrics, y_hat, y
            )

        metrics = dict(metrics)

        if self._save_pipeline_info:
            yaml_path = self._log_dir / f"metrics_{self.pipeline_id}.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(metrics, f)
                print(f"Metrics saved to {yaml_path}")

        print(metrics)
        return metrics

    # Default run method (entry point)
    def _run(
        self,
        data: L.LightningDataModule,
        task: Literal["fit", "test", "predict", "evaluate"],
        ckpt_path: Optional[PathLike] = None,
    ):
        """
        Run the specified task on the given data.

        Parameters
        ----------
        data : L.LightningDataModule
            The LightningDataModule object containing the data for the task.
        task : Literal["fit", "test", "predict", "evaluate"], optional
            The task to be performed. Valid options are "fit", "test",
            "predict", and "evaluate".
        ckpt_path : PathLike, optional
            The path to the checkpoint file to be used for resuming training or
            performing inference. Defaults to None.

        Returns
        -------
        Any
            The result of the specified task.

        Raises
        ------
        ValueError
            If an unknown task is provided.
        """
        self._data = data

        if task == "fit":
            return self._fit(data, ckpt_path)
        elif task == "test":
            return self._test(data, ckpt_path)
        elif task == "predict":
            return self._predict(data, ckpt_path)
        elif task == "evaluate":
            return self._evaluate(data, ckpt_path)
        else:
            raise ValueError(f"Unknown task: {task}")


def main():
    from jsonargparse import CLI

    CLI(SimpleLightningPipeline, as_positional=False)
    print("✨ 🍰 ✨")


if __name__ == "__main__":
    main()
