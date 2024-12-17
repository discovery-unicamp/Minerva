from collections import defaultdict
from typing import Any, Dict, Literal, Optional, Union

import lightning as L

import numpy as np
import torch
import time
import yaml
from torchmetrics import Metric

from minerva.pipelines.base import Pipeline
from minerva.utils.typing import PathLike
from minerva.analysis.model_analysis import _ModelAnalysis


class PredictWrapper(L.LightningModule):
    """Wrapper to predict the model and return the batch and the
    predictions in Lightning's predict_step method.

    Note that data module should return a tuple with (X, y) in the
    predict_dataloader method.
    """

    def __init__(self, model: L.LightningModule):
        super().__init__()
        self.model = model

    def predict_step(self, batch, *args, **kwargs):
        x, y = batch
        y_hat = self.model.predict_step(batch, *args, **kwargs)
        return (x, y, y_hat)


class SimpleLightningPipeline(Pipeline):
    """Simple pipeline to train, test, predict and evaluate models using Pytorch
    Lightning. This class is intended to be seamlessly integrated with
    jsonargparse CLI.
    """

    def __init__(
        self,
        model: L.LightningModule,
        trainer: L.Trainer,
        log_dir: Optional[PathLike] = None,
        save_run_status: bool = True,
        classification_metrics: Optional[Dict[str, Metric]] = None,
        regression_metrics: Optional[Dict[str, Metric]] = None,
        model_analysis: Optional[Dict[str, _ModelAnalysis]] = None,
        apply_metrics_per_sample: bool = False,
        seed: Optional[int] = None,
        save_predictions: Optional[Union[bool, PathLike]] = None,
        classification_reduce: Optional[str] = "argmax",
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
        model_analysis: Dict[str, _ModelAnalysis], optional
            The model analysis to be performed after the model is trained. This
            dictionary should have the analysis name as key and the
            `_ModelAnalysis`-like object as value. The analysis should be able
            to receive the model and the data and return a result. If None, no
            model analysis will be performed. By default None.
        apply_metrics_per_sample : bool, optional
            Apply the metrics per sample. If True, the metrics will be
            calculated for each sample and the results will be a list of
            metrics. If False, the metrics will be calculated for the whole
            dataset and the results will be a single metric (single-element
            list). By default False
        seed : int, optional
            The seed to be used in the pipeline. By default None.
        save_predictions : Optional[Union[bool, PathLike]], optional
            If True, save the predictions to a file (predictions.pth). If a
            PathLike object is provided, save the predictions to the given file.
            Note that the predictions will be saved in the log directory,
            without any argmax applied. By default None.
        classification_reduce : Optional[str], optional
            The reduction method to be applied to the classification
            predictions before calculating the metrics (on dimension=1). The
            options are "argmax" (apply argmax to the predictions), "none" (do
            not apply any reduction), "squeeze" (squeeze the predictions). By
            default "argmax".
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
            seed=seed,
        )
        self._model = model
        self._trainer = trainer
        self._data = None
        self._model_analysis = model_analysis
        self._classification_metrics = classification_metrics
        self._classification_reduce = classification_reduce
        self._regression_metrics = regression_metrics
        self._apply_metrics_per_sample = apply_metrics_per_sample
        self._save_predictions_file = save_predictions

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
    def data(self) -> Optional[L.LightningDataModule]:
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
        if self._apply_metrics_per_sample:
            # Iterate over the batch dimension
            pass
        else:
            # Calculate the metric for the whole dataset at once (create a
            # batch dimension with size 1)
            y, y_hat = y.unsqueeze(0), y_hat.unsqueeze(0)

        for metric_name, metric in metrics.items():
            final_results = []
            print(f"\tCalculating {metric_name}...", end=" ")
            for i, (y_i, y_hat_i) in enumerate(zip(y, y_hat)):
                res = metric(y_i, y_hat_i).float().item()
                final_results.append(res)
            results[metric_name] = final_results
            print("done!")
        return results

    # Private methods
    def _fit(
        self, data: L.LightningDataModule, ckpt_path: Optional[PathLike] = None
    ):
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

    def _test(
        self, data: L.LightningDataModule, ckpt_path: Optional[PathLike] = None
    ):
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
        )  # type: ignore

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

        start_time = time.time()
        preds = self.trainer.predict(
            PredictWrapper(self._model), datamodule=data, ckpt_path=ckpt_path
        )
        overall_time = time.time() - start_time
        print(f"Inference took: {overall_time:.3f} seconds!")

        if preds is None:
            raise ValueError("No predictions were generated.")

        x, y, y_hat = zip(*preds)
        x = torch.cat(x).detach().cpu()
        y = torch.cat(y).detach().cpu()
        y_hat = torch.cat(y_hat).detach().cpu()

        if self._save_predictions_file is not None:
            if isinstance(self._save_predictions_file, bool):
                path = self.log_dir / "predictions.npy"
            else:
                path = self.log_dir / self._save_predictions_file
            np.save(str(path), y_hat.numpy())
            print(f"Predictions saved to {path}. Shape: {y_hat.shape}")

            metrics["predictions"]["file"] = str(path)
            metrics["predictions"]["shape"] = list(y_hat.shape)
            metrics["predictions"]["time"] = overall_time

        # Argmax and calculate metrics
        if self._classification_metrics is not None:
            print(f"Running classification metrics...")
            if self._classification_reduce == "argmax":
                y_hat = torch.argmax(y_hat, dim=1)
            elif (
                self._classification_reduce == "none"
                or self._classification_reduce is None
            ):
                pass
            elif self._classification_reduce == "squeeze":
                y_hat = torch.squeeze(y_hat, dim=1)
            else:
                raise ValueError(
                    f"Unknown classification reduce method: {self._classification_reduce}"
                )
            metrics["classification"] = self._calculate_metrics(
                self._classification_metrics, y_hat, y
            )

        # Just calculate metrics (without argmax)
        elif self._regression_metrics is not None:
            print(f"Running regression metrics...")
            metrics["regression"] = self._calculate_metrics(
                self._regression_metrics, y_hat, y
            )

        else:
            pass

        # Run model analysis
        if self._model_analysis is not None:
            print(f"Running model analysis...")
            metrics["analysis"] = {}
            for analysis_name, analysis in self._model_analysis.items():
                analysis.path = self._log_dir
                metrics["analysis"][analysis_name] = analysis.compute(
                    self._model, data
                )

        # Convert metrics from defaultdict to dict
        metrics = dict(metrics)

        # Save metrics to a YAML file
        if self._save_pipeline_info:
            yaml_path = self._log_dir / f"metrics_{self.pipeline_id}.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(metrics, f)
                print(f"Metrics saved to {yaml_path}")

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


def cli_main():
    from jsonargparse import CLI

    CLI(
        SimpleLightningPipeline, as_positional=False
    )  # , parser_mode="omegaconf")
    print("✨ 🍰 ✨")


if __name__ == "__main__":
    cli_main()
