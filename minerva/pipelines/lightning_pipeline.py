from pathlib import Path
from typing import Dict, List, Literal
import lightning as L
import torch
from minerva.pipelines.base import Pipeline
from torchmetrics import Metric
from minerva.utils.data import get_full_data_split
from collections import defaultdict
import yaml


class SimpleLightningPipeline(Pipeline):
    def __init__(
        self,
        model: L.LightningModule,
        trainer: L.Trainer,
        cwd: str = None,
        save_run_status: bool = False,
        classification_metrics: Dict[str, Metric] = None,
        regression_metrics: Dict[str, Metric] = None,
        apply_metrics_per_sample: bool = False,
    ):
        super().__init__(
            cwd=cwd,
            ignore=["model", "trainer"],
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
    def model(self):
        return self._model

    @property
    def trainer(self):
        return self._trainer

    @property
    def data(self):
        return self._data

    def _calculate_metrics(self, metrics: Dict[str, Metric], y_hat, y):
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
    def _fit(self, data: L.LightningDataModule, ckpt_path: str | Path):
        return self._trainer.fit(
            model=self._model, datamodule=data, ckpt_path=ckpt_path
        )

    def _test(self, data: L.LightningDataModule, ckpt_path: str | Path):
        return self._trainer.test(
            model=self._model, datamodule=data, ckpt_path=ckpt_path
        )

    def _predict(
        self,
        data: L.LightningDataModule,
        ckpt_path: str | Path = None,
    ):
        return self._trainer.predict(
            model=self._model, datamodule=data, ckpt_path=ckpt_path
        )

    def _evaluate(
        self,
        data: L.LightningDataModule,
        ckpt_path: str | Path = None,
    ):
        metrics = defaultdict(dict)

        X, y = get_full_data_split(data, "predict")
        y = torch.tensor(y, device="cpu")

        y_hat = self.trainer.predict(
            self._model, datamodule=data, ckpt_path=ckpt_path
        )
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
            yaml_path = self.working_dir / f"metrics_{self.pipeline_id}.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(metrics, f)
                print(f"Metrics saved to {yaml_path}")
                
        print(metrics)

        return metrics

    # Default run method (entry point)
    def _run(
        self,
        data: L.LightningDataModule,
        task: Literal["fit", "test", "predict", "evaluate"] = "fit",
        ckpt_path: str | Path = None,
    ):
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
    # print("✨ 🍰 ✨")


if __name__ == "__main__":
    main()
