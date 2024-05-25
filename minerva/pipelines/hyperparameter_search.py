from typing import Any, Dict, Literal, Optional

import lightning as L
from ray import tune
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.sample import Categorical, Float, Integer
from torchmetrics import Metric

from minerva.pipelines.base import Pipeline
from minerva.utils.typing import PathLike


class HyperParameterSearch(Pipeline):

    def __init__(
        self,
        model: L.LightningModule,
        trainer: L.Trainer,
        log_dir: PathLike,
        save_run_status: bool,
        classification_metrics: Dict[str, Metric],
        regression_metrics: Dict[str, Metric],
        apply_metrics_per_sample: bool,
        search_space=Dict[str, Categorical | Float | Integer],
    ):
        self.model = model
        self.trainer = trainer
        self.log_dir = log_dir
        self.save_run_status = save_run_status
        self.classification_metrics = classification_metrics
        self.regression_metrics = regression_metrics
        self.apply_metrics_per_sample = apply_metrics_per_sample
        self.search_space = search_space

    def _search(
        self, data: L.LightningDataModule, ckpt_path: Optional[PathLike]
    ) -> Any:
        tune.loguniform

    def _test(self, data: L.LightningDataModule, ckpt_path: Optional[PathLike]) -> Any:
        raise NotImplementedError

    def _predict(
        self, data: L.LightningDataModule, ckpt_path: Optional[PathLike]
    ) -> Any:
        raise NotImplementedError

    def _evaluate(
        self, data: L.LightningDataModule, ckpt_path: Optional[PathLike]
    ) -> Any:
        raise NotImplementedError

    def _run(
        self,
        data: L.LightningDataModule,
        task: Optional[Literal["search", "test", "predict", "evaluate"]],
        ckpt_path: Optional[PathLike] = None,
    ) -> Any:
        if task == "search":
            return self._search(data, ckpt_path)
        elif task == "test":
            return self._test(data, ckpt_path)
        elif task == "predict":
            return self._predict(data, ckpt_path)
        elif task == "evaluate":
            return self._evaluate(data, ckpt_path)
        elif task is None:
            search = self._search(data, ckpt_path)
            test = self._test(data, ckpt_path)
            predict = self._predict(data, ckpt_path)
            evaluate = self._evaluate(data, ckpt_path)
            return search, test, predict, evaluate
