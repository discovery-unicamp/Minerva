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
        search_space: Dict[str, Any],
        log_dir: Optional[PathLike] = None,
        save_run_status: bool = False,
        num_epochs: int = 2,
        num_samples: int = 2,
    ):
        super().__init__(log_dir=log_dir, save_run_status=save_run_status)
        self.model = model
        self.search_space = search_space
        self.trainer = prepare_trainer(
            L.Trainer(
                devices="auto",
                accelerator="auto",
                strategy=RayDDPStrategy(),
                callbacks=[RayTrainReportCallback],
                plugins=[RayLightningEnvironment()],
                enable_progress_bar=False,
            )
        )
        self.num_epochs = num_epochs
        self.num_samples = num_samples

    def _search(
        self, data: L.LightningDataModule, ckpt_path: Optional[PathLike]
    ) -> Any:
        def _tuner_train_func(config):
            model = self.model(config)
            self.trainer.fit(model, data, ckpt_path=ckpt_path)

        scheduler = ASHAScheduler(max_t=self.num_epochs, grace_period=1)
        tuner = tune.Tuner(
            _tuner_train_func,
            param_space=self.search_space,
            tune_config=tune.TuneConfig(
                metric="ptl/val_miou",
                mode="max",
                num_samples=self.num_samples,
                scheduler=scheduler,
            ),
        )
        return tuner.fit()

    def _test(self, data: L.LightningDataModule, ckpt_path: Optional[PathLike]) -> Any:

        return self.trainer.test(self.model, data, ckpt_path=ckpt_path)

    def _predict(
        self, data: L.LightningDataModule, ckpt_path: Optional[PathLike]
    ) -> Any:
        return self.trainer.predict(self.model, data, ckpt_path=ckpt_path)

    def _run(
        self,
        data: L.LightningDataModule,
        task: Optional[Literal["search", "test", "predict"]],
        ckpt_path: Optional[PathLike] = None,
    ) -> Any:
        if task == "search":
            return self._search(data, ckpt_path)
        elif task == "test":
            return self._test(data, ckpt_path)
        elif task == "predict":
            return self._predict(data, ckpt_path)
        elif task is None:
            search = self._search(data, ckpt_path)
            test = self._test(data, ckpt_path)
            return search, test
