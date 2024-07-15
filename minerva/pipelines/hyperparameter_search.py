from copy import deepcopy
from typing import Any, Dict, Literal, Optional

import lightning.pytorch as L
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer
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
        self.num_epochs = num_epochs
        self.num_samples = num_samples

    def _search(
        self, data: L.LightningDataModule, ckpt_path: Optional[PathLike]
    ) -> Any:
        def _tuner_train_func(config):
            dm = deepcopy(data)
            model = self.model(config_dict=config)
            trainer = L.Trainer(
                devices="auto",
                accelerator="auto",
                strategy=RayDDPStrategy(),
                callbacks=[RayTrainReportCallback()],
                plugins=[RayLightningEnvironment()],
                enable_progress_bar=False,
                num_nodes=0,
            )
            trainer = prepare_trainer(trainer)
            trainer.fit(model, dm, ckpt_path=ckpt_path)

        scheduler = ASHAScheduler(max_t=self.num_epochs, grace_period=1)

        scaling_config = ScalingConfig(
            num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1}
        )

        run_config = RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="ptl/val_accuracy",
                checkpoint_score_order="max",
            ),
        )

        ray_trainer = TorchTrainer(
            _tuner_train_func,
            scaling_config=scaling_config,
            run_config=run_config,
        )
        tuner = tune.Tuner(
            ray_trainer,
            param_space={"train_loop_config": self.search_space},
            tune_config=tune.TuneConfig(
                metric="ptl/val_miou",
                mode="max",
                num_samples=self.num_samples,
                scheduler=scheduler,
            ),
        )
        return tuner.fit()

    def _test(self, data: L.LightningDataModule, ckpt_path: Optional[PathLike]) -> Any:
        # TODO fix this
        return self.trainer.test(self.model, data, ckpt_path=ckpt_path)

    def _predict(
        self, data: L.LightningDataModule, ckpt_path: Optional[PathLike]
    ) -> Any:
        # TODO fix this
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
