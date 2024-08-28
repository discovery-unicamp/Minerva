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
    ):
        super().__init__(log_dir=log_dir, save_run_status=save_run_status)
        self.model = model
        self.search_space = search_space

    def _search(
        self,
        data: L.LightningDataModule,
        ckpt_path: Optional[PathLike],
        configs: Dict[str, Any],
    ) -> Any:
        def _tuner_train_func(config):
            dm = deepcopy(data)
            model = self.model(config_dict=config)
            trainer = L.Trainer(
                devices=configs.get("devices", "auto"),
                accelerator=configs.get("accelerator", "auto"),
                strategy=configs.get(
                    "strategy", RayDDPStrategy(find_unused_parameters=True)
                ),
                callbacks=configs.get("callbacks", [RayTrainReportCallback()]),
                plugins=configs.get("plugins", [RayLightningEnvironment()]),
                enable_progress_bar=False,
                num_nodes=configs.get("num_nodes", 1),
                enable_checkpointing=(
                    False if configs.get("debug_mode") is True else None
                ),
            )
            trainer = prepare_trainer(trainer)
            trainer.fit(model, dm, ckpt_path=ckpt_path)

        scheduler = configs.get(
            "scheduler",
            ASHAScheduler(
                time_attr=configs.get("time_attr", "training_iteration"),
                max_t=configs.get("max_t", 2),
                grace_period=configs.get("grace_period", 1),
            ),
        )

        scaling_config = configs.get(
            "scaling_config",
            ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1}),
        )

        run_config = configs.get(
            "run_config",
            RunConfig(
                checkpoint_config=(
                    CheckpointConfig(
                        num_to_keep=configs.get("num_to_keep", 1),
                        checkpoint_score_attribute=configs.get(
                            "tuner_metric", "val_loss"
                        ),
                        checkpoint_score_order=configs.get("tuner_mode", "min"),
                    )
                    if configs.get("debug_mode", False) is True
                    else None
                ),
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
                metric=configs.get("tuner_metric", "val_loss"),
                mode=configs.get("tuner_mode", "min"),
                num_samples=configs.get("num_samples", 10),
                scheduler=scheduler,
            ),
        )
        return tuner.fit()

    def _test(self, data: L.LightningDataModule, ckpt_path: Optional[PathLike]) -> Any:
        # TODO fix this
        return self.trainer.test(self.model, data, ckpt_path=ckpt_path)

    def _run(
        self,
        data: L.LightningDataModule,
        task: Optional[Literal["search", "test", "predict"]],
        ckpt_path: Optional[PathLike] = None,
        configs: Dict[str, Any] = {},
    ) -> Any:
        if task == "search":
            return self._search(data, ckpt_path, configs)
        elif task == "test":
            return self._test(data, ckpt_path)
        elif task is None:
            search = self._search(data, ckpt_path, configs)
            test = self._test(data, ckpt_path)
            return search, test
