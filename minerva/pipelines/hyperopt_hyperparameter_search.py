from copy import deepcopy
from typing import Any, Dict, Literal, Optional

import lightning.pytorch as L
from lightning.pytorch.strategies import Strategy
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler, TrialScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.stopper import TrialPlateauStopper

from minerva.callbacks.HyperSearchCallbacks import TrainerReportOnIntervalCallback
from minerva.pipelines.base import Pipeline
from minerva.utils.typing import PathLike


class HyperoptHyperParameterSearch(Pipeline):

    def __init__(
        self,
        model: type,
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
        devices: Optional[str] = "auto",
        accelerator: Optional[str] = "auto",
        strategy: Optional[Strategy] = None,
        callbacks: Optional[Any] = None,
        plugins: Optional[Any] = None,
        num_nodes: int = 1,
        debug_mode: Optional[bool] = False,
        scaling_config: Optional[ScalingConfig] = None,
        run_config: Optional[RunConfig] = None,
        tuner_metric: Optional[str] = "val_loss",
        tuner_mode: Optional[str] = "min",
        num_samples: Optional[int] = 10,
        scheduler: Optional[TrialScheduler] = None,
        max_concurrent: Optional[int] = 4,
        initial_parameters: Optional[Dict[str, Any]] = None,
        max_epochs: Optional[int] = None,
    ) -> Any:

        print(tuner_metric, tuner_mode)

        def _tuner_train_func(config):
            dm = deepcopy(data)
            model = self.model.create_from_dict(config)
            trainer = L.Trainer(
                max_epochs=max_epochs or 500,
                devices=devices or "auto",
                accelerator=accelerator or "auto",
                strategy=strategy or RayDDPStrategy(find_unused_parameters=True),
                callbacks=callbacks or [TrainerReportOnIntervalCallback(500)],
                plugins=plugins or [RayLightningEnvironment()],
                enable_progress_bar=False,
                num_nodes=num_nodes,
                enable_checkpointing=False if debug_mode else None,
            )
            trainer = prepare_trainer(trainer)
            trainer.fit(model, dm, ckpt_path=ckpt_path)

        scheduler = scheduler or ASHAScheduler(
            time_attr="training_iteration",
            metric=tuner_metric or "val_loss",
            mode=tuner_mode or "min",
            max_t=500,
            grace_period=100,
        )

        scaling_config = scaling_config or ScalingConfig(
            num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1}
        )

        run_config = run_config or RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
            stop=TrialPlateauStopper(
                metric=tuner_metric or "val_loss",
                mode=tuner_mode or "min",
                num_results=5,
                std=0.01,
                grace_period=50,
            ),
        )

        ray_trainer = TorchTrainer(
            _tuner_train_func,
            scaling_config=scaling_config,
            run_config=run_config,
        )

        algo = ConcurrencyLimiter(
            HyperOptSearch(initial_parameters), max_concurrent=max_concurrent or 4
        )

        tuner = tune.Tuner(
            ray_trainer,
            param_space={"train_loop_config": self.search_space},
            tune_config=tune.TuneConfig(
                metric=tuner_metric or "val_loss",
                mode=tuner_mode or "min",
                num_samples=num_samples or -1,
                search_alg=algo,
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
        config: Dict[str, Any] = {},
        **kwargs,
    ) -> Any:
        if task == "search":
            print(config)
            return self._search(data, ckpt_path, **config)
        elif task == "test":
            return self._test(data, ckpt_path)
        elif task is None:
            search = self._search(data, ckpt_path, **config)
            test = self._test(data, ckpt_path)
            return search, test


def main():
    from jsonargparse import CLI

    print("Hyper Searching 🔍")
    CLI(HyperoptHyperParameterSearch, as_positional=False)


if __name__ == "__main__":
    main()
