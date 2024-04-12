#!/usr/bin/env python3

import copy
from dataclasses import dataclass
from datetime import datetime
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List

import lightning as L
import ray
import yaml
from jsonargparse import CLI
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from torchmetrics import Accuracy

from typing import Any, Dict, List


@dataclass
class ExperimentArgs:
    trainer: Dict[str, Any]
    model: Dict[str, Any]
    data: Dict[str, Any]
    test_data: Dict[str, Any]
    seed: int = 42
        

def cli_main(experiment: ExperimentArgs):
    class DummyModel(L.LightningModule):
        def __init__(self, *args, **kwargs):
            pass

    class DummyTrainer(L.Trainer):
        def __init__(self, *args, **kwargs):
            pass

    # Unpack experiment into a dict, ignoring the test_data for now
    cli_args = {
        "trainer": experiment.trainer,
        "model": experiment.model,
        "data": experiment.data,
        "seed_everything": experiment.seed,
    }
    
    # print(cli_args)

    # Instantiate model, trainer, and train_datamodule
    train_cli = LightningCLI(
        args=cli_args, run=False, parser_kwargs={"parser_mode": "omegaconf"}
    )

    test_cli = LightningCLI(
        model_class=DummyModel,
        trainer_class=DummyTrainer,
        args={
            "trainer": {},
            "model": {},
            "data": experiment.test_data,
        },
        run=False,
    )

    # Shortcut to access the trainer, model and datamodule
    trainer = train_cli.trainer
    model = train_cli.model
    train_data_module = train_cli.datamodule
    test_data_module = test_cli.datamodule

    # Attach model test metrics
    model.metrics["test"]["accuracy"] = Accuracy(
        task="multiclass", num_classes=7
    )

    # Perform FIT
    trainer.fit(model, train_data_module)

    # Perform test and return metrics
    metrics = trainer.test(model, test_data_module, ckpt_path="best")
    return metrics


def _run_experiment_wrapper(experiment_args: ExperimentArgs):
    try:
        print()
        print("*" * 80)
        print(f"Running Experiment")
        print(f"    Model: {experiment_args.model['class_path']}")
        print(
            f"    Train Data: {experiment_args.data['init_args']['data_path']}"
        )
        print(
            f"    Test Data: {experiment_args.test_data['init_args']['data_path']}"
        )
        print("*" * 80)
        print()

        return cli_main(experiment_args)
    except Exception as e:
        print(f" ------- Error running evaluator: {e} ----------")
        traceback.print_exc()
        print("----------------------------------------------------")
        raise e


def run_using_ray(experiments: List[ExperimentArgs], ray_address: str = None):
    print(f"Running {len(experiments)} experiments using RAY...")
    ray.init(address=ray_address)
    remotes_to_run = [
        ray.remote(
            num_gpus=0.25,
            num_cpus=4,
            max_calls=1,
            max_retries=0,
            retry_exceptions=False,
        )(_run_experiment_wrapper).remote(exp_args)
        for exp_args in experiments
    ]
    ready, not_ready = ray.wait(remotes_to_run, num_returns=len(remotes_to_run))
    print(f"Ready: {len(ready)}. Not ready: {len(not_ready)}")
    ray.shutdown()
    return ready, not_ready


def run_serial(experiments: List[ExperimentArgs]):
    print(f"Running {len(experiments)} experiments...")
    for exp_args in experiments:
        _run_experiment_wrapper(exp_args)


class SupervisedConfigParser:
    def __init__(
        self,
        data_path: str,
        default_trainer_config: str,
        data_module_configs: str,
        model_configs: str,
        output_dir: str = "benchmarks/",
        skip_existing: bool = True,
        enable_checkpointing: bool = False,
        seed: int = 42,
        leave_one_out: bool = False,
    ):
        self.data_path = data_path
        self.default_trainer_config = default_trainer_config
        self.data_module_configs = data_module_configs
        self.model_configs = model_configs
        self.output_dir = output_dir
        self.skip_existing = skip_existing
        self.enable_checkpointing = enable_checkpointing
        self.seed = seed
        self.leave_one_out = leave_one_out

    # TODO automate this, using the a query string, sql like
    def filter_experiments(self, experiments: List[ExperimentArgs]):
        return experiments
        # return [
        #     exp
        #     for exp in experiments
        #     if exp.data["class_path"] == "ssl_tools.data.data_modules.har.AugmentedMultiModalHARSeriesDataModule"
        # ]

    def __call__(self) -> List[ExperimentArgs]:
        input_type = ["1D", "2D"]
        now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        models = list(sorted(self.model_configs.rglob("*.yaml")))
        data_modules = list(sorted(self.data_module_configs.rglob("*.yaml")))
        datasets = list(sorted(self.data_path.glob("*")))
        experiments = []

        initial_trainer_config = yaml.safe_load(
            self.default_trainer_config.read_text()
        )
        initial_trainer_config["enable_checkpointing"] = (
            self.enable_checkpointing
        )

        # Segment datasets into train and test
        if self.leave_one_out:
            datasets = [
                {
                    "train": [d1 for d1 in datasets if d1 != dataset],
                    "test": dataset,
                }
                for dataset in datasets
            ]
        else:
            datasets = [
                {
                    "train": dataset,
                    "test": dataset,
                }
                for dataset in datasets
            ]

        # Scanning for configs
        for model in models:
            input_type = model.parent.stem
            model_name = model.stem
            initial_model_config = yaml.safe_load(model.read_text())

            for data_module in data_modules:
                data_module_input_type = data_module.parent.stem
                data_module_name = data_module.stem
                initial_data_module_config = yaml.safe_load(
                    data_module.read_text()
                )

                if input_type != data_module_input_type:
                    continue

                for dataset in datasets:
                    train_set = dataset["train"]
                    test_set = dataset["test"]

                    if isinstance(train_set, list):
                        train_set_name = "+".join([d.stem for d in train_set])
                    else:
                        train_set_name = train_set.stem
                    test_set_name = test_set.stem

                    name = f"{model_name}-{data_module_name}-train_on_{train_set_name}-test_on_{test_set_name}"
                    version = now

                    if self.skip_existing:
                        if (self.output_dir / name).exists():
                            print(f"-- Skipping existing experiment: {name}")
                            continue

                    trainer_config = copy.deepcopy(initial_trainer_config)
                    model_config = copy.deepcopy(initial_model_config)
                    data_module_config = copy.deepcopy(
                        initial_data_module_config
                    )

                    trainer_config["logger"] = {
                        "class_path": "lightning.pytorch.loggers.CSVLogger",
                        "init_args": {
                            "save_dir": self.output_dir,
                            "name": name,
                            "version": version,
                        },
                    }

                    # TODO ------- remove this when all gpu bugs were fixed --------
                    if (
                        "lstm" in model_name
                        or "inception" in model_name
                        or "cnn_haetal_2d" in model_name
                        or "cnn_pf" in model_name
                        or "cnn_pff" in model_name
                    ):
                        trainer_config["accelerator"] = "cpu"
                    else:
                        trainer_config["accelerator"] = "gpu"

                    data_module_config["init_args"]["data_path"] = train_set

                    test_data_module_config = copy.deepcopy(data_module_config)
                    test_data_module_config["init_args"]["data_path"] = test_set

                    experiments.append(
                        ExperimentArgs(
                            trainer=trainer_config,
                            model=model_config,
                            data=data_module_config,
                            test_data=test_data_module_config,
                            seed=self.seed,
                        )
                    )
        experiments = self.filter_experiments(experiments)

        return experiments


def hack_to_avoid_lightning_cli_sys_argv_warning(func, *args, **kwargs):
    # Hack to avoid LightningCLI parse warning
    # The warning is something like:
    # /usr/local/lib/python3.10/dist-packages/lightning/pytorch/cli.py:520:
    # LightningCLI's args parameter is intended to run from with in Python like
    # if it were from the command line. To prevent mistakes it is not
    # recommended to provide both args and command line arguments, got:
    # sys.argv[1:]=['--config', 'benchmark_a1_dry_run.yaml'],
    def hack_to_avoid_lightning_cli_sys_argv_warning_wrapper(*args, **kwargs):
        import sys

        old_args = sys.argv
        sys.argv = sys.argv[:1]
        func(*args, **kwargs)
        sys.argv = old_args

    return hack_to_avoid_lightning_cli_sys_argv_warning_wrapper


@hack_to_avoid_lightning_cli_sys_argv_warning
def run(
    config_parser: SupervisedConfigParser,
    use_ray: bool,
    ray_address: str = None,
    dry_run: bool = False,
    dry_run_limit: int = 3,
):
    experiments = config_parser()
    if dry_run:
        if dry_run_limit is None:
            dry_run_limit = len(experiments)

        print(
            f"** Dry run. Limiting to a maximum of {dry_run_limit} experiments, shuffled **"
        )
        experiments = random.sample(
            experiments, min(dry_run_limit, len(experiments))
        )

    if use_ray:
        return run_using_ray(experiments, ray_address)
    else:
        return run_serial(experiments)


def main(
    data_path: str,
    default_trainer_config_file: str,
    data_module_configs_path: str,
    model_configs_path: str,
    output_path: str = "benchmarks/",
    skip_existing: bool = True,
    ray_address: str = None,
    use_ray: bool = True,
    enable_checkpointing: bool = False,
    seed: int = 42,
    dry_run: bool = False,
    dry_run_limit: int = 5,
    leave_one_out: bool = False,
):
    parser = SupervisedConfigParser(
        data_path=Path(data_path),
        default_trainer_config=Path(default_trainer_config_file),
        data_module_configs=Path(data_module_configs_path),
        model_configs=Path(model_configs_path),
        output_dir=Path(output_path),
        skip_existing=skip_existing,
        enable_checkpointing=enable_checkpointing,
        seed=seed,
        leave_one_out=leave_one_out,
    )
    return run(
        parser,
        use_ray,
        ray_address,
        dry_run=dry_run,
        dry_run_limit=dry_run_limit,
    )


if __name__ == "__main__":
    CLI(main)
