"""Generate synthetic time series and evaluate TS2Vec FID during training."""

import csv
import json
import logging
import os
import random
import shutil
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.strategies import DDPStrategy
from scipy import stats
from torch.utils.data import DataLoader, IterableDataset

from minerva.analysis.metrics.fid_score import compute_ts_fid
from minerva.utils.typing import PathLike

log = logging.getLogger(__name__)


@contextmanager
def _preserve_rng_state():
    """Restore training RNGs even when generation or FID raises an exception."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def _seed_rng(seed: int) -> None:
    """Set Python, NumPy, and Torch random-number generator seeds for evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def _atomic_path(destination: Path):
    """Publish a complete artifact using a temporary file in the same folder."""
    with NamedTemporaryFile(
        dir=destination.parent, suffix=destination.suffix, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


class SyntheticDataFIDCallback(Callback):
    def __init__(
        self,
        every_n_train_steps: int = 100_000,
        generation_batch_size: int = 64,
        num_fid_runs: int = 5,
        output_dir: Optional[PathLike] = None,
        csv_filename: str = "synthetic_data_quality.csv",
        synthetic_data_prefix: str = "synthetic_data",
        num_samples: Optional[int] = None,
        sampling_method: str = "sample",
        sample_size_arg: str = "batch_size",
        sample_kwargs: Optional[Dict[str, Any]] = None,
        transpose_real_data: bool = True,
        transpose_generated_data: bool = True,
        data_key: Union[int, str] = 0,
        fid_device: Optional[str] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        fit_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        evaluate_at_start: bool = False,
        save_plot: bool = False,
        checkpoint_dir: Optional[PathLike] = None,
        save_best_checkpoint: bool = True,
        best_checkpoint_filename: str = "best.ckpt",
        step_var_name: Optional[str] = None,
    ) -> None:
        """Evaluate synthetic time-series quality during training.

        At each configured optimizer step, the callback generates one synthetic
        dataset from the current model and compares it with the training data in
        a TS2Vec representation space. It trains a fresh TS2Vec encoder for every
        repetition and reports the mean FID score. Lower FID values indicate more
        similar real and synthetic representations.

        Generated samples and scores are saved so completed evaluations can be
        reused after resuming training. The callback can also update a score plot
        and copy the checkpoint with the lowest mean FID. A matching checkpoint
        must exist before each evaluation; place ``SpecificCheckpointCallback``
        before this callback and configure both with the same steps.

        Evaluation runs synchronously on the main process. The data module must
        provide one map-style training DataLoader and set
        ``use_val_with_train=False``. The full training set is cached on CPU,
        and the training random-number-generator states are restored afterward.
        Single-device and DDP training are supported, but sharded strategies are
        not. Under DDP, other ranks wait while rank 0 evaluates. Configure
        ``DDPStrategy(timeout=...)`` to exceed the longest expected evaluation;
        a timeout warning is only advisory and does not prevent a collective
        timeout.

        Parameters
        ----------
        every_n_train_steps : int, optional
            Number of optimizer steps between evaluations, by default 100000.
            Uses ``Trainer.global_step`` unless ``step_var_name`` is set.
        generation_batch_size : int, optional
            Maximum number of synthetic samples generated per model call, by
            default 64.
        num_fid_runs : int, optional
            Number of independent TS2Vec trainings per evaluation, by default 5.
            Must be at least 2.
        output_dir : path-like, optional
            Directory for samples and evaluation results. Defaults to
            ``trainer.log_dir / "synthetic_data"``.
        csv_filename : str, optional
            Results filename inside ``output_dir``, by default
            ``"synthetic_data_quality.csv"``.
        synthetic_data_prefix : str, optional
            Prefix for generated ``.npy`` files, by default ``"synthetic_data"``.
        num_samples : int, optional
            Total number of synthetic samples. By default, use the number of real
            training samples.
        sampling_method : str, optional
            Model method used to generate samples, by default ``"sample"``. For
            Diffusion-TS, use ``"generate_mts"``.
        sample_size_arg : str, optional
            Name of the sample-count argument accepted by ``sampling_method``, by
            default ``"batch_size"``. The callback supplies its value.
        sample_kwargs : dict, optional
            Additional keyword arguments passed to ``sampling_method``.
        transpose_real_data : bool, optional
            If ``True``, transpose real data from ``(samples, channels, time)``
            to ``(samples, time, channels)`` for TS2Vec, by default ``True``.
        transpose_generated_data : bool, optional
            Apply the same conversion to generated data, by default ``True``.
            Set to ``False`` for Diffusion-TS output.
        data_key : int or str, optional
            Position or key containing the input data in each training batch, by
            default 0. Tensor batches are used directly.
        fid_device : str, optional
            Device used to train TS2Vec. Defaults to the diffusion model's device.
        encoder_kwargs : dict, optional
            Additional arguments used to initialize TS2Vec.
        fit_kwargs : dict, optional
            Additional arguments passed to ``TS2Vec.fit``.
        seed : int, optional
            Fallback evaluation seed. The callback first uses Lightning's active
            seed, then this value, and finally 42. Separate deterministic seeds
            are derived for generation and each FID repetition.
        evaluate_at_start : bool, optional
            If ``True``, also evaluate the initial weights at step 0, by default
            ``False``. Requires an ``epoch=-1.ckpt`` checkpoint.
        save_plot : bool, optional
            If ``True``, update ``fid_scores.png`` after each evaluation, by
            default ``False``.
        checkpoint_dir : path-like, optional
            Directory containing the matching checkpoints. Defaults to
            ``trainer.log_dir / "checkpoints"``.
        save_best_checkpoint : bool, optional
            If ``True``, copy the checkpoint with the lowest mean FID, by default
            ``True``.
        best_checkpoint_filename : str, optional
            Filename used for the best checkpoint inside ``checkpoint_dir``, by
            default ``"best.ckpt"``.
        step_var_name : str, optional
            Model or trainer attribute used as the step counter. By default,
            ``Trainer.global_step`` is used. Set to ``"step_counter"`` for
            Diffusion-TS and use the same value in ``SpecificCheckpointCallback``.
        """
        super().__init__()
        for name, value, minimum in (
            ("every_n_train_steps", every_n_train_steps, 1),
            ("generation_batch_size", generation_batch_size, 1),
            ("num_fid_runs", num_fid_runs, 2),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None.")
        if num_samples is not None and (
            type(num_samples) is not int or num_samples < 2
        ):
            raise ValueError("num_samples must be an integer >= 2 or None.")
        for name, value in (
            ("transpose_real_data", transpose_real_data),
            ("transpose_generated_data", transpose_generated_data),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean.")
        if sample_size_arg in (sample_kwargs or {}):
            raise ValueError("The sampling batch size is controlled by the callback.")
        for name, value in (
            ("csv_filename", csv_filename),
            ("synthetic_data_prefix", synthetic_data_prefix),
        ):
            if (
                not isinstance(value, str)
                or not value
                or value in (".", "..")
                or Path(value).name != value
                or "\\" in value
            ):
                raise ValueError(f"{name} must be a nonempty name without a directory.")
        if not csv_filename.endswith(".csv"):
            raise ValueError("csv_filename must end in .csv.")
        if (
            not isinstance(best_checkpoint_filename, str)
            or Path(best_checkpoint_filename).name != best_checkpoint_filename
            or not best_checkpoint_filename.endswith(".ckpt")
            or best_checkpoint_filename.startswith(("step=", "epoch="))
        ):
            raise ValueError(
                "best_checkpoint_filename must be a .ckpt filename without a "
                "directory or a step=/epoch= prefix."
            )
        self.every_n_train_steps = every_n_train_steps
        self.generation_batch_size = generation_batch_size
        self.num_fid_runs = num_fid_runs
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.csv_filename = csv_filename
        self.synthetic_data_prefix = synthetic_data_prefix
        self.dataset_name = None
        self.num_samples = num_samples
        self.sampling_method = sampling_method
        self.sample_size_arg = sample_size_arg
        self.sample_kwargs = dict(sample_kwargs or {})
        self.transpose_real_data = transpose_real_data
        self.transpose_generated_data = transpose_generated_data
        self.data_key = data_key
        self.fid_device = fid_device
        self.encoder_kwargs = dict(encoder_kwargs or {})
        self.fit_kwargs = dict(fit_kwargs or {})
        self.seed = seed
        self._evaluation_seed = None
        self.evaluate_at_start = evaluate_at_start
        self.save_plot = save_plot
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )
        self.save_best_checkpoint = save_best_checkpoint
        self.best_checkpoint_filename = best_checkpoint_filename
        if step_var_name is not None and (
            not isinstance(step_var_name, str) or not step_var_name
        ):
            raise ValueError("step_var_name must be a nonempty string or None.")
        self.step_var_name = step_var_name
        self.best_model_score = None
        self.best_model_path = None
        self._best_checkpoint_id = None
        self._last_step = -1
        self._real_series = None
        self._ddp_timeout_warning_issued = False

    @property
    def state_key(self) -> str:
        """Keep the state key stable before and after the datamodule is attached."""
        return self._generate_state_key(
            every_n_train_steps=self.every_n_train_steps,
            csv_filename=self.csv_filename,
            synthetic_data_prefix=self.synthetic_data_prefix,
            **({"step_var_name": self.step_var_name} if self.step_var_name else {}),
        )

    def state_dict(self) -> Dict[str, Any]:
        """Keep checkpoints small; samples and results live in output_dir."""
        return {"last_step": self._last_step}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore the last completed evaluation step."""
        self._last_step = state_dict.get("last_step", -1)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Resolve the run directory and optionally evaluate initial weights."""
        if getattr(getattr(trainer, "datamodule", None), "use_val_with_train", False):
            raise ValueError(
                "SyntheticDataFIDCallback requires use_val_with_train=False "
                "to evaluate only the training partition."
            )
        if trainer.world_size > 1 and not isinstance(trainer.strategy, DDPStrategy):
            raise ValueError(
                "SyntheticDataFIDCallback supports single-device and DDP only."
            )
        if not callable(getattr(pl_module, self.sampling_method, None)):
            raise ValueError(f"Model has no callable {self.sampling_method!r} method.")
        if self.output_dir is None:
            self.output_dir = (
                Path(trainer.log_dir or trainer.default_root_dir) / "synthetic_data"
            )
        if self.checkpoint_dir is None:
            self.checkpoint_dir = (
                Path(trainer.log_dir or trainer.default_root_dir) / "checkpoints"
            )
        step = self._get_training_step(trainer, pl_module)
        self.dataset_name, dataset_source = self._resolve_dataset_name(trainer)
        self._evaluation_seed, seed_source = self._resolve_seed()
        if trainer.is_global_zero:
            log.info(
                "SyntheticDataFIDCallback: dataset=%s (source=%s).",
                self.dataset_name,
                dataset_source,
            )
            log.info(
                "SyntheticDataFIDCallback: evaluation seed=%d (source=%s).",
                self._evaluation_seed,
                seed_source,
            )
            for logger in trainer.loggers:
                logger.log_hyperparams(
                    {
                        "synthetic_data_fid": {
                            "dataset": self.dataset_name,
                            "dataset_source": dataset_source,
                            "seed": self._evaluation_seed,
                            "seed_source": seed_source,
                        }
                    }
                )
        if self.evaluate_at_start and step == 0 and self._last_step < 0:
            self._evaluate(trainer, pl_module, 0)

    @staticmethod
    def _resolve_dataset_name(trainer: Trainer) -> tuple[str, str]:
        """Derive the dataset name from the datamodule's configured data folders."""
        datamodule = getattr(trainer, "datamodule", None)
        paths = getattr(datamodule, "data_path", None)
        if paths is not None:
            if isinstance(paths, (str, Path)):
                paths = [paths]
            names = [Path(path).name for path in paths]
            if names and all(names):
                return " + ".join(names), "datamodule.data_path"
        dataset = getattr(trainer.train_dataloader, "dataset", None)
        if dataset is not None:
            return type(dataset).__name__, "training dataset class"
        raise ValueError(
            "Cannot identify the dataset: no data_path directory name or training dataset."
        )

    def _resolve_seed(self) -> tuple[int, str]:
        """Reuse the pipeline's active Lightning seed before applying fallbacks."""
        pipeline_seed = os.environ.get("PL_GLOBAL_SEED")
        if pipeline_seed is not None:
            try:
                seed = int(pipeline_seed)
            except ValueError as exc:
                raise ValueError(
                    "PL_GLOBAL_SEED must be a nonnegative integer."
                ) from exc
            if seed < 0:
                raise ValueError("PL_GLOBAL_SEED must be a nonnegative integer.")
            return seed, "Lightning PL_GLOBAL_SEED"
        if self.seed is not None:
            return self.seed, "callback seed parameter"
        return 42, "default"

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx
    ) -> None:
        """Evaluate once at each configured optimizer-step boundary."""
        step = self._get_training_step(trainer, pl_module)
        if step > self._last_step and step > 0 and step % self.every_n_train_steps == 0:
            self._evaluate(trainer, pl_module, step)

    def _get_training_step(self, trainer: Trainer, pl_module: LightningModule) -> int:
        """Resolve the same counter used by the independent checkpoint callback."""
        if self.step_var_name is None:
            step = trainer.global_step
        elif hasattr(pl_module, self.step_var_name):
            step = getattr(pl_module, self.step_var_name)
        elif hasattr(trainer, self.step_var_name):
            step = getattr(trainer, self.step_var_name)
        else:
            raise ValueError(
                f"Training step counter {self.step_var_name!r} was not found."
            )
        if (
            not isinstance(step, (int, np.integer))
            or isinstance(step, bool)
            or step < 0
        ):
            raise ValueError("The training step counter must be a nonnegative integer.")
        return int(step)

    def _evaluate(
        self, trainer: Trainer, pl_module: LightningModule, step: int
    ) -> None:
        """Evaluate once in the main process, preserving training random states.

        With multiple training processes, the others wait for the evaluation's
        outcome before continuing. This avoids duplicate generation and multiple
        processes writing to the same files. Any evaluation error is reported
        to every process so they stop together.
        """
        if trainer.world_size > 1 and trainer.is_global_zero:
            timeout = getattr(trainer.strategy, "_timeout", None)
            # Duration is workload-dependent; flag the common 30-minute limit.
            if not self._ddp_timeout_warning_issued and (
                timeout is None
                or (isinstance(timeout, timedelta) and timeout <= timedelta(minutes=30))
            ):
                log.warning(
                    "Synthetic-data FID: DDP process-group timeout (%s) may be "
                    "too short for rank-0 evaluation with %d TS2Vec runs. Other "
                    "ranks wait for the broadcast; configure DDPStrategy(timeout=...) "
                    "to exceed the longest expected evaluation time.",
                    timeout if timeout is not None else "backend default",
                    self.num_fid_runs,
                )
                self._ddp_timeout_warning_issued = True
        error = None
        failure = None
        if trainer.is_global_zero:
            try:
                with _preserve_rng_state():
                    self._evaluate_synthetic_data(trainer, pl_module, step)
            except Exception as exc:
                error = f"Synthetic-data FID failed at step {step}: {exc}"
                failure = exc
        if trainer.world_size > 1:
            error = trainer.strategy.broadcast(error, src=0)
        if error is not None:
            raise RuntimeError(error) from failure
        self._last_step = step

    @staticmethod
    def _as_series(data, transpose: bool) -> np.ndarray:
        """Validate numeric series and optionally swap time and channel axes."""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu()
            if data.dtype == torch.bfloat16:
                data = data.float()
            data = data.numpy()
        data = np.asarray(data)
        if data.ndim != 3 or min(data.shape) < 1:
            raise ValueError(
                "Expected nonempty series with shape (N, C, T) or (N, T, C)."
            )
        if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
            raise ValueError("Series must contain real numeric values.")
        if not np.isfinite(data).all():
            raise ValueError("Series contain NaN or infinite values.")
        return data.transpose(0, 2, 1) if transpose else data

    def _load_real_series(self, trainer: Trainer) -> np.ndarray:
        """Read the full map-style dataset without disturbing the train iterator."""
        loader = trainer.train_dataloader
        if not isinstance(loader, DataLoader) or isinstance(
            loader.dataset, IterableDataset
        ):
            raise TypeError("A single map-style training DataLoader is required.")
        if len(loader.dataset) < 2:
            raise ValueError("FID requires at least two real samples.")
        reference_loader = DataLoader(
            loader.dataset,
            batch_size=self.generation_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=loader.collate_fn,
        )
        chunks = []
        for batch in reference_loader:
            data = (
                batch
                if isinstance(batch, (torch.Tensor, np.ndarray))
                else batch[self.data_key]
            )
            chunks.append(self._as_series(data, self.transpose_real_data))
        result = np.concatenate(chunks)
        if len(result) != len(loader.dataset):
            raise ValueError("The collate function changed the training sample count.")
        return result

    def _generate(self, pl_module: LightningModule, count: int) -> np.ndarray:
        """Generate bounded batches, preserving every submodule's training mode."""
        training_modes = [(module, module.training) for module in pl_module.modules()]
        result = None
        try:
            pl_module.eval()
            with torch.no_grad():
                for start in range(0, count, self.generation_batch_size):
                    size = min(self.generation_batch_size, count - start)
                    kwargs = {**self.sample_kwargs, self.sample_size_arg: size}
                    generated = getattr(pl_module, self.sampling_method)(**kwargs)
                    series = self._as_series(generated, self.transpose_generated_data)
                    if (
                        len(series) != size
                        or series.shape[1:] != self._real_series.shape[1:]
                    ):
                        raise ValueError(
                            "Generated batch count, time length or channel count does not match the reference."
                        )
                    if result is None:
                        result = np.empty(
                            (count, *series.shape[1:]), dtype=series.dtype
                        )
                    result[start : start + size] = series
        finally:
            for module, training in training_modes:
                module.training = training
        return result

    def _evaluate_synthetic_data(
        self, trainer: Trainer, pl_module: LightningModule, step: int
    ) -> None:
        """Save samples before fitting encoders, then atomically update the CSV."""
        ckpt_id = "epoch=-1" if step == 0 else f"step={step}"
        checkpoint_path = self.checkpoint_dir / f"{ckpt_id}.ckpt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint must be saved before FID evaluation: {checkpoint_path}. "
                "Place SpecificCheckpointCallback before SyntheticDataFIDCallback "
                "in trainer.callbacks and include this step in its schedule."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sample_path = self.output_dir / f"{self.synthetic_data_prefix}_{ckpt_id}.npy"
        csv_path = self.output_dir / self.csv_filename
        rows = []
        if csv_path.exists():
            with csv_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
        config = json.dumps(
            dict(
                dataset_name=self.dataset_name,
                csv_filename=self.csv_filename,
                synthetic_data_prefix=self.synthetic_data_prefix,
                num_fid_runs=self.num_fid_runs,
                seed=self._evaluation_seed,
                transpose_real_data=self.transpose_real_data,
                transpose_generated_data=self.transpose_generated_data,
                num_samples=self.num_samples,
                encoder_kwargs=self.encoder_kwargs,
                fit_kwargs=self.fit_kwargs,
                sampling_method=self.sampling_method,
                sample_size_arg=self.sample_size_arg,
                sample_kwargs=self.sample_kwargs,
                generation_batch_size=self.generation_batch_size,
                fid_device=str(self.fid_device or pl_module.device),
                **({"step_var_name": self.step_var_name} if self.step_var_name else {}),
            ),
            sort_keys=True,
        )
        # Record the protocol before generation so a failed first FID run cannot
        # leave samples that are silently reused with different settings.
        config_path = self.output_dir / "evaluation_config.json"
        if config_path.exists():
            if json.loads(config_path.read_text()) != json.loads(config):
                raise ValueError(
                    "The output directory contains a different evaluation protocol. Use a new output_dir."
                )
        else:
            with _atomic_path(config_path) as temporary:
                temporary.write_text(config + "\n")
        for row in rows:
            if (
                row.get("evaluation_config") != config
                or row.get("dataset") != self.dataset_name
            ):
                raise ValueError(
                    "The output directory contains a different evaluation protocol. Use a new output_dir."
                )
            if row["ckpt_id"] == ckpt_id:
                if not sample_path.exists():
                    raise FileNotFoundError(
                        f"Completed CSV row has no sample file: {sample_path}"
                    )
                self._update_best_checkpoint(rows)
                return

        seeds = np.random.SeedSequence([self._evaluation_seed, step]).generate_state(
            self.num_fid_runs + 1
        )
        _seed_rng(self._evaluation_seed % 2**32)
        if self._real_series is None:
            self._real_series = self._load_real_series(trainer)
        count = self.num_samples or len(self._real_series)
        if sample_path.exists():
            generated = self._as_series(
                np.load(sample_path, allow_pickle=False), self.transpose_generated_data
            )
            if generated.shape != (count, *self._real_series.shape[1:]):
                raise ValueError(
                    "Existing synthetic data does not match the current dataset or sample count."
                )
        else:
            _seed_rng(int(seeds[0]))
            generated = self._generate(pl_module, count)
            # Preserve the generator's original axis order in the saved file.
            saved = (
                generated.transpose(0, 2, 1)
                if self.transpose_generated_data
                else generated
            )
            with _atomic_path(sample_path) as temporary:
                np.save(temporary, saved, allow_pickle=False)

        real = self._real_series
        scores = []
        for run in range(self.num_fid_runs):
            _seed_rng(int(seeds[run + 1]))
            # Encoder training must not inherit a surrounding inference context.
            with torch.inference_mode(False), torch.enable_grad():
                score = float(
                    compute_ts_fid(
                        real,
                        generated,
                        device=self.fid_device or pl_module.device,
                        encoder_kwargs=self.encoder_kwargs,
                        fit_kwargs=self.fit_kwargs,
                    )
                )
            if not np.isfinite(score):
                raise ValueError(f"Non-finite FID in repetition {run + 1}.")
            scores.append(score)
            log.info(
                "%s %s FID %d/%d: %.6f",
                self.dataset_name,
                ckpt_id,
                run + 1,
                self.num_fid_runs,
                score,
            )
        mean = float(np.mean(scores))
        std = float(np.std(scores, ddof=1))
        # Original display_scores formula, generalized beyond five repetitions.
        sigma = float(
            stats.sem(scores) * stats.t.ppf((1 + 0.95) / 2.0, len(scores) - 1)
        )
        row = dict(dataset=self.dataset_name, ckpt_id=ckpt_id, step=step)
        row.update({f"FID_score_{i + 1}": value for i, value in enumerate(scores)})
        row.update(
            FID_score_mean=mean,
            FID_score_sigma=sigma,
            FID_score_std=std,
            num_real_samples=len(real),
            num_synthetic_samples=count,
            transpose_real_data=self.transpose_real_data,
            transpose_generated_data=self.transpose_generated_data,
            evaluation_config=config,
        )
        rows.append(row)
        rows.sort(key=lambda item: int(item["step"]))
        with _atomic_path(csv_path) as temporary:
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
        self._update_best_checkpoint(rows)
        for logger in trainer.loggers:
            logger.log_metrics(
                {
                    "synthetic/fid_mean": mean,
                    "synthetic/fid_std": std,
                    "synthetic/fid_sigma": sigma,
                },
                step=step,
            )
        if self.save_plot:
            self._plot(rows)

    def _update_best_checkpoint(self, rows) -> None:
        """Recover the minimum from CSV and copy its original checkpoint atomically."""
        if not self.save_best_checkpoint or not rows:
            return
        best = min(
            rows, key=lambda row: (float(row["FID_score_mean"]), int(row["step"]))
        )
        destination = self.checkpoint_dir / self.best_checkpoint_filename
        if self._best_checkpoint_id == best["ckpt_id"] and destination.is_file():
            return
        source = self.checkpoint_dir / f"{best['ckpt_id']}.ckpt"
        if not source.is_file():
            raise FileNotFoundError(f"Cannot copy the best FID checkpoint: {source}")
        with _atomic_path(destination) as temporary:
            shutil.copyfile(source, temporary)
        self._best_checkpoint_id = best["ckpt_id"]
        self.best_model_score = float(best["FID_score_mean"])
        self.best_model_path = str(destination)
        log.info("Best FID checkpoint: %s (mean %.6f)", source, self.best_model_score)

    def _plot(self, rows) -> None:
        """Plot actual optimizer steps without hard-coded datasets or axis limits."""
        from matplotlib.figure import Figure

        figure = Figure(figsize=(8, 4))
        axes = figure.subplots()
        axes.errorbar(
            [int(row["step"]) for row in rows],
            [float(row["FID_score_mean"]) for row in rows],
            yerr=[float(row["FID_score_sigma"]) for row in rows],
            marker="o",
            capsize=3,
        )
        axes.set(
            xlabel="Training steps",
            ylabel="TS2Vec FID",
            title=self.dataset_name,
        )
        axes.grid(alpha=0.3)
        with _atomic_path(self.output_dir / "fid_scores.png") as temporary:
            figure.savefig(temporary, dpi=150, bbox_inches="tight")
