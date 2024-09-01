import os
import shutil
import tempfile
from pathlib import Path

import lightning.pytorch as L
from ray import train
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train import Checkpoint


class TrainerReportOnIntervalCallback(L.Callback):

    CHECKPOINT_NAME = "checkpoint.ckpt"

    def __init__(self, interval: int = 1) -> None:
        super().__init__()
        self.trial_name = train.get_context().get_trial_name()
        self.local_rank = train.get_context().get_local_rank()
        self.tmpdir_prefix = Path(tempfile.gettempdir(), self.trial_name).as_posix()
        self.interval = interval
        self.step = 0
        if os.path.isdir(self.tmpdir_prefix) and self.local_rank == 0:
            shutil.rmtree(self.tmpdir_prefix)

        record_extra_usage_tag(TagKey.TRAIN_LIGHTNING_RAYTRAINREPORTCALLBACK, "1")

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self.step += 1

        # Fetch metrics
        metrics = trainer.callback_metrics
        metrics = {k: v.item() for k, v in metrics.items()}

        # (Optional) Add customized metrics
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = trainer.global_step

        if self.step % self.interval == 0:
            tmpdir = Path(self.tmpdir_prefix, str(trainer.current_epoch)).as_posix()
            os.makedirs(tmpdir, exist_ok=True)

            # Save checkpoint to local
            ckpt_path = Path(tmpdir, self.CHECKPOINT_NAME).as_posix()
            trainer.save_checkpoint(ckpt_path, weights_only=False)

        # Report to train session
        checkpoint = Checkpoint.from_directory(tmpdir)
        train.report(metrics=metrics, checkpoint=checkpoint)

        # Add a barrier to ensure all workers finished reporting here
        trainer.strategy.barrier()

        if self.local_rank == 0:
            shutil.rmtree(tmpdir)


class TrainerReportKeepOnlyLastCallback(L.Callback):

    CHECKPOINT_NAME = "checkpoint.ckpt"

    def __init__(self) -> None:
        super().__init__()
        self.trial_name = train.get_context().get_trial_name()
        self.local_rank = train.get_context().get_local_rank()
        self.tmpdir_prefix = Path(tempfile.gettempdir(), self.trial_name).as_posix()
        if os.path.isdir(self.tmpdir_prefix) and self.local_rank == 0:
            shutil.rmtree(self.tmpdir_prefix)

        record_extra_usage_tag(TagKey.TRAIN_LIGHTNING_RAYTRAINREPORTCALLBACK, "1")

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        # Fetch metrics
        metrics = trainer.callback_metrics
        metrics = {k: v.item() for k, v in metrics.items()}

        # (Optional) Add customized metrics
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = trainer.global_step

        tmpdir = Path(self.tmpdir_prefix, "last").as_posix()
        os.makedirs(tmpdir, exist_ok=True)

        # Delete previous checkpoint
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir)

        # Save checkpoint to local
        ckpt_path = Path(tmpdir, self.CHECKPOINT_NAME).as_posix()
        trainer.save_checkpoint(ckpt_path, weights_only=False)

        # Report to train session
        checkpoint = Checkpoint.from_directory(tmpdir)
        train.report(metrics=metrics, checkpoint=checkpoint)

        # Add a barrier to ensure all workers finished reporting here
        trainer.strategy.barrier()

        if self.local_rank == 0:
            shutil.rmtree(tmpdir)
