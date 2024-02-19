import lightning as L
from lightning.pytorch.callbacks import Callback
import time


class PerformanceLogger(Callback):
    """This callback logs the time taken for each epoch and the overall fit 
    time.
    """
    def __init__(self):
        super().__init__()
        self.train_epoch_start_time = None
        self.fit_start_time = None

    def on_train_epoch_start(
        self, trainer: L.Trainer, module: L.LightningModule
    ):
        """Called when the train epoch begins."""
        self.train_epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer: L.Trainer, module: L.LightningModule):
        """Called when the train epoch ends.
        """
        end = time.time()
        duration = end - self.train_epoch_start_time
        module.log(
            "train_epoch_time",
            duration,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        self.train_epoch_start_time = end

    def on_fit_start(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        """Called when fit begins."""
        self.fit_start_time = time.time()

    def on_fit_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        """Called when fit ends."""
        end = time.time()
        duration = end - self.fit_start_time
        print(f"--> Overall fit time: {duration:.3f} seconds")