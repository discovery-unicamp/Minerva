from torch.optim.lr_scheduler import _LRScheduler
import math
from torch.optim.optimizer import Optimizer


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    A custom learning rate scheduler that combines linear warmup with cosine
    annealing. The learning rate increases linearly over the first 'warmup_epochs',
    and then decreases until 'total_epochs' following a cosine curve.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: int = 0,
        last_epoch: int = -1,
    ):
        """
        Initializes the scheduler.

        Parameters
        ----------
        optimizer : torch.optim.optimizer.Optimizer
            Wrapped optimizer.
        warmup_epochs : int
            Number of epochs for linear warmup.
        total_epochs : int
            Total number of training epochs.
        min_lr : float
            Minimum learning rate expected at the end of the cosine annealing.
        last_epoch: int
            Index of the last epoch. Used for resuming training.
        """
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        if warmup_epochs > total_epochs:
            raise ValueError("total_epochs must be greater than warmup_epochs.")

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]
