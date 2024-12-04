import lightning as L
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import torch
from torch import nn
from typing import Optional, Callable


import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]


class DIET(L.LightningModule):
    def __init__(
            self,
            encoder: nn.Module,
            linear_layer: nn.Module,
            loss: Optional[Callable]=None,
            learning_rate: float=1e-3,
            weight_decay: float=5e-2,
            scheduler: Optional[str]=None,
            cosine_annealing_total_epochs: int=500
        ):
        """
        DIET model.

        Parameters
        ----------
        encoder : torch.nn.Module
            Encoder model.
        linear_layer : torch.nn.Module
            Linear layer model. It receives the output of the encoder and outputs the logits used in the cross entropy loss.
        loss : Callable, optional
            Classification loss , by default None. If None, CrossEntropyLoss with 0.8 label smoothing is used.
        learning_rate : float, optional
            Learning rate used in the optimizer, by default 1e-3.
        weight_decay : float, optional
            Weight decay used in the optimizer, by default 5e-2.
        scheduler : str, optional
            Learning rate scheduler, by default None. If None, no scheduler is sent to Lightning. If 'WarmupCosineAnnealingLR', a
            WarmupCosineAnnealingLR scheduler is sent to Lightning. If any other value, a ValueError is raised.
        cosine_annealing_total_epochs : int
            Total number of epochs used in the scheduler, by default 500.
            This parameter is only used if scheduler is 'WarmupCosineAnnealingLR'.
        """
        super(DIET, self).__init__()
        # Saving hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # Defining layers
        self.encoder = encoder
        self.linear_layer = linear_layer
        # Defining reconstruction loss
        self.loss = loss if loss is not None else CrossEntropyLoss(label_smoothing=0.8)
        self.scheduler = scheduler
        self.cosine_annealing_total_epochs = cosine_annealing_total_epochs

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # If self.scheduler is None, we return only the optimizer
        if not self.scheduler:
            return optimizer
        elif self.scheduler == 'WarmupCosineAnnealingLR':
            # If self.scheduler is WarmupCosineAnnealingLR, we return the optimizer and the scheduler
            scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=10, total_epochs=self.cosine_annealing_total_epochs)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                    'reduce_on_plateau': False,
                    'monitor': 'train_loss'
                }
            }
        else:
            # If self.scheduler is not None or WarmupCosineAnnealingLR, we raise an error
            raise ValueError('Invalid scheduler')