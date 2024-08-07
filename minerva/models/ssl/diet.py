import lightning as L
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
from torch import nn
from typing import Optional, Callable

class DIET(L.LightningModule):
    def __init__(
            self,
            encoder: nn.Module,
            linear_layer: nn.Module,
            loss: Optional[Callable]=None,
            learning_rate: float=1e-3):
        """
        DIET model.

        Parameters
        ----------
        encoder : torch.nn.Module
            Encoder model
        linear_layer : torch.nn.Module
            Linear layer model. It receives the output of the encoder and outputs the logits used in the cross entropy loss
        loss : Callable, optional
            Reconstruction loss, by default None
        learning_rate : float, optional
            Learning rate, by default 1e-3
        """
        super(DIET, self).__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        # Defining layers
        self.encoder = encoder
        self.linear_layer = linear_layer
        # Defining reconstruction loss
        self.loss = loss if loss is not None else CrossEntropyLoss(label_smoothing=0.8)

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)