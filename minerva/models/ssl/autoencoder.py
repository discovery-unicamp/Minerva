import lightning as L
from torch.nn import MSELoss
from torch.optim import Adam
import torch
from torch import nn
from typing import Optional, Callable


class Autoencoder(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        loss: Optional[Callable] = None,
        learning_rate: float = 1e-3,
    ):
        """
        Autoencoder model.

        Parameters
        ----------
        encoder : torch.nn.Module
            Encoder model
        decoder : torch.nn.Module
            Decoder model
        loss : Callable, optional
            Reconstruction loss, by default None
        learning_rate : float, optional
            Learning rate, by default 1e-3
        """
        super(Autoencoder, self).__init__()
        # Saving parameters
        self.learning_rate = learning_rate
        # Defining layers
        self.encoder = encoder
        self.decoder = decoder
        # Defining reconstruction loss
        self.reconstruction_loss = loss if loss is not None else MSELoss()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.reconstruction_loss(x, x_hat)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.reconstruction_loss(x, x_hat)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
