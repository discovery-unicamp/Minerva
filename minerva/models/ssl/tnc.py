from torch import nn, optim
import lightning as L

class TNC(L.LightningModule):
    """Implements the TNC technique, as described in ... 
    """
    def __init__(
        self,
        encoder: nn.Module,
        discriminator: nn.Module, 
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        # TODO: add other parameters
    ):
        self.encoder = encoder
        self.discriminator = discriminator
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    # TODO: Implement the forward method.
    def forward(self, x):
        raise NotImplementedError("Must be implemented")

    # TODO: Implement the training step (code invoked by Lightning Trainer).
    def training_step(self, batch, batch_idx):
        # Remember to invoke self.log("train_loss", loss) to report the training loss
        raise NotImplementedError("Must be implemented")

    # TODO: Implement the validation step (code invoked by Lightning Trainer).
    def validation_step(self, batch, batch_idx):
        # Remember to invoke self.log("val_loss", loss) to report the validation loss
        raise NotImplementedError("Must be implemented")

    # TODO: Implement the test step (code invoked by Lightning Trainer).
    def test_step(self, batch, batch_idx):
        # Remember to invoke self.log("test_loss", loss) to report the test loss
        raise NotImplementedError("Must be implemented")

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
