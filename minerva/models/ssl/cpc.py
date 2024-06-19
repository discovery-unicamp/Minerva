from torch import nn, optim
import lightning as L

class CPC(L.LightningModule):
    """Implements the Contrastive Predictive Coding (CPC) model, as described in
    https://arxiv.org/abs/1807.03748. The implementation was adapted from
    https://github.com/sanatonek/TNC_representation_learning
    """
    def __init__(
        self,
        encoder: nn.Module,
        density_estimator: nn.Module,
        auto_regressor: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        # TODO: add other parameters
    ):
        self.encoder = encoder
        self.density_estimator = density_estimator
        self.auto_regressor = auto_regressor
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_func = nn.CrossEntropyLoss()

    # TODO: Implement the training step (code invoked by Lightning Trainer).
    def forward(self, x):
        z = self.encoder(x)
        c = self.auto_regressor(z)
        _, c_t = self.auto_regressor(z) # TODO: Verify this code. Is it retrieving c_t?
        c_t = c_t.squeeze(1).squeeze(0)  # Equivalent to c_t.view(-1)
        return c_t
    
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
