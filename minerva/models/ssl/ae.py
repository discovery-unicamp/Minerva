import lightning as L
from torch.nn import MSELoss
from torch.optim import Adam

class Autoencoder(L.LightningModule):
    def __init__(self, encoder, decoder, loss=None, learning_rate=1e-3):
        """
        Autoencoder model.

        Parameters
        ----------
        encoder : torch.nn.Module
            Encoder model
        decoder : torch.nn.Module
            Decoder model
        loss : torch.nn.Module, optional
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
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.reconstruction_loss(x, x_hat)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)