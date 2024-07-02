import lightning as L
from torch.nn import MSELoss
from torch.optim import Adam
from minerva.losses.topo_loss import TopologicalLoss

class TopologicalAutoencoder(L.LightningModule):
    def __init__(self, encoder, decoder, topological_loss=None, reconstruction_loss=None, lambda_param=1e-3,learning_rate=1e-3):
        """
        Topological autoencoder model.

        Parameters
        ----------
        encoder : torch.nn.Module
            Encoder model
        decoder : torch.nn.Module
            Decoder model
        topological_loss : torch.nn.Module, optional
            Topological loss, by default None
        reconstruction_loss : torch.nn.Module, optional
            Reconstruction loss, by default None
        lambda_param : float, optional
            Weight of the topological loss, by default 1e-3
        learning_rate : float, optional
            Learning rate, by default 1e-3
        """
        super(TopologicalAutoencoder, self).__init__()
        # Saving parameters
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        # Defining layers
        self.encoder = encoder
        self.decoder = decoder
        # Defining topological loss
        self.topological_loss = topological_loss if topological_loss is not None else TopologicalLoss()
        # Defining reconstruction loss
        self.reconstruction_loss = reconstruction_loss if reconstruction_loss is not None else MSELoss()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_encoded = self.encoder(x)
        x_hat = self.decoder(x_encoded)
        loss = self.reconstruction_loss(x, x_encoded) + self.lambda_param*self.topological_loss(x, x_hat)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_encoded = self.encoder(x)
        x_hat = self.decoder(x_encoded)
        loss = self.reconstruction_loss(x, x_encoded) + self.lambda_param*self.topological_loss(x, x_hat)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)