import lightning as L
import torch


class TNC(L.LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        projection_head: torch.nn.Module,
        loss_fn: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        learning_rate: float = 0.00001,
        w: float = 0.05,
    ):
        """
        This class defines a the TNC technique using a backbone neural network (backbone)
        and a projection head neural network (projection_head) for training.

        Parameters:
        -----------
        - backbone (torch.nn.Module):
            The feature extraction backbone network, such as an encoder or a pre-trained model.
        - projection_head (torch.nn.Module):
            The projection head network that maps feature vectors into a latent space.
        - loss_fn (torch.nn.Module, optional):
            The loss function used for training (default: torch.nn.BCEWithLogitsLoss()).
        - learning_rate (float, optional):
            The learning rate for the optimizer (default: 0.00001).
        - w (float, optional):
            The weight for the negative loss term (default: 0.05).
        Examples:
        ----------
        >>> backbone = RnnEncoder(hidden_size=100, in_channel=6, encoding_size=320, cell_type='GRU', num_layers=1)
        >>> projection_head = Discriminator_TNC(input_size=320, max_pool=False)
        >>> model = TNC(backbone=backbone, projection_head=projection_head, learning_rate=0.0001, w=0.1)

        >>> backbone = TSEncoder(input_dims=6, output_dims=320, hidden_dims=64, depth=10)
        >>> projection_head = Discriminator_TNC(input_size=320, max_pool=True)
        >>> model = TNC(backbone=backbone, projection_head=projection_head, loss_fn=torch.nn.CrossEntropyLoss(), learning_rate=0.001, w=0.05)
        """

        super(TNC, self).__init__()
        self.backbone = backbone
        self.projection_head = projection_head
        self.loss_fn = loss_fn
        self.mc_sample_size = 5
        self.learning_rate = learning_rate
        self.w = w

    def forward(self, x_t, X_close, X_distant):
        """
        Forward pass through the backbone and projection head networks.

        Parameters:
        -----------
        x_t : torch.Tensor
            Tensor of shape (batch_size, seq_length, feature_size) representing the primary input.
        X_close : torch.Tensor
            Tensor of shape (batch_size, seq_length, feature_size) representing close samples.
        X_distant : torch.Tensor
            Tensor of shape (batch_size, seq_length, feature_size) representing distant samples.

        Returns:
        --------
        d_p : torch.Tensor
            Projection head output for positive (x_t, X_close) pairs.
        d_n : torch.Tensor
            Projection head output for negative (x_t, X_distant) pairs.
        """
        x_t_encoded = self.backbone(x_t)
        batch_size, len_size, f_size = x_t.shape
        x_t_encoded = torch.repeat_interleave(x_t_encoded, self.mc_sample_size, axis=0)
        X_close = X_close.reshape((-1, len_size, f_size))
        X_close_encoded = self.backbone(X_close)

        X_distant = X_distant.reshape((-1, len_size, f_size))

        X_distant_encoded = self.backbone(X_distant)
        d_p = self.projection_head(x_t_encoded, X_close_encoded)
        d_n = self.projection_head(x_t_encoded, X_distant_encoded)

        return d_p, d_n

    def training_step(self, batch):
        """
        Single training step using contrastive loss.

        Parameters:
        -----------
        - batch (tuple of torch.Tensor):
            Batch of input data consisting of (x_t, X_close, X_distant).

        Returns:
        --------
        - loss (torch.Tensor):
            Calculated contrastive loss for the current batch.
        """
        # print(f'batch: {batch}\n batch[0]: {batch[0]}\n batch size: batch.size\n batch[0] shape: {batch[0].shape}\n')
        x_t, X_close, X_distant = batch

        # print(x_t.shape)
        batch_size = x_t.size(0)

        # Create target tensors and repeat them to match the repeated predictions
        neighbors = torch.ones((batch_size * self.mc_sample_size), device=self.device)
        non_neighbors = torch.zeros(
            (batch_size * self.mc_sample_size), device=self.device
        )
        # print(f'shape of x_t, X_close, X_distant: {x_t.shape},{X_close.shape},{X_distant.shape}')
        d_p, d_n = self.forward(x_t, X_close, X_distant)
        p_loss = self.loss_fn(d_p, neighbors)
        n_loss = self.loss_fn(d_n, non_neighbors)
        n_loss_u = self.loss_fn(d_n, neighbors)

        loss = (p_loss + self.w * n_loss_u + (1 - self.w) * n_loss) / 2

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Single validation step using contrastive loss.

        Parameters:
        -----------
        - batch (tuple of torch.Tensor):
            Batch of input data consisting of (x_t, X_close, X_distant).

        Returns:
        --------
        - val_loss (torch.Tensor):
            Calculated contrastive loss for the current batch.
        """
        x_t, X_close, X_distant = batch
        batch_size = x_t.size(0)

        neighbors = torch.ones((batch_size * self.mc_sample_size), device=self.device)
        non_neighbors = torch.zeros(
            (batch_size * self.mc_sample_size), device=self.device
        )

        d_p, d_n = self.forward(x_t, X_close, X_distant)
        p_loss = self.loss_fn(d_p, neighbors)
        n_loss = self.loss_fn(d_n, non_neighbors)
        n_loss_u = self.loss_fn(d_n, neighbors)

        val_loss = (p_loss + self.w * n_loss_u + (1 - self.w) * n_loss) / 2

        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optim
