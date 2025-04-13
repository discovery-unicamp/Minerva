import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import lightning as L
from minerva.models.nets.cpc_networks import PredictionNetwork


class CPC(L.LightningModule):
    """
    Implements the Contrastive Predictive Coding (CPC) model, as described in
    https://dl.acm.org/doi/10.1145/3463506. The implementation was adapted from
    https://github.com/harkash/contrastive-predictive-coding-for-har.
    """

    def __init__(
        self,
        g_enc: nn.Module,
        g_ar: nn.Module,
        prediction_head_in_channels: int,
        prediction_head_out_channels: int,
        num_steps_prediction: int = 28,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        minimum_steps: int = 2,  # Hack to allow a minimum number of steps
    ):
        """
        Parameters
        ----------
        g_enc : nn.Module
            Encoder network that processes the input sequence and extracts features.
        g_ar : nn.Module
            Autoregressive network that models the temporal dependencies in the feature space.
        prediction_head_in_channels : int
            Number of input channels for the prediction head.
        prediction_head_out_channels : int
            Number of output channels for the prediction head.
        num_steps_prediction : int, optional
            Number of future steps to predict, by default 28.
        batch_size : int, optional
            Size of the batch, by default 64.
        learning_rate : float, optional
            Learning rate for the optimizer, by default 1e-3.
        minimum_steps : int, optional
            Minimum number of steps for predictions, by default 2.
        """
        super(CPC, self).__init__()
        self.g_enc = g_enc
        self.g_ar = g_ar
        self.minimum_steps = minimum_steps
        self.predictors = nn.ModuleList(
            [
                PredictionNetwork(
                    prediction_head_in_channels, prediction_head_out_channels
                )
                for i in range(num_steps_prediction)
            ]
        )
        self.learning_rate = learning_rate
        self.softmax = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.batch_size = batch_size
        self.num_steps_prediction = num_steps_prediction

    def forward(self, x):
        s = x.size(2)
        start = (
            torch.randint(
                high=int(s - self.num_steps_prediction),
                size=(1,),
                low=self.minimum_steps,
            )
            .long()
            .item()
        )
        z = self.g_enc(x)
        rnn_input = z[:, : start + 1, :]
        r_out, _ = self.g_ar(rnn_input, None)
        r_out = r_out[:, -1, :]
        y = torch.stack([pred(r_out) for pred in self.predictors], dim=1)
        return z, y, start

    def compute_nce_loss(self, z, y_pred, start):
        batch_size = z.size(0)
        y_truth = z[:, start + 1 : start + 1 + self.num_steps_prediction, :].permute(
            1, 0, 2
        )
        y_pred = y_pred.permute(1, 0, 2)

        nce = 0
        correct = 0

        for k in range(self.num_steps_prediction):
            log_density_ratio = torch.mm(y_truth[k], y_pred[k].transpose(0, 1))
            positive_batch_pred = torch.argmax(self.softmax(log_density_ratio), dim=0)
            positive_batch_actual = torch.arange(0, batch_size).to(self.device)
            correct += torch.sum(
                torch.eq(positive_batch_pred, positive_batch_actual)
            ).item()
            nce += torch.sum(torch.diag(self.lsoftmax(log_density_ratio)))

        nce = nce / (-1.0 * batch_size * self.num_steps_prediction)
        return nce

    def training_step(self, batch, batch_idx):
        z, y_pred, start = self.forward(batch[0])
        nce = self.compute_nce_loss(z, y_pred, start)
        self.log("train_loss", nce)
        return nce

    def validation_step(self, batch, batch_idx):
        z, y_pred, start = self.forward(batch[0])
        nce = self.compute_nce_loss(z, y_pred, start)
        self.log("val_loss", nce)
        return nce

    def test_step(self, batch, batch_idx):
        z, y_pred, start = self.forward(batch[0])
        nce = self.compute_nce_loss(z, y_pred, start)
        self.log("test_loss", nce)
        return nce

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.8)
        return [optimizer], [scheduler]
