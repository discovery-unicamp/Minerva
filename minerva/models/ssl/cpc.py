from torch import nn, optim
import lightning as L
import torch
from torch.optim.lr_scheduler import StepLR

class CPC(L.LightningModule):
    """Implements the Contrastive Predictive Coding (CPC) model, as described in
    https://dl.acm.org/doi/10.1145/3463506. The implementation was adapted from
    https://github.com/harkash/contrastive-predictive-coding-for-har
    """

    def __init__(
        self,
        g_enc: L.LightningModule,
        g_ar: L.LightningModule,
        prediction_head: L.LightningModule,
        num_steps_prediction: int = 28,
        batch_size: int = 64,
        learning_rate: float = 5e-4,
    ):
        """
        Parameters
        ----------
        g_enc :  L.LightningModule
            Encoder network that processes the input sequence and extracts features.
        g_ar :  L.LightningModule
            Autoregressive network that models the temporal dependencies in the feature space.
        prediction_head : L.LightningModule
            Network used to predict future representations.
        num_steps_prediction : int, optional
            Number of future steps to predict, by default 28.
        batch_size : int, optional
            Size of the batch, by default 64.
        learning_rate : float, optional
            Learning rate for the optimizer, by default 5e-4.
        """
        super(CPC, self).__init__()
        self.g_enc = g_enc
        self.g_ar = g_ar
        self.prediction_head = prediction_head
        self.predictors = nn.ModuleList([self.prediction_head for _ in range(num_steps_prediction)])
        self.learning_rate = learning_rate
        self.softmax = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.batch_size = batch_size
        self.num_steps_prediction = num_steps_prediction

    def forward(self, x):
        z = self.g_enc(x)
        S = x.size(2)
        start = (
            torch.randint(int(S - self.num_steps_prediction), size=(1,)).long().item()
        )
        rnn_input = z[:, : start + 1, :]
        r_out, (_) = self.g_ar(rnn_input, None)
        return z, r_out, start

    def compute_nce_loss(self, z, r_out, start):
        batch_size = z.size(0)
        context = r_out[:, start, :].squeeze(1)
        y_pred = torch.stack(
            [
                self.predictors[pred](context)
                for pred in range(self.num_steps_prediction)
            ]
        )
        y_truth = z[:, start + 1 : start + 1 + self.num_steps_prediction, :].permute(
            1, 0, 2
        )
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
        z, r_out, start = self.forward(batch[0])
        nce = self.compute_nce_loss(z, r_out, start)
        self.log("train_loss", nce)
        return nce

    def validation_step(self, batch, batch_idx):
        z, r_out, start = self.forward(batch[0])
        nce = self.compute_nce_loss(z, r_out, start)
        self.log("val_loss", nce)
        return nce

    def test_step(self, batch, batch_idx):
        z, r_out, start = self.forward(batch[0])
        nce = self.compute_nce_loss(z, r_out, start)
        self.log("test_loss", nce)
        return nce

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.8)
        return [optimizer], [scheduler]