from torch.nn.modules.loss import _Loss
import torch
import numpy as np
import torch.nn.functional as F


class BatchWiseBarlowTwinLoss(_Loss):
    """
    Implementation of the Batch-wise Barlow Twins loss function (https://arxiv.org/abs/2310.07756).
    """

    def __init__(self, diag_lambda: float = 0.01, normalize: bool = False):
        """
        Initialize the BatchWiseBarlowtwinsLoss class.

        Parameters
        ----------
        diag_lambda: float
            The value of the diagonal lambda parameter. By default, 0.01.
        normalize: bool
            Whether to normalize the loss or not. By default, False.
        """
        super().__init__()
        self.diag_lambda = diag_lambda
        self.normalize = normalize

    def forward(self, prediction_data, projection_data):
        """
        Calculate the loss between the prediction and projection data. This implementation uses a batch-wise
        version of the Barlow Twins loss function.

        Parameters
        ----------
        prediction_data : torch.Tensor
            The prediction data.
        projection_data : torch.Tensor
            The projection data.
        """
        return self.bt_loss_bs(
            prediction_data, projection_data, self.diag_lambda, self.normalize
        )

    def bt_loss_bs(self, p, z, lambd=0.01, normalize=False):
        # barlow twins loss but in batch dims
        c = torch.matmul(F.normalize(p), F.normalize(z).T)
        assert c.min() > -1 and c.max() < 1
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + lambd * off_diag
        if normalize:
            loss = loss / p.shape[0]
        return loss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
