import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor
import torch.distributed as dist


def _off_diagonal(x):
    """
    Returns a flattened view of the off-diagonal elements of a square matrix.

    Parameters
    ----------
    x : Tensor
        A square 2D tensor (cross-correlation matrix).

    Returns
    -------
    Tensor
        A 1D tensor containing the flattened off-diagonal elements of the input matrix.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _normalize(z_a: Tensor, z_b: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Normalizes each embedding tensor independently across the batch.

    Parameters
    ----------
    z_a : Tensor
        Embeddings from the first view.
    z_b : Tensor
        Embeddings from the second view.

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple containing the normalized versions of `z_a` and `z_b`.
    """
    combined = torch.stack([z_a, z_b], dim=0)  # Shape: 2 x N x D
    normalized = F.batch_norm(
        combined.flatten(0, 1),
        running_mean=None,
        running_var=None,
        weight=None,
        bias=None,
        training=True,
    ).view_as(combined)
    return normalized[0], normalized[1]


class BatchWiseBarlowTwinLoss(_Loss):
    """
    Implementation of the Batch-wise Barlow Twins loss function (https://arxiv.org/abs/2310.07756).
    """

    def __init__(self, diag_lambda: float = 0.01, normalize: bool = False):
        """
        Initializes the BatchWiseBarlowTwinLoss class.

        Parameters
        ----------
        diag_lambda : float, optional
            The value of the diagonal lambda parameter. Default is 0.01.
        normalize : bool, optional
            Whether to normalize the loss. Default is False.
        """
        super().__init__()
        self.diag_lambda = diag_lambda
        self.normalize = normalize

    def forward(self, prediction_data, projection_data):
        """
        Calculates the loss between the prediction and projection data using a batch-wise
        version of the Barlow Twins loss function.

        Parameters
        ----------
        prediction_data : torch.Tensor
            Prediction data tensor.
        projection_data : torch.Tensor
            Projection data tensor.

        Returns
        -------
        torch.Tensor
            The computed batch-wise Barlow Twins loss.
        """
        return self.bt_loss_bs(
            prediction_data, projection_data, self.diag_lambda, self.normalize
        )

    def bt_loss_bs(self, p, z, lambd=0.01, normalize=False):
        # barlow twins loss but in batch dims
        c = torch.matmul(F.normalize(p), F.normalize(z).T)
        assert c.min() > -1 and c.max() < 1
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = _off_diagonal(c).pow_(2).sum()
        loss = on_diag + lambd * off_diag
        if normalize:
            loss = loss / p.shape[0]
        return loss


class BarlowTwinsLoss(torch.nn.Module):
    """
    Implementation of the Barlow Twins loss function for self-supervised learning.

    The loss encourages embeddings of two augmented views of the same input to be
    similar (invariance) while reducing redundancy between the components of
    their representations (decorrelation).
    """

    def __init__(self, lambda_param: float = 5e-3, gather_distributed: bool = False):
        """
        Initializes the BarlowTwinsLoss module.

        Parameters
        ----------
        lambda_param : float, optional
            Coefficient for off-diagonal penalty in the loss. Defaults to 5e-3.
        gather_distributed : bool, optional
            If True, performs all-reduce on the cross-correlation matrix across GPUs. Defaults to False.

        Raises
        ------
        ValueError
            If gather_distributed is True but torch.distributed is not available.
        """
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.gather_distributed = gather_distributed

        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        """
        Computes the Barlow Twins loss.

        Parameters
        ----------
        z_a : Tensor
            Embedding tensor from the first view. Shape: [batch_size, dim].
        z_b : Tensor
            Embedding tensor from the second view. Shape: [batch_size, dim].

        Returns
        -------
        Tensor
            Scalar loss value combining invariance and redundancy reduction terms.
        """
        # normalize repr. along the batch dimension
        z_a_norm, z_b_norm = _normalize(z_a, z_b)

        N = z_a.size(0)

        # cross-correlation matrix
        c = z_a_norm.T @ z_b_norm
        c.div_(N)

        # sum cross-correlation matrix between multiple gpus
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                c = c / world_size
                dist.all_reduce(c)

        invariance_loss = torch.diagonal(c).add_(-1).pow_(2).sum()
        redundancy_reduction_loss = _off_diagonal(c).pow_(2).sum()
        loss = invariance_loss + self.lambda_param * redundancy_reduction_loss

        return loss
