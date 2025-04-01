import torch
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss


class NTXentLoss(_Loss):

    def __init__(self, temperature: float):
        """
        The constructor of the NTXentLoss class.

        Parameters
        ----------
        - temperature: float
            The temperature of the softmax function

        """
        super(NTXentLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.eps = 1e-8

        if abs(temperature) < self.eps:
            raise ValueError(
                f"Temperature must be greater than 1e-8, got {temperature}"
            )

        self.temperature = temperature

    def forward(self, y_0: Tensor, y_1: Tensor) -> Tensor:
        """
        Forward pass of the NTXentLoss class.

        Based on Lightly SSL's implementation.

        Parameters
        ----------
        - y_0: Tensor
            The first tensor to be compared
        - y_1: Tensor
            The second tensor to be compared

        Returns
        -------
        - Tensor
            The loss value
        """

        device = y_0.device
        batch_size, _ = y_0.shape

        # Normalize the output to length 1
        y_0 = nn.functional.normalize(y_0, dim=1)
        y_1 = nn.functional.normalize(y_1, dim=1)

        diag_mask = torch.eye(batch_size, device=device, dtype=torch.bool)

        # Calculate similarities
        # Here n = batch_size and m = batch_size * world_size
        # The resulting vectors have shape (n, m)
        logits_aa = torch.einsum("nc, mc -> nm", y_0, y_0) / self.temperature
        logits_ab = torch.einsum("nc, mc -> nm", y_0, y_1) / self.temperature
        logits_ba = torch.einsum("nc, mc -> nm", y_1, y_0) / self.temperature
        logits_bb = torch.einsum("nc, mc -> nm", y_1, y_1) / self.temperature

        # Remove similarities between same views of the same image
        logits_aa = logits_aa[~diag_mask].view(batch_size, -1)
        logits_bb = logits_bb[~diag_mask].view(batch_size, -1)

        # Concatenate logits
        # The logits tensor in the end has shape (2*n, 2*m-1)
        logits_abaa = torch.cat([logits_ab, logits_aa], dim=1)
        logits_babb = torch.cat([logits_ba, logits_bb], dim=1)
        logits = torch.cat([logits_abaa, logits_babb], dim=0)

        # Create labels
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        labels = labels.repeat(2)

        # Compute loss
        loss = self.criterion(logits, labels)

        return loss
