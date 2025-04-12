import torch
import numpy as np
from torch.nn.modules.loss import _Loss


class NTXentLoss_poly(_Loss):
    """
    Loss function used on the pretraining of the TFC model. It is based on the NTXentLoss, but it includes a polynomial loss term.
    """

    def __init__(
        self,
        device: str,
        batch_size: int,
        temperature: float,
        use_cosine_similarity: bool,
    ):
        """
        The constructor of the NTXentLoss_poly class.

        Parameters
        ----------
        - device: str
            The device to be used in the training of the model
        - batch_size: int
            The batch size of the model
        - temperature: float
            The temperature of the softmax function
        - use_cosine_similarity: bool
            If True, the cosine similarity is used. If False, the dot product is used

        """
        super(NTXentLoss_poly, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity: bool):
        """
        Define the similarity function to be used in the loss calculation.

        Parameters
        ----------
        - use_cosine_similarity: bool
            If True, the cosine similarity is used. If False, the dot product is used

        Returns
        -------
        - function
            The similarity function to be used in the loss calculation

        """
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self) -> torch.Tensor:
        """
        Get the mask of correlated samples.

        Returns
        -------
        - torch.Tensor
            The mask of correlated samples

        """
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y) -> torch.Tensor:
        """
        Function to calculate the dot similarity between two tensors.

        Parameters
        ----------
        - x: torch.Tensor
            The first tensor
        - y: torch.Tensor
            The second tensor

        Returns
        -------
        - torch.Tensor
            The dot similarity between the two tensors


        """

        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y) -> torch.Tensor:
        """
        Function to calculate the cosine similarity between two tensors.

        Parameters
        ----------
        - x: torch.Tensor
            The first tensor
        - y: torch.Tensor
            The second tensor

        Returns
        -------
        - torch.Tensor
            The cosine similarity between the two tensors

        """
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis: torch.Tensor, zjs: torch.Tensor) -> _Loss:
        """
        The forward method of the NTXentLoss_poly class. It receives the samples and returns the loss of the model.

        Parameters
        ----------
        - zis: torch.Tensor
            The positive samples
        - zjs: torch.Tensor
            The negative samples

        Returns
        -------
        - _Loss
            The loss of the model

        """
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        try:
            positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
            # RuntimeError: shape '[128, 1]' is invalid for input of size 672
        except RuntimeError as e:
            # mostra o tipo de e
            if "is invalid for input of size" in e.args[0]:
                raise RuntimeError(
                    f"Maybe you missed the batch size of the loss or set the drop_last to False. You should only use dataloaders with drop_last = True"
                ) from e
            else:
                raise e

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(
            2 * self.batch_size, -1
        )

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        CE = self.criterion(logits, labels)

        onehot_label = (
            torch.cat(
                (
                    torch.ones(2 * self.batch_size, 1),
                    torch.zeros(2 * self.batch_size, negatives.shape[-1]),
                ),
                dim=-1,
            )
            .to(self.device)
            .long()
        )
        # Add poly loss
        pt = torch.mean(onehot_label * torch.nn.functional.softmax(logits, dim=-1))

        epsilon = self.batch_size
        # loss = CE/ (2 * self.batch_size) + epsilon*(1-pt) # replace 1 by 1/self.batch_size
        loss = CE / (2 * self.batch_size) + epsilon * (1 / self.batch_size - pt)
        # loss = CE / (2 * self.batch_size)

        return loss
