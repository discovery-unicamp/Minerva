import torch
from torchmetrics import Metric


class PixelAccuracy(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        """
        Initializes a PixelAccuracy metric object.

        Parameters
        ----------
            dist_sync_on_step: bool, optional
                Whether to synchronize metric state across processes at each step.
                Defaults to False.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Updates the metric state with the predictions and targets.

        Parameters
        ----------
            preds: torch.Tensor
                The predicted tensor.
            target:
                torch.Tensor The target tensor.
        """
        correct = torch.sum(preds == target)
        total = target.numel()
        self.correct += correct
        self.total += total

    def compute(self) -> float:
        """
        Computes the pixel accuracy.

        Returns:
            float: The pixel accuracy.
        """
        return self.correct.float() / self.total
