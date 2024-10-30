import torch
from torchmetrics import Metric


class CroppedMetric(Metric):
    def __init__(
        self,
        target_h_size: int,
        target_w_size: int,
        metric: Metric,
        dist_sync_on_step: bool = False,
    ):
        """
        Initializes a new instance of CroppedMetric.

        Parameters
        ----------
            target_h_size: int
                The target height size.
            target_w_size: int
                The target width size.
            dist_sync_on_step: bool, optional
                Whether to synchronize metric state across processes at each step.
                Defaults to False.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.metric = metric
        self.target_h_size = target_h_size
        self.target_w_size = target_w_size

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

        preds = self.crop(preds)
        target = self.crop(target)
        self.metric.update(preds, target)

    def compute(self) -> float:
        """
        Computes the cropped metric.

        Returns:
            float: The cropped metric.
        """
        return self.metric.compute()

    def crop(self, x: torch.Tensor) -> torch.Tensor:
        """crops the input tensor to the target size.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The cropped tensor.
        """
        h, w = x.shape[-2:]
        start_h = (h - self.target_h_size) // 2
        start_w = (w - self.target_w_size) // 2
        end_h = start_h + self.target_h_size
        end_w = start_w + self.target_w_size
        return x[..., start_h:end_h, start_w:end_w]
