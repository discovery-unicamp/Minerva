import warnings
from typing import Optional

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


class ResizedMetric(Metric):
    def __init__(
        self,
        target_h_size: Optional[int],
        target_w_size: Optional[int],
        metric: Metric,
        keep_aspect_ratio: bool = False,
        dist_sync_on_step: bool = False,
    ):
        """
        Initializes a new instance of ResizeMetric.

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

        if target_h_size is None and target_w_size is None:
            raise ValueError(
                "At least one of target_h_size or target_w_size must be provided."
            )

        if (
            target_h_size is not None and target_w_size is None
        ) and keep_aspect_ratio is False:
            warnings.warn(
                "A target_w_size is not provided, but keep_aspect_ratio is set to False. keep_aspect_ratio will be set to True. If you want to resize the image to a specific width, please provide a target_w_size."
            )
            keep_aspect_ratio = True

        if (
            target_w_size is not None and target_h_size is None
        ) and keep_aspect_ratio is False:
            warnings.warn(
                "A target_h_size is not provided, but keep_aspect_ratio is set to False. keep_aspect_ratio will be set to True. If you want to resize the image to a specific height, please provide a target_h_size."
            )
            keep_aspect_ratio = True

        self.metric = metric
        self.target_h_size = target_h_size
        self.target_w_size = target_w_size
        self.keep_aspect_ratio = keep_aspect_ratio

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

        preds = self.resize(preds)
        target = self.resize(target)
        self.metric.update(preds, target)

    def compute(self) -> float:
        """
        Computes the resized metric.

        Returns:
            float: The resized metric.
        """
        return self.metric.compute()

    def resize(self, x: torch.Tensor) -> torch.Tensor:
        """Resizes the input tensor to the target size.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The resized tensor.
        """
        h, w = x.shape[-2:]

        target_h_size = self.target_h_size
        target_w_size = self.target_w_size
        if self.keep_aspect_ratio:
            if self.target_h_size is None:
                scale = target_w_size / w
                target_h_size = int(h * scale)
            elif self.target_w_size is None:
                scale = target_h_size / h
                target_w_size = int(w * scale)
        type_convert = False
        if "LongTensor" in x.type():
            x = x.to(torch.uint8)
            type_convert = True

        return (
            torch.nn.functional.interpolate(x, size=(target_h_size, target_w_size))
            if not type_convert
            else torch.nn.functional.interpolate(
                x, size=(target_h_size, target_w_size)
            ).to(torch.long)
        )
