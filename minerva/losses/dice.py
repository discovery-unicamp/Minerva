from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from typing import Union

from minerva.losses._functional import dice_score
from minerva.utils.tensor import to_tensor

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


# Borrowed from https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/dice.py
class DiceLoss(_Loss):
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        """
        Initialize the DiceLoss class.

        Parameters
        ----------
        mode : str
            Loss mode. Valid options are 'binary', 'multiclass', or 'multilabel'.
        classes : Optional[List[int]], optional
            List of classes that contribute in loss computation. By default, all channels are included. By default None
        log_loss : bool, optional
            If True, loss is computed as `- log(dice_coeff)`. If False, loss is computed as `1 - dice_coeff`, by default False
        from_logits : bool, optional
            If True, assumes input is raw logits. If False, assumes input is probabilities., by default True
        smooth : float, optional
            Smoothness constant for dice coefficient (a), by default 0.0
        ignore_index : Optional[int], optional
            Label that indicates ignored pixels (does not contribute to loss), by default None
        eps : float, optional
            A small epsilon for numerical stability to avoid zero division error (denominator will be always greater or equal to eps), by default 1e-7

        Raises
        ------
        AssertionError
            If the mode is not one of 'binary', 'multiclass', or 'multilabel' and classes are being masked with mode='binary'.
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert (
                mode != BINARY_MODE
            ), "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot(
                    (y_true * mask).to(torch.long), num_classes
                )  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # N, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # N, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(
            y_pred,
            y_true.type_as(y_pred),
            smooth=self.smooth,
            eps=self.eps,
            dims=dims,
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(
        self, output, target, smooth=0.0, eps=1e-7, dims=None
    ) -> torch.Tensor:
        return dice_score(output, target, smooth, eps, dims)


class MultiClassDiceCELoss(nn.Module):
    def __init__(self, weight_ce: float = 1.0, weight_dice: float = 1.0) -> None:
        """Combined Dice and Cross-Entropy loss for multi-class segmentation.

        Combines Dice loss for handling class imbalance and Cross-Entropy loss for
        pixel-wise classification stability, improving segmentation performance.

        Parameters
        ----------
        weight_ce : float, optional
            Weight for Cross-Entropy loss component (default is 1.0).
        weight_dice : float, optional
            Weight for Dice loss component (default is 1.0).
        """

        super(MultiClassDiceCELoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(
        self, inputs: Union[torch.Tensor, List[torch.Tensor]], targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate combined Dice and Cross-Entropy loss.

        Parameters
        ----------
        inputs : torch.Tensor or list of torch.Tensor
            Model predictions. If a list, assumes deep supervision with multiple outputs.
        targets : torch.Tensor
            Ground truth segmentation masks of shape (batch_size, H, W) or
            (batch_size, 1, H, W).
        """

        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        if isinstance(inputs, list):
            loss = 0
            for logits in inputs:
                ce_loss = nn.CrossEntropyLoss()(logits, targets.long())
                probs = F.softmax(logits, dim=1)
                dice_loss = self._dice_loss(probs, targets)
                combined_loss = self.weight_ce * ce_loss + self.weight_dice * dice_loss
                loss += combined_loss
            return loss / len(inputs)
        else:
            ce_loss = nn.CrossEntropyLoss()(inputs, targets.long())
            probs = F.softmax(inputs, dim=1)
            dice_loss = self._dice_loss(probs, targets)
            return self.weight_ce * ce_loss + self.weight_dice * dice_loss

    def _dice_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate Dice loss for all classes.

        Computes Dice loss as 1 - (2 * intersection) / (pred + target) averaged
        over all classes.

        Parameters
        ----------
        probs : torch.Tensor
            Softmax probabilities of shape (batch_size, num_classes, H, W).
        targets : torch.Tensor
            Ground truth labels of shape (batch_size, H, W).
        """

        dice_loss = 0
        num_classes = probs.shape[1]
        for cls in range(num_classes):
            pred_cls = probs[:, cls]
            target_cls = (targets == cls).float()
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum() + 1e-8
            dice_score = (2.0 * intersection) / union
            dice_loss += 1 - dice_score
        return dice_loss / num_classes
