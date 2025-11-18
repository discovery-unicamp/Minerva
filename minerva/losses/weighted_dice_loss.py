import torch
import torch.nn.functional as F


class WeightedDiceLoss(torch.nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0):
        """
        Implements Weighted Dice Loss for multiclass segmentation tasks.

        The Dice Loss is calculated for each class individually and then
        weighted averaged by the class frequencies to compensate for imbalance.

        Dice Loss formula for each class c:
        Dice = (2 * intersection + smooth) / (sum of both + smooth)
        DiceLoss = 1 - Dice

        The weights are calculated inversely proportional to the frequency of the class in the target,
        giving more importance to classes with fewer pixels.

        Note:
        The weights are applied within the intersection and union calculation, by multiplying the terms,
        and not directly as an overall weight on the class loss.

        Parameters
        ----------
        num_classes : int
            total number of classes in the segmentation.
        smooth : float, optional
            value to smooth the calculation and avoid division by zero.
            Default is 1.0.
        """
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Calculates the Weighted Dice Loss.

        Parameters
        ----------
        pred : torch.Tensor
            tensor of raw predictions (logits) with shape (B, C, H, W).
        target : torch.Tensor
            target tensor with integer class labels, shape (B, H, W).

        Returns:
            torch.Tensor: scalar tensor containing the weighted average loss value.
        """
        B, C, H, W = pred.shape
        weights = self.get_weight(target)

        pred = F.softmax(pred, dim=1)
        target = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2)
        target = target.contiguous().view(B, C, H, W)
        total_loss = 0

        for class_idx in range(self.num_classes):
            class_true = target[:, class_idx, ...]
            class_pred = pred[:, class_idx, ...]
            intersection = torch.sum(weights[class_idx] * (class_true * class_pred))
            union = torch.sum(weights[class_idx] * class_true) + torch.sum(
                weights[class_idx] * class_pred
            )
            dice_loss = 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)
            total_loss += dice_loss

        return total_loss / self.num_classes

    def get_weight(self, target: torch.Tensor):
        """
        Calculates weights for each class based on the inverse frequency in the target.

        Parameters
        ----------
        target : torch.Tensor
            target tensor with integer class labels, shape(B, H, W).

        Returns:
            torch.Tensor: 1D tensor with normalized weights for each class, shape(num_classes,).
        """
        cls = torch.arange(self.num_classes).reshape(-1, 1).to(target.device)
        counts = torch.bincount(torch.flatten(target), minlength=self.num_classes)
        cls_num = counts[cls]
        denominator = torch.where(cls_num != 0, cls_num.float(), torch.tensor(1e10))
        alpha = 1 / denominator
        alpha_norm = alpha / alpha.sum()
        return alpha_norm


class BinaryWeightedDiceLoss(torch.nn.Module):
    def __init__(self, smooth: float = 1.0):
        """
        Implements Weighted Dice Loss for binary segmentation.

        Applies sigmoid to predictions to obtain probabilities and calculates
        weights for foreground and background based on pixel frequency
        in the target, to balance the contribution of each class.

        Dice Loss formula:
        dice_loss = 1 - (2 * Σ(w_i * p_i * t_i) + smooth) / (Σ(w_i * p_i) + Σ(w_i * t_i) + smooth)

        Where w_i are the weights per pixel, p_i are the predictions after sigmoid, and t_i are the binary targets.

        Parameters
        ----------
        smooth : float, optional
            value to smooth calculation and avoid division by zero. Default is 1.0.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Calculates the binary Weighted Dice Loss.

        Parameters
        ----------
        pred : torch.Tensor
            tensor of raw predictions (logits) with shape (B, 1, H, W).
        target : torch.Tensor
            target binary tensor (0 or 1), shape (B, 1, H, W).

        Returns:
            torch.Tensor: scalar tensor containing the loss value.
        """
        pred = torch.sigmoid(pred)

        weights = self.get_weight(target)

        intersection = torch.sum(weights * (target * pred))
        union = torch.sum(weights * target) + torch.sum(weights * pred)
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)

        return dice_loss

    def get_weight(self, target: torch.Tensor):
        """
        Calculates per-pixel weights based on the ratio of foreground and background in the target.

        Parameters
        ----------
        target : torch.Tensor
            target binary tensor (0 or 1), shape (B, 1, H, W).

        Returns:
            torch.Tensor: tensor of per-pixel weights, same shape as target.
        """
        num_foreground = torch.sum(target)
        num_background = torch.sum(1 - target)
        total = num_foreground + num_background
        alpha_foreground = num_background / total
        alpha_background = num_foreground / total
        weights = target * alpha_foreground + (1 - target) * alpha_background
        return weights
