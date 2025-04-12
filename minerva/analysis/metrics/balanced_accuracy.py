import torch
import torchmetrics
from torchmetrics import Metric
from torchmetrics.functional import confusion_matrix
import warnings


class BalancedAccuracy(Metric):
    def __init__(self, num_classes: int, task: str, adjusted: bool = False):
        """
        Compute the balanced accuracy.

        The balanced accuracy in binary, multiclass, and multilabel classification problems
        deals with imbalanced datasets. It is defined as the average of recall obtained on each class.

        Parameters
        ----------
        num_classes : int
            The number of classes in the target data.

        task : str
            The type of classification task, should be one of 'binary' or 'multiclass'

        adjusted : bool, optional (default=False)
            When true, the result is adjusted for chance, so that random performance would score 0,
            while keeping perfect performance at a score of 1.

        Attributes
        ----------
        confmat : torch.Tensor
            Confusion matrix to keep track of true positives, false positives, true negatives, and false negatives.

        Examples
        --------
        >>> y_true = torch.tensor([0, 1, 0, 0, 1, 0])
        >>> y_pred = torch.tensor([0, 1, 0, 0, 0, 1])
        >>> metric = BalancedAccuracy(num_classes=2, task='binary')
        >>> metric(y_pred, y_true)
        0.625
        """
        super().__init__()
        self.num_classes = num_classes
        self.adjusted = adjusted
        self.task = task
        self.add_state(
            "confmat",
            default=torch.zeros((num_classes, num_classes)),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.confmat += confusion_matrix(
            preds, target, num_classes=self.num_classes, task=self.task
        )

    def compute(self):
        with torch.no_grad():
            per_class = torch.diag(self.confmat) / self.confmat.sum(dim=1)
            if torch.any(torch.isnan(per_class)):
                warnings.warn(f"y_pred contains nan values and not all classes passed")
                per_class = per_class[~torch.isnan(per_class)]  # Filter out NaN values
            if len(per_class) == 0:
                return torch.tensor(0.0)  # Return 0 if no valid classes remain
            score = torch.mean(per_class)
            if self.adjusted:
                n_classes = len(per_class)
                chance = 1 / n_classes
                score -= chance
                score /= 1 - chance
            return score
