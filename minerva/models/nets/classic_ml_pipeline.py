from sklearn.base import BaseEstimator
from typing import Dict, Optional, Union
from torchmetrics import Metric
import torch
from torch import nn
import lightning as L


class ClassicMLModel(L.LightningModule):
    """
    A PyTorch Lightning module that wraps a classic ML model (e.g. a scikit-learn model)
    and uses it as a head of a neural network. The backbone of the network is frozen and
    the head is trained on the features extracted by the backbone. More complex models,
    that do not follow this pipeline, should not inherit from this class.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: BaseEstimator,
        use_only_train_data: bool = False,
        test_metrics: Optional[Dict[str, Metric]] = None,
    ):
        """
        Initialize the model with the backbone and head. The backbone is frozen and the head
        is trained on the features extracted by the backbone. The head should implement the
        `BaseEstimator` interface. The model can be trained using only the training data or
        using both training and validation data. The test metrics are used to evaluate the
        model during testing. It will be logged using lightning logger at the end of each epoch.

        Parameters
        ----------
        backbone : torch.nn.Module
            The backbone model.
        head : BaseEstimator
            The head model. Usually, a scikit-learn model, like a classifier or regressor that
            implements the `predict` and `fit` methods.
        use_only_train_data : bool, optional
            If `True`, the model will be trained using only the training data, by default False.
            If `False`, the model will be trained using both training and validation data, concatenated.
        test_metrics : Dict[str, Metric], optional
            The metrics to be used during testing, by default None
        """
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.head = head
        self.train_data = []
        self.val_data = []
        self.train_y = []
        self.val_y = []
        self.use_only_train_data = use_only_train_data
        self.tensor1 = torch.tensor(1.0, requires_grad=True)
        self.test_metrics = test_metrics

    def forward(self, x):
        """
        Forward pass of the model. Extracts features from the backbone and predicts the
        target using the head.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The predicted target.
        """
        z = self.backbone(x)
        z = z.flatten(start_dim=1)
        y_pred = self.head.predict(z.cpu())
        return y_pred

    def training_step(self, batch, batch_index):
        """
        Training step of the model. Collects all the training batchs into one variable
        and logs a dummy loss to keep track of the training process.
        """
        self.log("train_loss", self.tensor1)
        if self.current_epoch != 1:
            return self.tensor1

        self.train_data.append(self.backbone(batch[0]).flatten(start_dim=1))
        self.train_y.append(batch[1])
        return self.tensor1

    def on_train_epoch_end(self):
        """
        At the end of the first epoch, the model is trained on the concatenated training
        and validation data. The training data is flattened and the head is trained on it.
        """
        if self.current_epoch != 1:
            return
        if not self.use_only_train_data:
            self.train_data.extend(self.val_data)
            self.train_y.extend(self.val_y)
        self.train_data = torch.concat(self.train_data)
        self.train_data = self.train_data.flatten(start_dim=1).cpu()
        self.train_y = torch.concat(self.train_y).cpu()
        self.head.fit(self.train_data, self.train_y)

    def validation_step(self, batch, batch_index):
        """
        Validation step of the model. Collects all the validation batchs into one variable
        and logs a dummy loss to keep track of the validation process.
        """
        self.log("val_loss", self.tensor1)
        if self.current_epoch != 1:
            return self.tensor1
        self.val_data.append(self.backbone(batch[0]).flatten(start_dim=1))
        self.val_y.append(batch[1])
        return self.tensor1

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Test step of the model.
        """
        x, y = batch
        y_hat = torch.tensor(self.forward(x)).to(self.device)
        for metric_name, metric in self.test_metrics.items():
            metric_value = metric.to(self.device)(y_hat, y)
            self.log(
                f"test_{metric_name}",
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return self.tensor1

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """
        Predict step of the model.
        """
        x, _ = batch
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        return None
