# import os
import torch
import torch.nn as nn
import lightning as L
import torchmetrics
from minerva.models.loaders import LoadableModule
from typing import Dict, Union
    
class HARClassifier(L.LightningModule):
    def __init__(
        self,
        backbone: Union[torch.nn.Module, LoadableModule],
        fc: Union[torch.nn.Module, LoadableModule],
        classifier_lr=5e-4,
        num_classes=6,
        freeze_backbone: bool=False,
    ):
        """
        HARClassifier class that defines the model architecture and training process for
        human activity recognition.

        This classifier combines a backbone feature extractor and a fully connected layer (fc)
        for the final classification task. The backbone can be frozen to prevent weight updates 
        during training.

        Parameters
        ----------
        backbone : Union[torch.nn.Module, LoadableModule]
            Pretrained or custom feature extractor model.
        fc : L.LightningModule
            Fully connected layer to be applied after the backbone.
        device : torch.device, optional
            Device to be used for computations, by default it will be set to "cuda" if available.
        classifier_lr : float, optional
            Learning rate for the classifier's optimizer, by default 5e-4.
        num_classes : int, optional
            Number of output classes, by default 6.
        freeze_backbone : bool, optional
            Whether to freeze the backbone during training, by default False.
        """
        super().__init__()
        self.backbone = backbone
        self.fc = fc
        self.criterion = nn.CrossEntropyLoss()
        self.classifier_lr = classifier_lr
        # Define metrics
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self.train_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        self.freeze_backbone = freeze_backbone  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.fc(x[:, -1, :])
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        # Update metrics
        self.train_accuracy(preds, labels)
        self.train_f1(preds, labels)

        self.log("train_loss", loss)
        self.log("train_accuracy", self.train_accuracy, on_epoch=True, on_step=False)
        self.log("train_f1", self.train_f1, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)

        self.log("val_loss", loss)
        self.log("val_accuracy", self.val_accuracy, on_epoch=True, on_step=False)
        self.log("val_f1", self.val_f1, on_epoch=True, on_step=False)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)

        self.log("test_loss", loss)
        self.log("test_accuracy", self.test_accuracy, on_epoch=True, on_step=False)
        self.log("test_f1", self.test_f1, on_epoch=True, on_step=False)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, _ = batch
        outputs = self(inputs)
        return outputs
    
    def configure_optimizers(self):

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        optimizer = torch.optim.Adam(self.parameters(), lr=self.classifier_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.8)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
