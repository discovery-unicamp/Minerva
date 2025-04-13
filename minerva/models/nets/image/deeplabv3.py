from typing import Any, Dict, Optional, Sequence, Tuple
from collections import OrderedDict
from torch import Tensor, nn, optim
from torchmetrics import Metric
from torchvision.models.resnet import resnet50
from torchvision.models.segmentation.deeplabv3 import ASPP
import torch
from minerva.models.nets.base import SimpleSupervisedModel


class DeepLabV3(SimpleSupervisedModel):
    """A DeeplabV3 with a ResNet50 backbone

    References
    ----------
    Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam.
    "Rethinking Atrous Convolution for Semantic Image Segmentation", 2017
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        pred_head: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
        learning_rate: float = 0.001,
        num_classes: int = 6,
        train_metrics: Optional[Dict[str, Metric]] = None,
        val_metrics: Optional[Dict[str, Metric]] = None,
        test_metrics: Optional[Dict[str, Metric]] = None,
        optimizer: type = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[type] = None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        output_shape: Optional[Tuple[int, ...]] = None,
    ):
        """
        Initializes a DeepLabV3 model.

        Parameters
        ----------
        backbone: Optional[nn.Module]
            The backbone network. Defaults to None, which will use a ResNet50
            backbone.
        pred_head: Optional[nn.Module]
            The prediction head network. Defaults to None, which will use a
            DeepLabV3PredictionHead with specified number of classes.
        loss_fn: Optional[nn.Module]
            The loss function. Defaults to None, which will use a
            CrossEntropyLoss.
        learning_rate: float
            The learning rate for the optimizer. Defaults to 0.001.
        num_classes: int
            The number of classes for prediction. Defaults to 6.
        train_metrics: Optional[Dict[str, Metric]]
            The metrics to be computed during training. Defaults to None.
        val_metrics: Optional[Dict[str, Metric]]
            The metrics to be computed during validation. Defaults to None.
        test_metrics: Optional[Dict[str, Metric]]
            The metrics to be computed during testing. Defaults to None.
        optimizer: type
            Optimizer class to be instantiated. By default, it is set to
            `torch.optim.Adam`. Should be a subclass of
            `torch.optim.Optimizer` (e.g., `torch.optim.SGD`).
        optimizer_kwargs : dict, optional
            Additional kwargs passed to the optimizer constructor.
        lr_scheduler : type, optional
            Learning rate scheduler class to be instantiated. By default, it is
            set to None, which means no scheduler will be used. Should be a
            subclass of `torch.optim.lr_scheduler.LRScheduler` (e.g.,
            `torch.optim.lr_scheduler.StepLR`).
        lr_scheduler_kwargs : dict, optional
            Additional kwargs passed to the scheduler constructor.
        output_shape: Optional[Tuple[int, ...]]
            The output shape of the model. If None, the output shape will be
            the same as the input shape. Defaults to None. This is useful for
            models that require a specific output shape, that is different from
            the input shape.
        """
        backbone = backbone or DeepLabV3Backbone()
        pred_head = pred_head or DeepLabV3PredictionHead(num_classes=num_classes)
        loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.output_shape = output_shape

        super().__init__(
            backbone=backbone,
            fc=pred_head,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        input_shape = self.output_shape or x.shape[-2:]
        h = self.backbone(x)
        if isinstance(h, OrderedDict):
            h = h["out"]
        z = self.fc(h)
        # Upscaling
        return nn.functional.interpolate(
            z, size=input_shape, mode="bilinear", align_corners=False
        )

    def _loss_func(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.loss_fn(y_hat, y.squeeze(1).long())


class DeepLabV3Backbone(nn.Module):
    """A ResNet50 backbone for DeepLabV3"""

    def __init__(self, num_classes: int = 6):
        """
        Initializes the DeepLabV3 model.

        Parameters
        ----------
        num_classes: int
            The number of classes for classification. Default is 6.
        """
        super().__init__()
        RN50model = resnet50(replace_stride_with_dilation=[False, True, True])
        self.RN50model = RN50model

    def freeze_weights(self):
        for param in self.RN50model.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.RN50model.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.RN50model.conv1(x)
        x = self.RN50model.bn1(x)
        x = self.RN50model.relu(x)
        x = self.RN50model.maxpool(x)
        x = self.RN50model.layer1(x)
        x = self.RN50model.layer2(x)
        x = self.RN50model.layer3(x)
        x = self.RN50model.layer4(x)
        # x = self.RN50model.avgpool(x)      # These should be removed for deeplabv3
        # x = torch.RN50model.flatten(x, 1)  # These should be removed for deeplabv3
        # x = self.RN50model.fc(x)           # These should be removed for deeplabv3
        return x


class DeepLabV3PredictionHead(nn.Sequential):
    """The prediction head for DeepLabV3"""

    def __init__(
        self,
        in_channels: int = 2048,
        num_classes: int = 6,
        atrous_rates: Sequence[int] = (12, 24, 36),
    ) -> None:
        """
        Initializes the DeepLabV3 model.

        Parameters
        ----------
        in_channels: int
            Number of input channels. Defaults to 2048.
        num_classes: int
            Number of output classes. Defaults to 6.
        atrous_rates: Sequence[int]
            A sequence of atrous rates for the ASPP module. Defaults to (12, 24, 36).
        """
        super().__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, input) -> Tensor:
        assert input.shape[0] > 1, "Batch size must be greater than 1 due to BatchNorm"
        return super().forward(input)
