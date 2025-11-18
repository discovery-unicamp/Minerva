from typing import Any, Dict, Optional, Sequence, Tuple
import os
from collections import OrderedDict
from torch import Tensor, nn, optim
from torchmetrics import Metric
from torchvision.models import ResNet50_Weights
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
        pretrained: bool = False,
        weights_path: Optional[str] = None,
        train_metrics: Optional[Dict[str, Metric]] = None,
        val_metrics: Optional[Dict[str, Metric]] = None,
        test_metrics: Optional[Dict[str, Metric]] = None,
        optimizer: type = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[type] = None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        output_shape: Optional[Tuple[int, ...]] = None,
        freeze_backbone: bool = False,
        interpolate_mode: Optional[str] = "bilinear",
        flatten: bool = False,
        loss_squeeze: bool = True,
        loss_long: bool = True,
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
        pretrained: bool
            Whether to use pretrained weights. Defaults to False.
        weights_path: Optional[str]
            Path to local pretrained weights file. If provided with pretrained=True,
            loads weights from this path instead of downloading. Defaults to None.
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
        freeze_backbone: bool
            Whether to freeze the backbone weights during training. Defaults to
            False.
        interpolate_mode: Optional[str]
            The interpolation mode to use when upscaling the output to the
            desired output shape. Defaults to "bilinear". Other options include
            "nearest", "bicubic", etc. See PyTorch documentation for
            `torch.nn.functional.interpolate` for all options. Use None to
            disable upscaling.
        flatten: bool
            Whether to flatten the output of the backbone before passing it to
            the prediction head. Defaults to False. Set to True for classification
            tasks where the prediction head is a fully connected layer.
        loss_squeeze: bool
            Whether to squeeze the target tensor in the loss function. Defaults
            to True. This is useful for segmentation tasks where the target tensor
            has a singleton channel dimension (e.g., shape (B, 1, H, W)) and the
            loss function expects shape (B, H, W).
        loss_long: bool
            Whether to convert the target tensor to long type in the loss
            function. Defaults to True. This is useful for classification tasks
            where the target tensor is of integer type.

        """
        backbone = backbone or DeepLabV3Backbone(
            num_classes=num_classes, pretrained=pretrained, weights_path=weights_path
        )
        pred_head = pred_head or DeepLabV3PredictionHead(num_classes=num_classes)
        loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.output_shape = output_shape
        self.interpolate_mode = interpolate_mode
        self.squeeze_loss = loss_squeeze
        self.loss_long = loss_long

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
            freeze_backbone=freeze_backbone,
            flatten=flatten,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass of the DeepLabV3 model.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, channels, height, width)

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, num_classes, height, width)
        """
        x = x.float()
        input_shape = self.output_shape or x.shape[-2:]
        h = self.backbone(x)
        if isinstance(h, OrderedDict):
            h = h["out"]

        if self.flatten:
            x = x.reshape(x.size(0), -1)
        if self.adapter is not None:
            x = self.adapter(x)

        z = self.fc(h)
        # Upscaling
        if self.interpolate_mode is None:
            return z
        return nn.functional.interpolate(
            z, size=input_shape, mode=self.interpolate_mode, align_corners=False
        )

    def _loss_func(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """Computes the loss between predictions and ground truth.

        Parameters
        ----------
        y_hat : Tensor
            Predicted tensor of shape (batch_size, num_classes, height, width)
        y : Tensor
            Ground truth tensor of shape (batch_size, 1, height, width)
        """
        if self.squeeze_loss:
            y = y.squeeze(1)  # Remove channel dim if singleton
        if self.loss_long:
            y = y.long()

        return self.loss_fn(y_hat, y)


class DeepLabV3Backbone(nn.Module):
    """A ResNet50 backbone for DeepLabV3"""

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = False,
        weights_path: Optional[str] = None,
    ):
        """
        Initializes the DeepLabV3 backbone model.

        Parameters
        ----------
        num_classes: int
            The number of classes for classification. Default is 6.
        pretrained: bool
            Whether to use pretrained weights. If True and weights_path is None,
            will attempt to download ImageNet pretrained weights. Default is False.
        weights_path: Optional[str]
            Path to local pretrained weights file. If provided with pretrained=True,
            loads weights from this path instead of downloading. Default is None.
        """
        super().__init__()

        if pretrained and weights_path is not None:
            # Validate file path exists
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found: {weights_path}")

            # Load from local weights file
            RN50model = resnet50(replace_stride_with_dilation=[False, True, True])

            state_dict = torch.load(weights_path, map_location="cpu")
            # Handle different weight file formats
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

            # Filter out classifier weights if they exist (fc layer)
            # since we don't use them in the backbone
            filtered_state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith("fc.")
            }

            # Load the filtered state dict
            missing_keys, unexpected_keys = RN50model.load_state_dict(
                filtered_state_dict, strict=False
            )

            if missing_keys:
                print(
                    f"Warning: Missing keys when loading pretrained weights: {missing_keys}"
                )
            if unexpected_keys:
                print(
                    f"Warning: Unexpected keys when loading pretrained weights: {unexpected_keys}"
                )

            print(f"Successfully loaded pretrained weights from {weights_path}")

        elif pretrained and weights_path is None:
            # Use torchvision's pretrained weights (requires internet)
            RN50model = resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V1,
                replace_stride_with_dilation=[False, True, True],
            )
            print("Successfully loaded ImageNet pretrained weights from torchvision")
        else:
            # No pretrained weights, random initialization
            RN50model = resnet50(replace_stride_with_dilation=[False, True, True])

        self.RN50model = RN50model

    def freeze_weights(self):
        """Freezes all parameters in the backbone, making them non-trainable."""
        for param in self.RN50model.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        """Unfreezes all parameters in the backbone, making them trainable."""
        for param in self.RN50model.parameters():
            param.requires_grad = True

    def forward(self, x):
        """Performs the forward pass of the backbone.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, channels, height, width)

        Returns
        -------
        Tensor
            Feature map tensor from the ResNet50 backbone
        """
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
        Initializes the DeepLabV3 prediction head.

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
        """Performs the forward pass of the prediction head.

        Parameters
        ----------
        input : Tensor
            Input tensor from the backbone

        Returns
        -------
        Tensor
            Output tensor with class predictions
        """
        assert input.shape[0] > 1, "Batch size must be greater than 1 due to BatchNorm"
        return super().forward(input)


class DeepLabV3RegressionHead(nn.Sequential):
    """Regression head for DeepLabV3 (continuous per-pixel/voxel prediction)."""

    def __init__(
        self,
        in_channels: int = 2048,
        out_channels: int = 1,
        atrous_rates: Sequence[int] = (12, 24, 36),
    ):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels from the backbone (typically 2048 for ResNet50).
        out_channels : int
            Number of output channels (1 for single regression target).
        atrous_rates : Sequence[int]
            Atrous (dilation) rates for ASPP.
        """
        super().__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        """Forward pass for regression head."""
        assert x.shape[0] > 1, "Batch size must be > 1 due to BatchNorm"
        return super().forward(x)
