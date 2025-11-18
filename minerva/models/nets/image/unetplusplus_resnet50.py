import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import lightning as L
from torchmetrics import Accuracy, F1Score, JaccardIndex, Precision, Recall
from torchvision.models import resnet50, ResNet50_Weights
from minerva.losses.dice import MultiClassDiceCELoss
from typing import Any, Callable, Optional, Tuple, Union, List


class DeepLabV3ResNet50Backbone(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        """Notes
        -----
        The dilation rates are set as follows:
        - layer3: dilation=2, stride=1
        - layer4: dilation=4, stride=1
        This preserves the spatial resolution at H/8, W/8 for c3, c4, and c5.

        References
        ----------
        .. [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
        recognition. In Proceedings of the IEEE conference on computer vision and pattern
        recognition (pp. 770-778).
        """

        super(DeepLabV3ResNet50Backbone, self).__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet50(
            weights=weights, replace_stride_with_dilation=[False, True, True]
        )
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the ResNet50 backbone.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, H, W).
        """

        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c1, c2, c3, c4, c5


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Double convolution block for UNet++ decoder.

        Applies two consecutive 3x3 convolutions, each followed by batch normalization
        and ReLU activation, commonly used in U-Net architectures for feature refinement.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        """

        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the double convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (batch_size, in_channels, H, W).

        """

        return self.conv(x)


class UNetPlusPlusDeepLabV3(nn.Module):
    """UNet++ with DeepLabV3 ResNet50 backbone for semantic segmentation.

    Combines DeepLabV3's multi-scale feature extraction with UNet++'s nested skip
    connections for robust semantic segmentation, particularly suited for seismic
    image segmentation tasks.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels (default is 3 for RGB images).
    num_classes : int, optional
        Number of segmentation classes (default is 6).
    deep_supervision : bool, optional
        If True, enables deep supervision with auxiliary losses (default is True).
    pretrained : bool, optional
        If True, uses ImageNet pre-trained ResNet50 backbone (default is True).

    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        deep_supervision: bool = True,
        pretrained: bool = True,
    ) -> None:
        """Notes
        -----
        The architecture includes:
        - DeepLabV3 ResNet50 backbone with dilated convolutions.
        - UNet++ decoder with nested skip connections for feature refinement.
        - Bilinear upsampling to restore original input resolution.
        - Optional deep supervision for improved training stability.

        References
        ----------
        .. [1] Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2018).
        Unet++: A nested u-net architecture for medical image segmentation. In Deep
        learning in medical image analysis and multimodal learning for clinical decision
        support (pp. 3-11). Springer.
        .. [2] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for
        image recognition. In Proceedings of the IEEE conference on computer vision and
        pattern recognition (pp. 770-778).
        """

        super(UNetPlusPlusDeepLabV3, self).__init__()
        self.deep_supervision = deep_supervision
        filters = [128, 256, 512, 1024]
        self.backbone = DeepLabV3ResNet50Backbone(pretrained=pretrained)
        self.proj0 = nn.Conv2d(256, filters[0], kernel_size=1)
        self.proj1 = nn.Conv2d(512, filters[1], kernel_size=1)
        self.proj2 = nn.Conv2d(1024, filters[2], kernel_size=1)
        self.proj3 = nn.Conv2d(2048, filters[3], kernel_size=1)
        self.up = nn.ModuleList(
            [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                for _ in range(3)
            ]
        )
        self.up_conv = nn.ModuleList(
            [nn.Conv2d(filters[i + 1], filters[i], kernel_size=1) for i in range(3)]
        )
        self.conv0_1 = ConvBlock(2 * filters[0], filters[0])
        self.conv1_1 = ConvBlock(2 * filters[1], filters[1])
        self.conv2_1 = ConvBlock(2 * filters[2], filters[2])
        self.conv0_2 = ConvBlock(3 * filters[0], filters[0])
        self.conv1_2 = ConvBlock(3 * filters[1], filters[1])
        self.conv0_3 = ConvBlock(4 * filters[0], filters[0])
        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through the UNet++ DeepLabV3 network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, H, W).
        """

        input_size = x.shape[2:]
        c1, c2, c3, c4, c5 = self.backbone(x)
        x0_0 = self.proj0(c2)
        x1_0 = self.proj1(c3)
        x2_0 = self.proj2(c4)
        x3_0 = self.proj3(c5)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up_conv[1](x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up_conv[2](x3_0)], dim=1))
        x1_0_up = self._safe_upsample(self.up_conv[0](self.up[0](x1_0)), x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, x1_0_up], dim=1))
        x2_1_up = self._safe_upsample(self.up_conv[1](x2_1), x1_0)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, x2_1_up], dim=1))
        x1_1_up = self._safe_upsample(self.up_conv[0](self.up[0](x1_1)), x0_0)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, x1_1_up], dim=1))
        x1_2_up = self._safe_upsample(self.up_conv[0](self.up[0](x1_2)), x0_0)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, x1_2_up], dim=1))
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output1 = F.interpolate(
                output1, size=input_size, mode="bilinear", align_corners=True
            )
            output2 = F.interpolate(
                output2, size=input_size, mode="bilinear", align_corners=True
            )
            output3 = F.interpolate(
                output3, size=input_size, mode="bilinear", align_corners=True
            )
            return [output1, output2, output3]
        else:
            output = self.final(x0_3)
            output = F.interpolate(
                output, size=input_size, mode="bilinear", align_corners=True
            )
            return output

    def _safe_upsample(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Upsample tensor to match target spatial dimensions.

        Handles cases where spatial dimensions may not align due to rounding errors
        in pooling or upsampling operations.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to be upsampled.
        target : torch.Tensor
            Reference tensor with target spatial dimensions.
        """

        if x.size()[2:] != target.size()[2:]:
            x = F.interpolate(
                x, size=target.size()[2:], mode="bilinear", align_corners=True
            )
        return x


class LitUNetPlusPlusDeepLabV3(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        deep_supervision: bool = True,
        lr: float = 3e-4,
        pretrained: bool = True,
    ) -> None:
        """PyTorch Lightning module for UNet++ with DeepLabV3 backbone.

        Wraps the UNet++ DeepLabV3 model with training, validation, and testing loops,
        optimizer configuration, and metrics for multi-class segmentation.

        Parameters
        ----------
        in_channels : int, optional
            Number of input image channels (default is 3).
        num_classes : int, optional
            Number of segmentation classes (default is 6).
        deep_supervision : bool, optional
            If True, enables deep supervision training (default is True).
        lr : float, optional
            Learning rate for the optimizer (default is 3e-4).
        pretrained : bool, optional
            If True, uses ImageNet pre-trained backbone (default is True).

        Notes
        -----
        Metrics include accuracy, F1-score, mean IoU, precision, and recall, all computed
        using torchmetrics.
        """

        super().__init__()
        self.save_hyperparameters()
        self.model = UNetPlusPlusDeepLabV3(
            in_channels=in_channels,
            num_classes=num_classes,
            deep_supervision=deep_supervision,
            pretrained=pretrained,
        )
        self.loss_fn = MultiClassDiceCELoss()
        self.lr = lr
        self.num_classes = num_classes
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_miou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.test_precision = Precision(task="multiclass", num_classes=num_classes)
        self.test_recall = Recall(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of images of shape (batch_size, in_channels, H, W).
        """

        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step for one batch.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            Batch containing (images, masks).
        batch_idx : int
            Index of the current batch.
        """

        imgs, masks = batch
        preds = self(imgs)
        loss = self.loss_fn(preds, masks)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step for one batch.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            Batch containing (images, masks).
        batch_idx : int
            Index of the current batch.
        """

        imgs, masks = batch
        preds = self(imgs)
        loss = self.loss_fn(preds, masks)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Test step for one batch with metrics evaluation.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            Batch containing (images, masks).
        batch_idx : int
            Index of the current batch.
        """

        imgs, masks = batch
        preds = self(imgs)
        if isinstance(preds, list):
            final_preds = preds[-1]
        else:
            final_preds = preds
        loss = self.loss_fn(preds, masks)
        self.log("test_loss", loss)
        self.test_accuracy(final_preds, masks)
        self.test_f1(final_preds, masks)
        self.test_miou(final_preds, masks)
        self.test_precision(final_preds, masks)
        self.test_recall(final_preds, masks)
        self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test_miou", self.test_miou, on_step=False, on_epoch=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer for training."""

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
